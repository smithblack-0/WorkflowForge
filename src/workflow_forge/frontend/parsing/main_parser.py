"""
Main UDPL Parser

This module provides the main entry points for parsing UDPL files and folders.
It orchestrates the entire parsing pipeline from TOML data to zcp node chains.
"""

from pathlib import Path
from typing import Dict, Any, Tuple

import toml

from .block_parsing import parse_block
from .config_parsing import Config, parse_config
from .sequence_parsing import parse_sequences
from .zone_parsing import parse_zone
from ...zcp.nodes import ZCPNode


class UDPLParseError(Exception):
    """Exception raised when UDPL parsing fails."""
    pass


def parse_udpl_file(file_path: str) -> Tuple[Dict[str, ZCPNode], Config]:
    """
    Parse a single UDPL TOML file into zcp node chains.

    Args:
        file_path: Path to the UDPL TOML file

    Returns:
        Tuple of (sequences_dict, config) where sequences_dict maps
        sequence names to zcp node chain heads

    Raises:
        UDPLParseError: If file loading or parsing fails
    """
    try:
        # Load the TOML file
        with open(file_path, 'r') as f:
            toml_data = toml.load(f)

        return _parse(toml_data)

    except FileNotFoundError:
        raise UDPLParseError(f"UDPL file not found: {file_path}")
    except toml.TomlDecodeError as e:
        raise UDPLParseError(f"Invalid TOML syntax in {file_path}: {str(e)}") from e
    except Exception as e:
        raise UDPLParseError(f"Error parsing UDPL file {file_path}: {str(e)}") from e


def parse_udpl_folder(folder_path: str) -> Tuple[Dict[str, ZCPNode], Config]:
    """
    Parse all UDPL TOML files in a folder into zcp node chains.

    Args:
        folder_path: Path to folder containing UDPL TOML files

    Returns:
        Tuple of (sequences_dict, config) where sequences_dict maps
        sequence names to zcp node chain heads

    Raises:
        UDPLParseError: If folder loading, merging, or parsing fails
    """
    try:
        folder = Path(folder_path)
        if not folder.exists():
            raise UDPLParseError(f"Folder not found: {folder_path}")
        if not folder.is_dir():
            raise UDPLParseError(f"Path is not a directory: {folder_path}")

        # Find all TOML files in the folder
        toml_files = list(folder.glob("*.toml"))
        if not toml_files:
            raise UDPLParseError(f"No TOML files found in folder: {folder_path}")

        # Parse each TOML file
        all_toml_data = {}
        file_sources = {}  # Track which file each key came from

        for toml_file in toml_files:
            try:
                with open(toml_file, 'r') as f:
                    file_data = toml.load(f)

                # Check for collisions with previously loaded files
                _check_for_collisions(file_data, all_toml_data, file_sources, str(toml_file))

                # Merge this file's data
                all_toml_data.update(file_data)

                # Track sources for all keys in this file
                for key in file_data.keys():
                    file_sources[key] = str(toml_file)

            except toml.TomlDecodeError as e:
                raise UDPLParseError(f"Invalid TOML syntax in {toml_file}: {str(e)}") from e

        return _parse(all_toml_data)

    except Exception as e:
        if isinstance(e, UDPLParseError):
            raise e
        else:
            raise UDPLParseError(f"Error parsing UDPL folder {folder_path}: {str(e)}") from e


def _check_for_collisions(new_data: Dict[str, Any],
                         existing_data: Dict[str, Any],
                         file_sources: Dict[str, str],
                         current_file: str) -> None:
    """
    Check for key collisions between TOML files.

    Args:
        new_data: Data from current file being processed
        existing_data: Previously accumulated data
        file_sources: Map of keys to their source files
        current_file: Path of current file being processed

    Raises:
        UDPLParseError: If collisions are found
    """
    collisions = []

    for key in new_data.keys():
        if key in existing_data:
            source_file = file_sources.get(key, "unknown file")
            collisions.append(f"Key '{key}' defined in both {source_file} and {current_file}")

    if collisions:
        collision_msg = "\n".join(collisions)
        raise UDPLParseError(f"Key collisions found between TOML files:\n{collision_msg}")


def _parse(toml_data: Dict[str, Any]) -> Tuple[Dict[str, ZCPNode], Config]:
    """
    Parse validated TOML data into zcp node chains.

    This is the core parsing logic that orchestrates all parsing stages.

    Args:
        toml_data: Validated and merged TOML data

    Returns:
        Tuple of (sequences_dict, config) where sequences_dict maps
        sequence names to zcp node chain heads

    Raises:
        UDPLParseError: If parsing fails at any stage
    """
    try:
        # Step 1: Parse and validate the config
        config = parse_config(toml_data)

        # Step 2: Parse each sequence listed in the config
        sequences_dict = parse_sequences(
            toml_data=toml_data,
            config=config,
            block_parser=_create_block_parser()
        )

        # Step 3: Validate that all sequences were found and parsed
        missing_sequences = set(config.sequences) - set(sequences_dict.keys())
        if missing_sequences:
            missing_list = ", ".join(sorted(missing_sequences))
            raise UDPLParseError(f"Sequences declared in config but not found: {missing_list}")

        return sequences_dict, config

    except Exception as e:
        if isinstance(e, UDPLParseError):
            raise e
        else:
            raise UDPLParseError(f"Error during UDPL parsing: {str(e)}") from e


def _create_block_parser():
    """
    Create a block parser function for use by the sequence parser.

    This function creates the block parser with the zone parser properly configured.

    Returns:
        Block parser function that can be passed to parse_sequences
    """
    def block_parser_with_zone_parser(block_data: Dict[str, Any], config: Config) -> ZCPNode:
        """
        Parse a block using the zone parser.

        Args:
            block_data: Raw block data from TOML
            config: Validated UDPL configuration

        Returns:
            ZCPNode chain head for this block
        """
        # The sequence parser will call this with sequence_name and block_index
        # But we need to extract that from the calling context
        # This is a bit of a hack - we'll let the sequence parser handle
        # the calling pattern properly
        pass

    # Actually, let's return a simpler approach - create a closure that captures zone parser
    def configured_block_parser(block_data: Dict[str, Any],
                              config: Config,
                              sequence_name: str,
                              block_index: int) -> ZCPNode:
        """
        Configured block parser that uses our zone parser.

        Args:
            block_data: Raw block data from TOML
            config: Validated UDPL configuration
            sequence_name: Name of sequence this block belongs to
            block_index: Index of this block within the sequence

        Returns:
            ZCPNode chain head for this block
        """
        return parse_block(
            block_data=block_data,
            config=config,
            sequence_name=sequence_name,
            block_index=block_index,
            zone_parser=parse_zone
        )

    return configured_block_parser