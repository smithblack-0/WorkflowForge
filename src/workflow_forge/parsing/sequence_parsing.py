"""
UDPL Sequence Parser

This module parses sequences from TOML data, validates them according to the UDPL
specification, and builds linked zcp node chains using a provided block parser.

The sequence parser:
1. Validates that each sequence in config exists in TOML
2. Ensures each sequence resolves to a list
3. Calls the block parser for each block in the sequence
4. Chains the returned zcp nodes into linked lists
5. Returns a dictionary mapping sequence names to zcp chain heads
"""

from typing import Dict, List, Any, Callable
from ..zcp.nodes import ZCPNode
from .config_parsing import Config


class SequenceParseError(Exception):
    """Exception raised when sequence parsing fails."""
    pass

def parse_sequences(
    toml_data: Dict[str, Any],
    config: Config,
    block_parser: Callable[[Dict[str, Any], Config, str, int], ZCPNode]
) -> Dict[str, ZCPNode]:
    """
    Parse all sequences from TOML data into zcp node chains.

    Args:
        toml_data: Raw parsed TOML data
        config: Validated UDPL configuration
        block_parser: Callable that accepts (block_data, config, sequence_name, block_index) and returns ZCPNode

    Returns:
        Dictionary mapping sequence names to the head zcp node of each chain

    Raises:
        SequenceParseError: If sequence validation or parsing fails
    """
    sequence_chains = {}

    for sequence_name in config.sequences:
        try:
            # Validate sequence exists at TOML top level
            if sequence_name not in toml_data:
                raise SequenceParseError(
                    f"Sequence '{sequence_name}' declared in config but not found in TOML data"
                )

            sequence_data = toml_data[sequence_name]

            # Validate sequence resolves to a list
            if not isinstance(sequence_data, list):
                raise SequenceParseError(
                    f"Sequence '{sequence_name}' must resolve to a list, got {type(sequence_data).__name__}"
                )

            # Handle empty sequences
            if len(sequence_data) == 0:
                raise SequenceParseError(f"Sequence '{sequence_name}' cannot be empty")

            # Parse each block and build the zcp chain
            chain_head = None
            current_tail = None

            for block_index, block_data in enumerate(sequence_data):
                try:
                    # Validate block data is a dictionary
                    if not isinstance(block_data, dict):
                        raise SequenceParseError(
                            f"Block {block_index} in sequence '{sequence_name}' must be a dictionary, "
                            f"got {type(block_data).__name__}"
                        )

                    # Call the block parser to get the head zcp node for this block
                    block_head = block_parser(block_data, config, sequence_name, block_index)

                    # Validate block parser returned a zcp node
                    if not isinstance(block_head, ZCPNode):
                        raise SequenceParseError(
                            f"Block parser must return a ZCPNode, got {type(block_head).__name__}"
                        )

                    # Set up the chain head if this is the first block
                    if chain_head is None:
                        chain_head = block_head
                        current_tail = block_head.get_last_node()
                    else:
                        # Link the previous chain tail to this block's head
                        current_tail.next_zone = block_head
                        current_tail = block_head.get_last_node()

                except Exception as e:
                    # Re-raise with more context about which block failed
                    if isinstance(e, SequenceParseError):
                        raise e
                    else:
                        raise SequenceParseError(
                            f"Error parsing block {block_index} in sequence '{sequence_name}': {str(e)}"
                        ) from e

            # Store the completed chain
            sequence_chains[sequence_name] = chain_head

        except Exception as e:
            # Re-raise with more context about which sequence failed
            if isinstance(e, SequenceParseError):
                raise e
            else:
                raise SequenceParseError(
                    f"Error parsing sequence '{sequence_name}': {str(e)}"
                ) from e

    return sequence_chains