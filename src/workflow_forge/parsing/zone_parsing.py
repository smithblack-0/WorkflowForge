"""
UDPL Zone Parser

The zone parser is responsible for converting individual ZoneInfo structures into ZCPNode objects.
This is the final stage of the UDPL parsing pipeline that handles:
- Flow control token validation
- Placeholder syntax validation
- Resource binding validation
- zcp node creation with proper callbacks
"""

import re
import string
import warnings
from typing import Dict, Any, List, Optional, Callable
from ..zcp.nodes import ZCPNode
from ..resources import AbstractResource
from .config_parsing import Config
from .block_parsing import ZoneInfo


class ZoneParseError(Exception):
    """Exception raised when zone parsing fails."""
    pass


def parse_zone(zone_info: ZoneInfo, config: Config) -> ZCPNode:
    """
    Parse a single zone into a ZCPNode.

    Args:
        zone_info: Zone information from block parser including block data
        config: Validated UDPL configuration

    Returns:
        ZCPNode: Configured zcp node for this zone

    Raises:
        ZoneParseError: If zone validation or processing fails
    """
    try:
        # Validate flow control tokens aren't in teacher-forced text without escaping
        validate_flow_control_safety(zone_info.zone_text, config, zone_info)

        # Extract and validate placeholders from zone text
        placeholders = extract_placeholders(zone_info.zone_text)

        # Build resource specs for placeholders from block data
        resource_specs = build_resource_specs(placeholders, zone_info.block_data, zone_info.sequence_name)

        # Create construction callback for this zone
        construction_callback = create_construction_callback(
            zone_info.zone_text,
            resource_specs
        )

        # Create the zcp node
        return ZCPNode(
            # Data payload features
            sequence=zone_info.sequence_name,
            block=zone_info.block_index,
            resource_specs=resource_specs,
            construction_callback=construction_callback,
            raw_text=zone_info.zone_text,
            zone_advance_str=zone_info.advance_token,
            tags=zone_info.tags,
            timeout=zone_info.max_gen_tokens,
        )

    except Exception as e:
        raise ZoneParseError(
            f"Error parsing zone {zone_info.zone_index} in sequence '{zone_info.sequence_name}', "
            f"block {zone_info.block_index}: {str(e)}"
        ) from e


def validate_flow_control_safety(zone_text: str, config: Config, zone_info: ZoneInfo) -> None:
    """
    Validate flow control usage in teacher-forced text and warn about likely mistakes.

    Args:
        zone_text: The raw text content of the zone
        config: UDPL configuration
        zone_info: Zone information for error context

    Raises:
        ZoneParseError: If unescaped flow control tokens are found
    """
    # Check if control token is present
    if config.control_pattern in zone_text:
        # Check if escape token is also present
        escaped_text, _ = _escape_text(zone_text, config)
        if config.control_pattern in escaped_text:
            msg = (
                f"In block {zone_info.block_index} of sequence '{zone_info.sequence_name}', "
                f"zone was detected with '{config.control_pattern}' but no '{config.escape_patterns}' tokens. "
                f"Do you mean to teacher-force flow control? If so, this will trigger immediate flow control during prompting."
            )
            warnings.warn(msg, UserWarning)




def _escape_text(text: str, config: Config) -> tuple[str, Dict[str, str]]:
    """
    Find and replace escaped regions with placeholders using sequential token processing.

    Returns:
        tuple: (processed_text_with_placeholders, escape_content_lookup)
    """
    escape_open = config.escape_patterns[0]
    escape_close = config.escape_patterns[1]

    # Create pattern for escape tokens only
    pattern = "|".join(re.escape(token) for token in config.escape_patterns)

    # Split into tokens and text pieces
    splits = re.split(f"({pattern})", text)

    if len(splits) < 2:
        return text, {}  # No special tokens found

    # Extract tokens and text pieces
    # splits = [text0, token0, text1, token1, ..., textN]
    tokens = splits[1::2]  # All the tokens
    text_pieces = splits[0::2]  # All the text pieces

    # Process tokens sequentially
    substitutions = {}
    stack_count = 0
    substitution_num = 0
    text_capturing = ""
    cleaned_text = [text_pieces[0]]  # Start with first text piece

    # Zip tokens with their following text pieces
    text_after_tokens = text_pieces[1:]
    for token, text_after in zip(tokens, text_after_tokens):

        if token == escape_open:
            stack_count += 1
            if stack_count == 1:  # Just opened top-level escape
                text_capturing = ""
        elif token == escape_close:
            if stack_count == 0:
                raise ZoneParseError("Found escape close token without matching open token")
            stack_count -= 1
            if stack_count == 0:  # Just closed top-level escape
                placeholder = f"__ESCAPED_{substitution_num}__"
                substitutions[placeholder] = text_capturing + config.escape_patterns[1]
                cleaned_text.append(placeholder)
                cleaned_text.append(text_after)
                substitution_num += 1
                continue  # Skip normal processing for this token

        # If we're inside an escape region, accumulate content
        if stack_count > 0:
            text_capturing += token + text_after
        else:
            # Normal token - add to result
            cleaned_text.append(token)
            cleaned_text.append(text_after)

    # Check for unclosed escapes
    if stack_count > 0:
        raise ZoneParseError("Found unclosed escape regions")

    processed_text = "".join(cleaned_text)
    return processed_text, substitutions


def extract_placeholders(zone_text: str) -> List[str]:
    """
    Extract placeholder names from zone text using string.Formatter.

    Args:
        zone_text: The raw text content that may contain {placeholder} syntax

    Returns:
        List of unique placeholder names found in the text

    Raises:
        ZoneParseError: If placeholder syntax is malformed
    """
    formatter = string.Formatter()
    placeholders = []

    try:
        # Use string.Formatter to properly parse placeholders
        for literal_text, field_name, format_spec, conversion in formatter.parse(zone_text):
            if field_name is not None:  # field_name is None for literal text
                if field_name not in placeholders:
                    placeholders.append(field_name)

    except ValueError as e:
        raise ZoneParseError(f"Malformed placeholder syntax: {str(e)}") from e

    return placeholders


def build_resource_specs(placeholders: List[str], block_data: Dict[str, Any], sequence_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Build resource specifications for placeholders from block data.

    Args:
        placeholders: List of placeholder names found in zone text
        block_data: Full block data from TOML
        sequence_name: Name of sequence for error messages

    Returns:
        Dictionary mapping placeholder names to their resource specifications

    Raises:
        ZoneParseError: If resource specs are missing or invalid
    """
    resource_specs = {}

    for placeholder in placeholders:
        # Look for [sequence.placeholder] section in block data
        if placeholder not in block_data:
            raise ZoneParseError(
                f"Missing resource specification for placeholder '{placeholder}' - "
                f"expected section '{sequence_name}.{placeholder}' in TOML"
            )

        placeholder_section = block_data[placeholder]

        # Validate placeholder section structure
        if not isinstance(placeholder_section, dict):
            raise ZoneParseError(
                f"Resource specification for placeholder '{placeholder}' must be a dictionary"
            )

        # Validate required 'name' field
        if 'name' not in placeholder_section:
            raise ZoneParseError(
                f"Resource specification for placeholder '{placeholder}' missing required 'name' field"
            )

        resource_name = placeholder_section['name']
        if not isinstance(resource_name, str):
            raise ZoneParseError(
                f"Resource 'name' for placeholder '{placeholder}' must be a string"
            )

        # Validate optional 'arguments' field
        arguments = placeholder_section.get('arguments')
        if arguments is not None and not isinstance(arguments, dict):
            raise ZoneParseError(
                f"Resource 'arguments' for placeholder '{placeholder}' must be a dictionary"
            )

        # Validate optional 'type' field, default to "standard"
        resource_type = placeholder_section.get('type', 'standard')
        if not isinstance(resource_type, str):
            raise ZoneParseError(
                f"Resource 'type' for placeholder '{placeholder}' must be a string"
            )
        if resource_type not in ('standard', 'custom', 'argument'):
            raise ZoneParseError(
                f'Resource "type" for placeholder "{placeholder}" must be in "standard", "custom", or "argument"'
            )

        # Store the resource spec
        resource_specs[placeholder] = {
            'name': resource_name,
            'arguments': arguments,
            'type': resource_type
        }

    return resource_specs


def create_construction_callback(zone_text: str,
                                 resource_specs: Dict[str, Dict[str, Any]]
                                 )->Callable[[Dict[str, AbstractResource]], str]:
    """
    Create a construction callback that resolves placeholders and returns final text.

    Args:
        zone_text: The raw zone text with placeholders
        resource_specs: Dictionary mapping placeholder names to resource specifications

    Returns:
        Callable that accepts a resource dictionary and returns the final constructed text
    """
    def construction_callback(resources: Dict[str, AbstractResource]) -> str:
        """
        Resolve all placeholders using the provided resources and return final text.

        Args:
            resources: Dictionary mapping resource names to resource objects

        Returns:
            Final constructed text with all placeholders resolved

        Raises:
            ValueError: If required resources are missing
        """
        resolved_values = {}

        for placeholder, spec in resource_specs.items():
            resource_name = spec['name']
            arguments = spec['arguments']

            # Check if required resource is available
            if resource_name not in resources:
                raise ValueError(
                    f"Required resource '{resource_name}' for placeholder '{placeholder}' not found"
                )

            resource = resources[resource_name]

            # Call the resource with arguments
            try:
                if arguments is not None:
                    resolved_value = resource(**arguments)
                else:
                    resolved_value = resource()

                resolved_values[placeholder] = resolved_value

            except Exception as e:
                raise ValueError(
                    f"Error calling resource '{resource_name}' for placeholder '{placeholder}': {str(e)}"
                ) from e

        # Format the zone text with resolved values
        try:
            return zone_text.format(**resolved_values)
        except KeyError as e:
            raise ValueError(f"Placeholder {str(e)} found in text but no resource specification provided") from e
        except Exception as e:
            raise ValueError(f"Error formatting zone text: {str(e)}") from e

    return construction_callback