"""
UDPL Block Parser

The block parser's single responsibility is to break a block into its constituent zones
and orchestrate their processing. This includes:
- Validating block structure
- Parsing text into zone structures
- Handling repetitions (repeats/tagset)
- Creating ZoneInfo for each actual zone
- Calling zone parser for each zone
- Chaining returned zcp nodes into linked list
- Returning the head of the chain
"""

from typing import Dict, Any, List, Callable
from dataclasses import dataclass
import re
import warnings
from ..zcp.nodes import ZCPNode
from .config_parsing import Config


class BlockParseError(Exception):
    """Exception raised when block parsing fails."""
    pass


@dataclass
class ZoneInfo:
    """Structure passed to zone processor containing zone-specific information."""
    advance_token: str  # Token that triggers advancement from this zone
    zone_text: str      # Raw text content for this zone (may include placeholders)
    tags: List[str]     # Tags that apply to this zone
    sequence_name: str  # Name of the sequence this zone belongs to
    block_index: int    # Index of the block within the sequence
    zone_index: int     # Index of this zone within the block
    max_gen_tokens: int # Maximum tokens this zone can generate
    block_data: Dict[str, Any]  # Full block data for resource spec lookup

# Type alias for the zone parser function signature
ZoneParser = Callable[[ZoneInfo, Config], ZCPNode]


def parse_block(
    block_data: Dict[str, Any],
    config: Config,
    sequence_name: str,
    block_index: int,
    zone_parser: ZoneParser
) -> ZCPNode:
    """
    Parse a single UDPL block into a zcp node chain.

    Takes a block definition and expands it into all constituent zones,
    handling repetitions, then processes each zone individually and
    chains the results into one linked list.

    Args:
        block_data: Raw block data from TOML parsing
        config: Validated UDPL configuration
        sequence_name: Name of the sequence this block belongs to
        block_index: Index of this block within the sequence
        zone_parser: Function that processes individual zones into zcp nodes

    Returns:
        ZCPNode: Head of the zcp chain for this block

    Raises:
        BlockParseError: If block validation or processing fails
    """
    try:
        # Validate block structure
        validate_block_structure(block_data, config)

        # Parse text into base zone structures
        zone_structures = parse_text_into_zones(block_data['text'], config)

        # Validate and resolve tags for all repetitions
        tags_list = resolve_and_validate_tags(block_data, config)

        # Get max generation tokens
        max_gen_tokens = block_data.get('max_gen_tokens', config.default_max_token_length)

        # Replicate zone structures for each repetition and process each
        all_zone_nodes = []
        overall_zone_index = 0

        for rep_index, tags_for_rep in enumerate(tags_list):
            # Validate tags match zones for this repetition
            if len(tags_for_rep) != len(zone_structures):
                raise BlockParseError(
                    f"Tags list length ({len(tags_for_rep)}) does not match number of zones ({len(zone_structures)}) "
                    f"in sequence '{sequence_name}', block {block_index}, repetition {rep_index}"
                )

            # Create fresh zone structures for this repetition (for independent resource resolution)
            rep_zone_structures = [zone_struct.copy() for zone_struct in zone_structures]

            # Process each zone in this repetition
            for zone_idx, (zone_struct, zone_tags) in enumerate(zip(rep_zone_structures, tags_for_rep)):
                zone_info = ZoneInfo(
                    advance_token=zone_struct['advance_token'],
                    zone_text=zone_struct['zone_text'],
                    tags=zone_tags,
                    sequence_name=sequence_name,
                    block_index=block_index,
                    zone_index=overall_zone_index,
                    max_gen_tokens=max_gen_tokens,
                    block_data=block_data  # Pass full block data for resource spec access
                )

                # Call zone parser to get zcp node
                zone_node = zone_parser(zone_info, config)
                if not isinstance(zone_node, ZCPNode):
                    raise BlockParseError(
                        f"Zone parser must return ZCPNode, got {type(zone_node).__name__} "
                        f"(sequence '{sequence_name}', block {block_index}, zone {zone_idx}, repetition {rep_index})"
                    )

                all_zone_nodes.append(zone_node)
                overall_zone_index += 1

        # Chain all zones together into linked list
        if not all_zone_nodes:
            raise BlockParseError(f"No zones produced from block in sequence '{sequence_name}', block {block_index}")

        head = all_zone_nodes[0]
        for i in range(len(all_zone_nodes) - 1):
            all_zone_nodes[i].next_zone = all_zone_nodes[i + 1]

        return head

    except Exception as e:
        raise BlockParseError(f"Error parsing block {block_index} in sequence '{sequence_name}': {str(e)}") from e


## SANITY CHECKS
#
# Initial checks that we do not have malformed block toml in the
# first place. Not all validation happens here, but quick initial
# checks do

def validate_block_structure(block_data: Dict[str, Any], config: Config) -> None:
    """
    Validate block-level structure, required fields, types, and mutual exclusivity rules.

    Args:
        block_data: Raw block data from TOML
        config: UDPL configuration

    Raises:
        BlockParseError: If validation fails
    """
    # Check required fields exist
    if 'text' not in block_data:
        raise BlockParseError("Block missing required 'text' field")
    if 'tags' not in block_data and 'tagset' not in block_data:
        raise BlockParseError("Block must have either 'tags' or 'tagset' field")

    # Check field types
    if not isinstance(block_data['text'], str):
        raise BlockParseError("Block 'text' field must be a string")

    # Check mutual exclusivity rules
    _validate_tags_tagset_repeats_exclusivity(block_data)

    # Check optional field types
    if 'repeats' in block_data:
        _validate_repeats_field(block_data['repeats'])
    if 'max_gen_tokens' in block_data:
        _validate_max_gen_tokens_field(block_data['max_gen_tokens'])


def _validate_tags_tagset_repeats_exclusivity(block_data: Dict[str, Any]) -> None:
    """Check tags/tagset/repeats mutual exclusivity rules."""
    has_tags = 'tags' in block_data
    has_tagset = 'tagset' in block_data
    has_repeats = 'repeats' in block_data

    if has_tags and has_tagset:
        raise BlockParseError("Block cannot have both 'tags' and 'tagset' fields")
    if has_tagset and has_repeats:
        raise BlockParseError("Block cannot have both 'tagset' and 'repeats' fields")


def _validate_repeats_field(repeats: Any) -> None:
    """Validate repeats field type and value."""
    if not isinstance(repeats, int):
        raise BlockParseError("'repeats' field must be an integer")
    if repeats <= 0:
        raise BlockParseError("'repeats' field must be greater than 0")
    if repeats > 100:
        warnings.warn( (
            f"Large number of repetitions ({repeats}) may cause memory or parsing issues. "
            f"Are you sure you want that many repeats without flow control? "),
            UserWarning
        )


def _validate_max_gen_tokens_field(max_gen_tokens: Any) -> None:
    """Validate max_gen_tokens field type and value."""
    if not isinstance(max_gen_tokens, int):
        raise BlockParseError("'max_gen_tokens' field must be an integer")
    if max_gen_tokens <= 0:
        raise BlockParseError("'max_gen_tokens' field must be greater than 0")


### zone text breakup ###
#
# Break up text into zone and advancement token information. This
# should corrolate one-to-one later on with the tagging sets. Notably,
# this process does return an extra zone that will have no associated tags,
# corrolated with receiving the first prompt token.

def parse_text_into_zones(text: str, config: Config) -> List[Dict[str, str]]:
    """
    Parse block text into zone structures using regex to split on zone tokens.
    Handles hierarchical escape regions by preprocessing them into placeholders.

    Args:
        text: Raw text from the block
        config: UDPL configuration

    Returns:
        List of zone structures with 'advance_token' and 'zone_text' keys

    Raises:
        BlockParseError: If text validation fails
    """
    processed_text, escape_lookup = _escape_text(text, config)
    zone_structures = _parse_zones_from_text(processed_text, config)
    return _restore_escaped_content(zone_structures, escape_lookup)


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
                raise BlockParseError("Found escape close token without matching open token")
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
        raise BlockParseError("Found unclosed escape regions")

    processed_text = "".join(cleaned_text)
    return processed_text, substitutions

def _parse_zones_from_text(text: str, config: Config) -> List[Dict[str, str]]:
    """
    Parse zone structures from text that has escaped regions replaced with placeholders.
    """

    # Use regex to split on zone tokens only (no escape tokens since they're already processed)
    zone_pattern = "|".join(re.escape(token) for token in config.zone_patterns)
    splits = re.split(f"({zone_pattern})", text)

    # Basic validation - need at least 3 splits for any zones to be possible
    if len(splits) < 3:
        raise BlockParseError("Text must contain at least one zone token")

    # Split into contents and tokens
    contents = splits[0::2]
    zone_tokens = splits[1::2]

    # Validate zone token sequence
    if len(zone_tokens) < len(config.required_patterns):
        raise BlockParseError("Not enough zone tokens in block to meet required tokens")
    if len(zone_tokens) > len(config.zone_patterns):
        raise BlockParseError("Too many zone tokens in block - exceeds maximum allowed by config")

    for i, (actual_token, expected_token) in enumerate(zip(zone_tokens, config.zone_patterns)):
        if actual_token != expected_token:
            raise BlockParseError(
                f"Zone token '{actual_token}' does not match expected token '{expected_token}' at position {i}"
            )

    # Build zone structures following original logic
    zone_structures = []

    # First zone gets just the first token
    if zone_tokens:
        zone_structures.append({
            "advance_token": zone_tokens[0],
            "zone_text": zone_tokens[0]
        })

    # Subsequent zones get text from after previous token up to and including current token
    for i in range(1, len(config.zone_patterns)):
        if i < len(zone_tokens):
            # Text from after previous token + current token
            zone_text = contents[i] + zone_tokens[i]
        else:
            # No token found, just remaining text
            zone_text = contents[i] if i < len(contents) else ""

        zone_structures.append({
            "advance_token": config.zone_patterns[i],
            "zone_text": zone_text
        })

    return zone_structures


def _restore_escaped_content(zone_structures: List[Dict[str, str]], escape_lookup: Dict[str, str]) -> List[
    Dict[str, str]]:
    """
    Replace placeholders in zone text with their original escaped content.
    Returns new zone structures with escaped content restored.
    """
    restored_structures = []

    for zone_struct in zone_structures:
        zone_text = zone_struct["zone_text"]

        # Replace all placeholders with their escaped content
        for placeholder, escaped_content in escape_lookup.items():
            zone_text = zone_text.replace(placeholder, escaped_content)

        # Create new structure with restored text
        restored_struct = zone_struct.copy()
        restored_struct["zone_text"] = zone_text
        restored_structures.append(restored_struct)

    return restored_structures

## Tags handling, and replication of tagging so we repeat zone configs
#
# We validate tag specs match config. We also replicate one additional
# tag dimension to account for the extra zone added for simplicity in the
# text parser. this tagset is empty.

def resolve_and_validate_tags(block_data: Dict[str, Any], config: Config) -> List[List[List[str]]]:
    """
    Validate and resolve tags/tagset/repeats into unified tagset format.

    Args:
        block_data: Raw block data from TOML
        config: UDPL configuration

    Returns:
        List of tag structures, one per repetition. Each repetition contains
        a list of tag lists (one per zone). Includes empty tag list for the extra
        initial zone that triggers prompting.

    Raises:
        BlockParseError: If tags structure is invalid
    """
    if 'tagset' in block_data:
        return _process_tagset(block_data['tagset'], config)
    elif 'repeats' in block_data:
        return _process_tags_with_repeats(block_data['tags'], block_data['repeats'], config)
    else:
        return _process_simple_tags(block_data['tags'], config)


def _process_simple_tags(tags: Any, config: Config) -> List[List[List[str]]]:
    """Process simple tags field into expanded tagset format."""
    _validate_tags_field(tags, config)
    # Add empty tags for initial trigger zone, then user's tags
    expanded_tags = [[]] + tags
    return [expanded_tags]


def _process_tags_with_repeats(tags: Any, repeats: Any, config: Config) -> List[List[List[str]]]:
    """Process tags with repeats into expanded tagset format."""
    _validate_tags_field(tags, config)
    _validate_repeats_field(repeats)
    # Add empty tags for initial trigger zone, then user's tags
    expanded_tags = [[]] + tags
    return [expanded_tags for _ in range(repeats)]


def _process_tagset(tagset: Any, config: Config) -> List[List[List[str]]]:
    """Process tagset field into expanded tagset format."""
    _validate_tagset_field(tagset, config)
    # Add empty tags for initial trigger zone to each repetition
    expanded_tagset = []
    for rep_tags in tagset:
        expanded_rep = [[]] + rep_tags
        expanded_tagset.append(expanded_rep)
    return expanded_tagset


def _validate_tags_field(tags: Any, config: Config) -> None:
    """Validate tags field structure and content."""
    if not isinstance(tags, list):
        raise BlockParseError("'tags' field must be a list")
    if len(tags) != config.num_zones_per_block:
        raise BlockParseError(
            f"'tags' field must have {config.num_zones_per_block} sublists, got {len(tags)}"
        )

    for i, tag_list in enumerate(tags):
        if not isinstance(tag_list, list):
            raise BlockParseError(f"tags[{i}] must be a list")
        for j, tag in enumerate(tag_list):
            if not isinstance(tag, str):
                raise BlockParseError(f"tags[{i}][{j}] must be a string")
            if tag not in config.valid_tags:
                raise BlockParseError(f"Invalid tag '{tag}' not in config.valid_tags")


def _validate_tagset_field(tagset: Any, config: Config) -> None:
    """Validate tagset field structure and content."""
    if not isinstance(tagset, list):
        raise BlockParseError("'tagset' field must be a list")
    if len(tagset) == 0:
        raise BlockParseError("'tagset' cannot be empty")

    for rep_idx, tag_list in enumerate(tagset):
        if not isinstance(tag_list, list):
            raise BlockParseError(f"tagset[{rep_idx}] must be a list")
        if len(tag_list) != config.num_zones_per_block:
            raise BlockParseError(
                f"tagset[{rep_idx}] must have {config.num_zones_per_block} sublists, got {len(tag_list)}"
            )

        for zone_idx, zone_tags in enumerate(tag_list):
            if not isinstance(zone_tags, list):
                raise BlockParseError(f"tagset[{rep_idx}][{zone_idx}] must be a list")
            for tag_idx, tag in enumerate(zone_tags):
                if not isinstance(tag, str):
                    raise BlockParseError(f"tagset[{rep_idx}][{zone_idx}][{tag_idx}] must be a string")
                if tag not in config.valid_tags:
                    raise BlockParseError(f"Invalid tag '{tag}' not in config.valid_tags")