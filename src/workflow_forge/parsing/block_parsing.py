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

    Args:
        text: Raw text from the block
        config: UDPL configuration

    Returns:
        List of zone structures with 'advance_token' and 'zone_text' keys

    Raises:
        BlockParseError: If text validation fails
    """
    # Use regex to split on any special token.
    #
    # We have to account for the fact that escape tokens
    # can escape zone edges too, or not, which makes this complex.

    pattern = "|".join(re.escape(token) for token in config.special_patterns)
    splits = re.split(f"({pattern})", text)

    # Basic validation - need at least 3 splits for any zones to even be possible
    if len(splits) < 3:
        raise BlockParseError("Text must contain at least one zone token")

    # Split up into intial zone configuration.

    contents = splits[0::2]
    control_tokens = splits[1::2]

    # We split on all control tokens. However, some control other processes,
    # that may have been escaped. We merge sections back together to get
    # the actual active zones, and skip zone construction when
    # escaped.

    zone_contents = []
    zone_tokens = []
    is_first_zone_done = False
    is_escaped = False
    current_string = ""
    for content, control_token in zip(contents, control_tokens):

        if is_escaped is True:
            current_string += content+control_token
            is_escaped = False
        else:
            current_string += content + control_token
            if control_token in config.zone_patterns:
                if not is_first_zone_done:
                    current_string = control_token
                    is_first_zone_done = True
                zone_contents.append(current_string)
                zone_tokens.append(control_token)
                current_string = ""
            elif control_token == config.escape_token:
                is_escaped = True
    zone_contents.append(contents[-1])

    if len(zone_tokens) < len(config.required_patterns):
        raise BlockParseError("Not enough zone tokens in block to meet required tokens")
    if len(zone_tokens) > len(config.zone_patterns):
        raise BlockParseError("Too many zone tokens in block - exceeds maximum allowed by config")
    for i, (config_token, actual_token) in enumerate(zip(zone_tokens, config.zone_patterns)):
        if config_token != actual_token:
            raise BlockParseError(f"Zone token '{config_token}' does not match zone token '{actual_token}'"
                                  f" at position {i}")

    # Walk through and make zone structure results, containing the zone
    # text, and the advance token that goes with it. Fill in anything missing

    zone_structures = []
    for i, zone_token in enumerate(config.zone_patterns):
        content = zone_contents[i] if i < len(zone_contents) else ""
        zone_structures.append({"advance_token": zone_token, "zone_text": content})
    return zone_structures


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