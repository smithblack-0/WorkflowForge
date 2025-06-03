"""
We define the prompt parsing to involve
parsing the toml into its individual
pieces, factories, and resource specs
"""
import string
import tomli
import re

from ..prompt_logic import ZoneFactoryStub, ResourceSpec
from ..tags import TagConverter
from typing import List, Tuple, Dict, Any, Callable
from dataclasses import dataclass

formatter = string.Formatter()

## ZONE CONFIG ###
#
# Logic parses zone config out of raw toml
# dump, validating type sanity and values along the way.
#
# Once we have a trusted zones config we can proceed
# into the later stages of parsing.

@dataclass
class ZonesConfig:
    zone_tokens: List[str]
    required_tokens: List[str]
    valid_tags: List[str]
    default_max_token_length: int


def get_zones_tokens(zone_config: Dict[str, Any]) -> List[str]:
    """Get the zone tokens, and validate all types and logic"""
    if "zone_tokens" not in zone_config:
        raise ValueError("feature 'zone_tokens' for prompts toml file is missing")
    zone_tokens = zone_config["zone_tokens"]
    if not isinstance(zone_tokens, list):
        raise ValueError("feature 'zone_tokens' for prompts toml file is not a list")
    if len(zone_tokens) < 2:
        raise ValueError("It is not possible to define a zone between less than two zone tokens")
    for token in zone_tokens:
        if not isinstance(token, str):
            raise ValueError("item in list zone_tokens is not a string.")
    return zone_tokens

def get_required_tokens(zone_config: Dict, zone_tokens: List[str]) -> List[str]:
    """Get any specified required tokens, check they are sane"""
    if "required_tokens" not in zone_config:
        return []
    else:
        required_tokens = zone_config["required_tokens"]
        if not isinstance(required_tokens, list):
            raise ValueError("feature 'required_tokens' for prompts toml file is not a list")
        for token in required_tokens:
            if not isinstance(token, str):
                raise ValueError("item in list required_tokens is not a string.")
            if token not in zone_tokens:
                raise ValueError("required token was not an option defined in zone_tokens.")

        return required_tokens

def get_valid_tags(zone_config: Dict) -> List[str]:
    if "valid_tags" not in zone_config:
        raise ValueError("feature 'valid_tags' for prompts toml file config does not exist")
    valid_tags = zone_config["valid_tags"]
    if not isinstance(valid_tags, list):
        raise ValueError("feature 'valid_tags' for prompts toml file is not a list")
    for tag in valid_tags:
        if not isinstance(tag, str):
            raise ValueError("item in list valid_tags is not a string.")
    return valid_tags

def get_default_max_token_length(zone_config: Dict) -> int:
    if "default_max_token_length" not in zone_config:
        raise ValueError("feature 'default_max_token_length' for prompts toml file config does not exist")
    default_max_token_length = zone_config["default_max_token_length"]
    if not isinstance(default_max_token_length, int):
        raise ValueError("feature 'default_max_token_length' for prompts toml file is not a int")
    if default_max_token_length < 1:
        raise ValueError("feature 'default_max_token_length' for prompts toml file is not a positive int")
    return default_max_token_length

def parse_zones_config(parsed_toml: Dict[str, Any])->ZonesConfig:
    if "config" not in parsed_toml:
        raise ValueError("Config for prompts toml file is missing")
    config = parsed_toml["config"]

    try:
        zone_tokens = get_zones_tokens(config)
        required_tokens = get_required_tokens(config, zone_tokens)
        valid_tags = get_valid_tags(config)
        max_gen_length = get_default_max_token_length(config)
    except Exception as err:
        raise RuntimeError("Issue in [Config] in toml file") from err

    return ZonesConfig(zone_tokens, required_tokens, valid_tags, max_gen_length)

### CONCRETE BLOCK PARSER
#
# The concrete block parser is provided
# a dictionary of features according to
# a standardized format with any repetition
# preprocessing resolved, then directly returns
# a list of zone factories.

def parse_block_text(block: Dict[str, Any],
                     zone_config: ZonesConfig
                     )->Tuple[List[str], List[str]]:
    """Parses the block of text into its zones, removing edge tokens"""

    # Basic type validation occurs
    if "text" not in block:
        raise ValueError("feature 'text' missing in block")
    text = block["text"]
    if not isinstance(text, str):
        raise ValueError("feature 'text' is not a string.")

    # Take apart text into its segments. End with a list of
    # prompt tokens, and the zone contents, what is between
    # those tokens.
    pattern = "|".join(re.escape(token) for token in zone_config.zone_tokens)
    splits = re.split(f"({pattern})", text)
    if len(splits) < 3:
        raise ValueError("feature 'text' cannot have used two zone tokens and so has no zones")
    if len(splits) == 3:
        # Pattern could be [Token], ..., [Token] or ..., [Token], ..., only one of which is valid
        if splits[0] not in zone_config.zone_tokens:
            raise ValueError("feature 'text' cannot have used two zone tokens and so has no zones")

    # In this block of code, we are stripping away any initial
    # and final token that is not a special token, and separating
    # the splits into the zone tokens themselves and then the
    # contents of each zone.
    if splits[0] not in zone_config.zone_tokens:
        splits = splits[1:]
    zone_tokens = splits[::2]
    zones = splits[1::2]
    if len(zones)==len(zone_tokens):
        zones = zones[:-1]

    # Verify required tokens exist, no excess of tokens exist,
    # they all exist in the right order.

    if len(zone_tokens) < len(zone_config.required_tokens):
        raise ValueError("feature 'text' does not have sufficient zone tokens to satisfy all required tokens")
    if len(zone_tokens) > len(zone_config.zone_tokens):
        raise ValueError("feature 'text' have more zone tokens than the possible maximum configuration")
    for i, (zone_token, expected_token) in enumerate(zip(zone_tokens, zone_config.zone_tokens)):
        if zone_token != expected_token:
            raise ValueError(f"tokens in 'text' in wrong order, token {i}. Expected {expected_token}, got {zone_token}")
    return zone_tokens, zones

def parse_block_tags(block: Dict[str, Any],
                     zone_config: ZonesConfig
                     )->List[List[str]]:
    """
    Parse the block tags off the block into a per-zone collection
    of the relevant tags, and performs all needed validation.
    """

    # Load and perform basic validation
    # on the entire tag collection itself.
    if "tags" not in block:
        raise ValueError("feature 'tags' missing in block")
    tags = block["tags"]
    if not isinstance(tags, list):
        raise ValueError("feature 'tags' is not a list")
    if len(tags) != len(zone_config.zone_tokens)-1:
        raise ValueError("The number of provided tag collections does not match the number of zones")

    # Enter the tag lists for each zone. Are they sane?
    for i, item in enumerate(tags):
        if not isinstance(item, list):
            raise ValueError(f"feature at index '{i}' in tags is not a list")
        for j, tag in enumerate(item):
            if not isinstance(tag, str):
                raise ValueError(f"feature at tags[{i}][{j}] is not a string")
            if tag not in zone_config.valid_tags:
                raise ValueError(f"feature at tags[{i}][{j}] is not a valid tag; got {tag},"
                                 f" but only {zone_config.valid_tags} allowed")

    # Validation complete. Return
    return tags

def parse_block_token_limit(block: Dict[str, Any])->int:
    """
    Parses the blocks token limit. Straightforward.
    """
    if "max_gen_tokens" not in block:
        raise RuntimeError("feature 'max_gen_tokens' missing in block; this should be impossible,"
                           " contact the maintainer")
    max_gen_tokens = block["max_gen_tokens"]
    if not isinstance(max_gen_tokens, int):
        raise ValueError("feature 'max_gen_tokens' is not an int")
    if max_gen_tokens < 1:
        raise ValueError("feature 'max_gen_tokens' must be greater than 0")
    return max_gen_tokens


def extract_placeholders_from_string(format_string: str) -> List[str]:
    """Parsing of string placeholders"""
    return [field_name for _, field_name, _, _ in formatter.parse(format_string) if field_name]

def parse_zone_resource_spec_collection(block: Dict[str, Any], zone: str):
    """
    Parses into the resource spec format. This is a mapping of the 
    placeholder name into the resource spec, which contains the resource
    call to resolve it. 
    :param zone: The zone under concern.
    :param block: The block to draw calls from
    :return: The resource spec dictionary for the zone
    """
    placeholders = extract_placeholders_from_string(zone)
    specs = {}
    for placeholder in placeholders:
        # Get and chekc the type of the specific placeholder
        # resolution call at the outermost level.
        if placeholder not in block:
            raise ValueError(f"feature '{placeholder}' missing from block, preventing resource resolution")
        placeholder_section = block[placeholder]
        if not isinstance(placeholder_section, dict):
            raise ValueError(f"feature '{placeholder}' of block not a dict")

        # Get, check the type of the name feature.
        if 'name' not in placeholder_section:
            raise ValueError(f"the dictionary '{placeholder}' does not have a 'name' key")
        name = placeholder_section['name']
        if not isinstance(name, str):
            raise ValueError(f"the dictionary '{placeholder}' has a 'name' key that is not a string")

        # Get the arguments.
        if 'arguments' not in placeholder_section:
            arguments = None
        else:
            arguments = placeholder_section['arguments']
            if not isinstance(arguments, dict):
                raise ValueError(f"'arguments' dictionary in '{placeholder}' must be omitted or a dictionary.")
        spec = ResourceSpec(name=name, arguments=arguments)
        specs[placeholder] = spec
    return specs


def parse_concrete_block(block_num: int,
                         block: Dict[str, Any],
                         zone_config: ZonesConfig,
                         tag_converter: TagConverter
                         )->List[ZoneFactoryStub]:
    """
    A block is "concrete" once it can be parsed directly from the contents
    of the tags with all defaults applied. This block is, by this point,
    concrete. We parse it. We return a list of ZoneFactory objects that can produce
    a zone once it is invoked with resources.
    :param block: The block to parse
    :param zone_config: The zones config with various tag and such validation info
    :param tag_converter: Tracks the tag mappings, and converts tag lists for a zone into their
           bool array form or back again.
    :return: A list of ZoneFactoryStub objects which can be invoked with
    a resource collection and a tokenizer to finish initialization.
    """

    # Perform primary parsing, with the majority of the validation
    # in place.
    try:
        zone_tokens, zones = parse_block_text(block, zone_config)
        zone_tags = parse_block_tags(block, zone_config)
        block_token_limit = parse_block_token_limit(block)
        zone_resources = [parse_zone_resource_spec_collection(block, zone) for zone in zones]
    except Exception as err:
        raise RuntimeError(f"Issue in [Block] in toml file for block {block_num}") from err

    # Assemble all the factories from each respective zone
    advancement_tokens = zone_tokens[1:]
    zone_factories = []
    for zone, token, tags, resources in zip(zones, advancement_tokens, zone_tags, zone_resources):
        tags = tag_converter.tensorize(tags)
        zone_factories.append(ZoneFactoryStub(resources,
                                              zone,
                                              tags,
                                              token,
                                              block_token_limit))
    return zone_factories

### MAIN PARSING LOGIC
# Glue code, preprocessing, and looping for parsing



def parse_block(block_num: int,
                block: Dict[str, Any],
                zone_config: ZonesConfig,
                tag_converter: TagConverter
                )->List[ZoneFactoryStub]:
    """
    Parses a given block completely based on
    the provided parameters, doing all needed validation, repetition
    preprocessing, and any other related actions.
    :param block_num: The block number to pass though, for error messages
    :param block: The block to parse
    :param zone_config: The zone config
    :param tag_converter: Tracks the tag mappings, and converts tag lists for a zone into their
       bool array form or back again.
    :return: A list of all the ZoneFactoryStubs this parses into.
    """
    # Internally, the primary responsibility of parse block
    # is to make a block concrete. This means to fill out any missing
    # defaults, handle repeats and other similar preprocessing effects,
    # to prepare something that parse_concrete_block would know how
    # to handle

    # Handle all default propogation first. Warning, side effects can happen on
    # interior modifications, even if the outer dicts are separate now.
    block = block.copy()
    if "max_gen_tokens" not in block:
        block["max_gen_tokens"] = zone_config.default_max_token_length

    # Handle repetitions and similar preprocessing phenomenon. We turn them into
    # equivalent block collections that have all needed concrete features populated.
    process_blocks = []
    if "repeats" in block and "tagset" in block:
        raise ValueError(f"You cannot use both repeats and tagset; On block {block_num}")
    elif 'repeats' in block:
        repeats = block['repeats']
        if not isinstance(repeats, int):
            raise ValueError(f"'repeats' must be an int; On block {block_num}")
        if repeats < 1:
            raise ValueError(f"'repeats' must be greater than 0; On block {block_num}")
        process_blocks += [block.copy() for  _ in range(repeats)]
    elif 'tagset' in block:
        tagset = block['tagset']
        if not isinstance(tagset, list):
            raise ValueError(f"'tagset' must be a list; On block {block_num}")
        if len(tagset) < 1:
            raise ValueError(f"'tagset' must have a length greater than zero; On block {block_num}")
        if any(not isinstance(tag_collection, list)  for tag_collection in tagset):
            raise ValueError(f"'tagset' must only contain tag collects, got wrong type; On block {block_num}")

        for tag_collection in tagset:
            process_block = block.copy()
            process_block['tags'] = tag_collection
            process_blocks.append(process_block)
    else:
        process_blocks += [block.copy()]

    # Perform primary processing
    output = []
    for i, process_block in enumerate(process_blocks):
        try:
            output = output + parse_concrete_block(block_num, process_block, zone_config, tag_converter)
        except Exception as err:
            raise RuntimeError(
                                f"Issue running block during repeat {i} "
                                f"If you are not using repeats or tagset, you can ignore this context and focus "
                                f"on the rest of your error chain"
                            ) from err

    return output

def parse_prompts_file(file_path: str)->Tuple[List[ZoneFactoryStub], TagConverter]:
    """
    Parses the prompts file into its constituent zone
    factories. Or at least their stub form, since they will need
    just a little bit more information later.
    :param file_path: The file to load and parse from.
    :return: The parsed list of ZoneFactoryStub objects, and the tag converter.
    """
    with open(file_path, "rb") as f:
        parsed_toml = tomli.load(f)
    zones_config = parse_zones_config(parsed_toml)
    tag_converter = TagConverter({tag : i for i, tag in enumerate(zones_config.valid_tags)})
    if "blocks" not in parsed_toml:
        raise ValueError("You must specify 'blocks' in the prompts file")
    blocks = parsed_toml["blocks"]
    if not isinstance(blocks, list):
        raise ValueError("'blocks' must be a list")
    output = []
    for i, block in enumerate(blocks):
        output.extend(parse_block(i, block, zones_config, tag_converter))
    return output, tag_converter



