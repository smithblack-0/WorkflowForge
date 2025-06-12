"""
All parsing specifications are formally defined in the
UDPL documentation specification. This is an implementation
of the constraints.
"""


from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import warnings


@dataclass
class Config:
    """
    The configuration object produced by good
    UDPL files. Provides most of the strong
    typing guarantees of the project.

    """

    @property
    def num_zones_per_block(self) -> int:
        return len(self.zone_tokens) - 1

    zone_tokens: List[str]
    required_tokens: List[str]
    valid_tags: List[str]
    default_max_token_length: int
    sequences: List[str]
    control_token: str
    escape_token: str
    special_tokens: List[str]
    misc: Dict[str, Any]


class ConfigParseError(Exception):
    """Exception raised when config parsing fails."""
    pass


def parse_config(toml_data: Dict[str, Any]) -> Config:
    """
    Parse and validate a UDPL config from raw TOML data.

    Args:
        toml_data: Raw parsed TOML data as a dictionary

    Returns:
        Config: Validated configuration object

    Raises:
        ConfigParseError: If config is missing or invalid
    """
    # Check if config section exists
    if 'config' not in toml_data:
        raise ConfigParseError("Missing required [config] section")

    config_section = toml_data['config']

    # Validate zone_tokens
    if 'zone_tokens' not in config_section:
        raise ConfigParseError("Missing required 'zone_tokens' in config")

    zone_tokens = config_section['zone_tokens']
    if not isinstance(zone_tokens, list):
        raise ConfigParseError("'zone_tokens' must be a list of strings")
    if len(zone_tokens) == 0:
        raise ConfigParseError("'zone_tokens' cannot be empty")
    if len(zone_tokens) < 2:
        raise ConfigParseError("'zone_tokens' must contain at least 2 tokens")
    if not all(isinstance(token, str) for token in zone_tokens):
        raise ConfigParseError("All 'zone_tokens' must be strings")

    # Validate required_tokens
    if 'required_tokens' not in config_section:
        raise ConfigParseError("Missing requirement 'required_tokens' in config")

    required_tokens = config_section['required_tokens']
    if not isinstance(required_tokens, list):
        raise ConfigParseError("'required_tokens' must be a list of strings")
    if len(required_tokens) == 0:
        raise ConfigParseError("'required_tokens' cannot be empty")
    if not all(isinstance(token, str) for token in required_tokens):
        raise ConfigParseError("All 'required_tokens' must be strings")

    # Check that all required_tokens are in zone_tokens
    for token in required_tokens:
        if token not in zone_tokens:
            raise ConfigParseError(f"Required token '{token}' not found in zone_tokens")

    # Validate valid_tags
    if 'valid_tags' not in config_section:
        raise ConfigParseError("Missing required 'valid_tags' in config")

    valid_tags = config_section['valid_tags']
    if not isinstance(valid_tags, list):
        raise ConfigParseError("'valid_tags' must be a list")
    if len(valid_tags) == 0:
        warnings.warn("'valid_tags' is empty", UserWarning)
    if not all(isinstance(tag, str) for tag in valid_tags):
        raise ConfigParseError("All 'valid_tags' must be strings")

    # Validate default_max_token_length
    if 'default_max_token_length' not in config_section:
        raise ConfigParseError("Missing required 'default_max_token_length' in config")

    default_max_token_length = config_section['default_max_token_length']
    if not isinstance(default_max_token_length, int):
        raise ConfigParseError("'default_max_token_length' must be an integer")
    if default_max_token_length <= 0:
        raise ConfigParseError("'default_max_token_length' must be greater than 0")

    # Validate sequences
    if 'sequences' not in config_section:
        raise ConfigParseError("Missing required 'sequences' in config")

    sequences = config_section['sequences']
    if not isinstance(sequences, list):
        raise ConfigParseError("'sequences' must be a list")
    if len(sequences) == 0:
        raise ConfigParseError("'sequences' cannot be empty")
    if not all(isinstance(seq, str) for seq in sequences):
        raise ConfigParseError("All 'sequences' must be strings")
    if not all(seq.strip() for seq in sequences):
        raise ConfigParseError("All 'sequences' must be non-empty strings")

    # Validate control_token
    if 'control_token' not in config_section:
        raise ConfigParseError("Missing required 'control_token' in config")

    control_token = config_section['control_token']
    if not isinstance(control_token, str):
        raise ConfigParseError("'control_token' must be a string")
    if not control_token.strip():
        raise ConfigParseError("'control_token' cannot be empty")

    # Validate escape_token
    if 'escape_token' not in config_section:
        raise ConfigParseError("Missing required 'escape_token' in config")

    escape_token = config_section['escape_token']
    if not isinstance(escape_token, str):
        raise ConfigParseError("'escape_token' must be a string")
    if not escape_token.strip():
        raise ConfigParseError("'escape_token' cannot be empty")

    # Create special_tokens list (all tokens needed based on config)
    special_tokens = zone_tokens.copy()
    if control_token not in special_tokens:
        special_tokens.append(control_token)
    if escape_token not in special_tokens:
        special_tokens.append(escape_token)
    for token in zone_tokens:
        if token not in special_tokens:
            special_tokens.append(token)

    # Create Config object
    return Config(
        zone_tokens=zone_tokens,
        required_tokens=required_tokens,
        valid_tags=valid_tags,
        default_max_token_length=default_max_token_length,
        sequences=sequences,
        control_token=control_token,
        escape_token=escape_token,
        special_tokens=special_tokens,
        misc=toml_data
    )