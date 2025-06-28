"""
All parsing specifications are formally defined in the
UDPL documentation specification. This is an implementation
of the constraints.
"""


from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
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
        return len(self.zone_patterns) - 1

    zone_patterns: List[str]
    required_patterns: List[str]
    valid_tags: List[str]
    default_max_token_length: int
    sequences: List[str]
    control_pattern: str
    escape_patterns: Tuple[str, str]
    tools: List[str]
    misc: Dict[str, Any]

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize config to a dictionary suitable for transport.
        Returns data as-is since msgpack will handle most types correctly.
        """
        return {
            "zone_patterns": self.zone_patterns,
            "required_patterns": self.required_patterns,
            "valid_tags": self.valid_tags,
            "default_max_token_length": self.default_max_token_length,
            "sequences": self.sequences,
            "control_pattern": self.control_pattern,
            "escape_patterns": self.escape_patterns,  # Let msgpack handle tuple serialization
            "tools": self.tools,
            "misc": self.misc
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'Config':
        """
        Deserialize config from dictionary, restoring proper types.
        Explicitly converts escape_patterns back to tuple.
        """
        # Make a copy to avoid mutating input
        config_data = data.copy()

        # Convert escape_patterns back to tuple if it was converted to list
        if "escape_patterns" in config_data:
            config_data["escape_patterns"] = tuple(config_data["escape_patterns"])

        return cls(**config_data)


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

    # Validate zone patterns
    if 'zone_patterns' not in config_section:
        raise ConfigParseError("Missing required 'zone_patterns' in config")

    zone_patterns = config_section['zone_patterns']
    if not isinstance(zone_patterns, list):
        raise ConfigParseError("'zone_patterns' must be a list of strings")
    if len(zone_patterns) == 0:
        raise ConfigParseError("'zone_patterns' cannot be empty")
    if len(zone_patterns) < 2:
        raise ConfigParseError("'zone_patterns' must contain at least 2 patterns")
    if not all(isinstance(token, str) for token in zone_patterns):
        raise ConfigParseError("All 'zone_patterns' must be strings")

    # Validate required_patterns
    if 'required_patterns' not in config_section:
        raise ConfigParseError("Missing requirement 'required_patterns' in config")

    required_patterns = config_section['required_patterns']
    if not isinstance(required_patterns, list):
        raise ConfigParseError("'required_patterns' must be a list of strings")
    if len(required_patterns) == 0:
        raise ConfigParseError("'required_patterns' cannot be empty")
    if not all(isinstance(token, str) for token in required_patterns):
        raise ConfigParseError("All 'required_patterns' must be strings")

    # Check that all required_patterns are in zone_patterns
    for token in required_patterns:
        if token not in zone_patterns:
            raise ConfigParseError(f"feature '{token}' in 'required_patterns' not found in zone_patterns")

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

    # Validate control_pattern
    if 'control_pattern' not in config_section:
        raise ConfigParseError("Missing required 'control_pattern' in config")

    control_pattern = config_section['control_pattern']
    if not isinstance(control_pattern, str):
        raise ConfigParseError("'control_pattern' must be a string")
    if not control_pattern.strip():
        raise ConfigParseError("'control_pattern' cannot be empty")

    # Validate escape patterns
    if 'escape_patterns' not in config_section:
        raise ConfigParseError("Missing required 'escape_patterns' in config")
    escape_patterns = config_section['escape_patterns']

    if not isinstance(escape_patterns, list):
        raise ConfigParseError("'escape_patterns' must be a list of strings")
    if len(escape_patterns) != 2:
        raise ConfigParseError("'escape_patterns' must contain exactly two strings")

    for i, string in enumerate(escape_patterns):
        if not isinstance(string, str):
            raise ConfigParseError(f"{i}th 'escape_patterns' must be a string")
        if not string.strip():
            raise ConfigParseError(f"{i}th 'escape_patterns' cannot be empty")
    escape_patterns = tuple(escape_patterns)

    # Validate tools config.
    if "tools" not in config_section:
        raise ConfigParseError("Missing required tools in config")
    tools = config_section['tools']
    if not isinstance(tools, list):
        raise ConfigParseError("'tools' must be a list")
    for tool in tools:
        if not isinstance(tool, str):
            raise ConfigParseError("each defined 'tool' must be a string")

    # Create Config object
    return Config(
        zone_patterns=zone_patterns,
        required_patterns=required_patterns,
        valid_tags=valid_tags,
        default_max_token_length=default_max_token_length,
        sequences=sequences,
        control_pattern=control_pattern,
        escape_patterns=escape_patterns,
        tools = tools,
        misc=toml_data
    )