import unittest
import warnings
from typing import Dict, Any
from src.workflow_forge.parsing.config_parsing import Config, ConfigParseError, parse_config


class BaseConfigTest(unittest.TestCase):
    """Base test class with helper methods."""

    def get_valid_config(self) -> Dict[str, Any]:
        """Return a valid config for testing."""
        return {
            'config': {
                'zone_patterns': ["[Prompt]", "[Answer]", "[EOS]"],
                'required_patterns': ["[Prompt]", "[Answer]"],
                'valid_tags': ["Training", "Correct", "Incorrect"],
                'default_max_token_length': 20000,
                'sequences': ["setup", "loop", "solving", "concluding"],
                'control_pattern': "[Jump]",
                'escape_patterns': ["[Escape]", "[EndEscape]"],
                'tools': []
            },
            'some_other_section': {
                'custom_field': 'custom_value'
            }
        }


class TestConfigSectionExists(BaseConfigTest):
    """Test that config section is present."""

    def test_missing_config_section(self):
        """Test error when config section is missing."""
        toml_data = {'other_section': {'key': 'value'}}

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("missing", str(err.exception).lower())
        self.assertIn("config", str(err.exception).lower())


class TestZonePatternsValidation(BaseConfigTest):
    """Test zone_patterns validation rules."""

    def test_missing_zone_patterns(self):
        """Test zone_patterns is present."""
        toml_data = self.get_valid_config()
        del toml_data['config']['zone_patterns']

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("zone_patterns", str(err.exception).lower())
        self.assertIn("missing", str(err.exception).lower())

    def test_zone_patterns_is_list(self):
        """Test zone_patterns is a list."""
        toml_data = self.get_valid_config()
        toml_data['config']['zone_patterns'] = "[Prompt]"

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("list", str(err.exception).lower())
        self.assertIn("zone_patterns", str(err.exception).lower())

    def test_zone_patterns_contains_strings(self):
        """Test zone_patterns contains only strings."""
        toml_data = self.get_valid_config()
        toml_data['config']['zone_patterns'] = ["[Prompt]", 123, "[EOS]"]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("zone_patterns", str(err.exception).lower())
        self.assertIn("string", str(err.exception).lower())

    def test_zone_patterns_nonempty(self):
        """Test zone_patterns is not empty."""
        toml_data = self.get_valid_config()
        toml_data['config']['zone_patterns'] = []

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("zone_patterns", str(err.exception).lower())
        self.assertIn("empty", str(err.exception).lower())

    def test_zone_patterns_length_at_least_two(self):
        """Test zone_patterns has at least 2 elements."""
        toml_data = self.get_valid_config()
        toml_data['config']['zone_patterns'] = ["[Prompt]"]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("zone_patterns", str(err.exception).lower())
        self.assertIn("2", str(err.exception).lower())


class TestRequiredPatternsValidation(BaseConfigTest):
    """Test required_patterns validation rules."""

    def test_missing_required_patterns(self):
        """Test required_patterns is present."""
        toml_data = self.get_valid_config()
        del toml_data['config']['required_patterns']

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("required_patterns", str(err.exception).lower())
        self.assertIn("missing", str(err.exception).lower())

    def test_required_patterns_is_list(self):
        """Test required_patterns is a list."""
        toml_data = self.get_valid_config()
        toml_data['config']['required_patterns'] = "[Prompt]"

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("required_patterns", str(err.exception).lower())
        self.assertIn("list", str(err.exception).lower())

    def test_required_patterns_contains_strings(self):
        """Test required_patterns contains only strings."""
        toml_data = self.get_valid_config()
        toml_data['config']['required_patterns'] = ["[Prompt]", 456]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("required_patterns", str(err.exception).lower())
        self.assertIn("string", str(err.exception).lower())

    def test_required_patterns_nonempty(self):
        """Test required_patterns is not empty."""
        toml_data = self.get_valid_config()
        toml_data['config']['required_patterns'] = []

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("required_patterns", str(err.exception).lower())
        self.assertIn("empty", str(err.exception).lower())

    def test_required_patterns_in_zone_patterns(self):
        """Test all required patterns exist in zone patterns."""
        toml_data = self.get_valid_config()
        toml_data['config']['required_patterns'] = ["[Prompt]", "[Missing]"]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("required_patterns", str(err.exception).lower())
        self.assertIn("zone_patterns", str(err.exception).lower())


class TestValidTagsValidation(BaseConfigTest):
    """Test valid_tags validation rules."""

    def test_missing_valid_tags(self):
        """Test valid_tags is present."""
        toml_data = self.get_valid_config()
        del toml_data['config']['valid_tags']

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("valid_tags", str(err.exception).lower())
        self.assertIn("missing", str(err.exception).lower())

    def test_valid_tags_is_list(self):
        """Test valid_tags is a list."""
        toml_data = self.get_valid_config()
        toml_data['config']['valid_tags'] = "Training"

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("valid_tags", str(err.exception).lower())
        self.assertIn("list", str(err.exception).lower())

    def test_valid_tags_contains_strings(self):
        """Test valid_tags contains only strings."""
        toml_data = self.get_valid_config()
        toml_data['config']['valid_tags'] = ["Training", 789]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("valid_tags", str(err.exception).lower())
        self.assertIn("string", str(err.exception).lower())

    def test_valid_tags_empty_warning(self):
        """Test warning is issued when valid_tags is empty."""
        toml_data = self.get_valid_config()
        toml_data['config']['valid_tags'] = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = parse_config(toml_data)
            self.assertEqual(len(w), 1)
            self.assertIn("valid_tags", str(w[0].message).lower())
            self.assertIn("empty", str(w[0].message).lower())
            self.assertEqual(config.valid_tags, [])


class TestDefaultMaxTokenLengthValidation(BaseConfigTest):
    """Test default_max_token_length validation rules."""

    def test_missing_default_max_token_length(self):
        """Test default_max_token_length is present."""
        toml_data = self.get_valid_config()
        del toml_data['config']['default_max_token_length']

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("default_max_token_length", str(err.exception).lower())
        self.assertIn("missing", str(err.exception).lower())

    def test_default_max_token_length_is_integer(self):
        """Test default_max_token_length is an integer."""
        toml_data = self.get_valid_config()
        toml_data['config']['default_max_token_length'] = "20000"

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("default_max_token_length", str(err.exception).lower())
        self.assertIn("integer", str(err.exception).lower())

    def test_default_max_token_length_greater_than_zero(self):
        """Test default_max_token_length is greater than zero."""
        toml_data = self.get_valid_config()
        toml_data['config']['default_max_token_length'] = 0

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("default_max_token_length", str(err.exception).lower())
        self.assertIn("greater", str(err.exception).lower())

    def test_default_max_token_length_not_negative(self):
        """Test default_max_token_length is not negative."""
        toml_data = self.get_valid_config()
        toml_data['config']['default_max_token_length'] = -100

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("default_max_token_length", str(err.exception).lower())
        self.assertIn("greater", str(err.exception).lower())


class TestSequencesValidation(BaseConfigTest):
    """Test sequences validation rules."""

    def test_missing_sequences(self):
        """Test sequences is present."""
        toml_data = self.get_valid_config()
        del toml_data['config']['sequences']

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("sequences", str(err.exception).lower())
        self.assertIn("missing", str(err.exception).lower())

    def test_sequences_is_list(self):
        """Test sequences is a list."""
        toml_data = self.get_valid_config()
        toml_data['config']['sequences'] = "setup"

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("sequences", str(err.exception).lower())
        self.assertIn("list", str(err.exception).lower())

    def test_sequences_contains_strings(self):
        """Test sequences contains only strings."""
        toml_data = self.get_valid_config()
        toml_data['config']['sequences'] = ["setup", 123]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("sequences", str(err.exception).lower())
        self.assertIn("string", str(err.exception).lower())

    def test_sequences_nonempty(self):
        """Test sequences is not empty."""
        toml_data = self.get_valid_config()
        toml_data['config']['sequences'] = []

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("sequences", str(err.exception).lower())
        self.assertIn("empty", str(err.exception).lower())

    def test_sequences_strings_nonempty(self):
        """Test sequences contains non-empty strings."""
        toml_data = self.get_valid_config()
        toml_data['config']['sequences'] = ["setup", ""]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("sequences", str(err.exception).lower())
        self.assertIn("non-empty", str(err.exception).lower())

    def test_sequences_strings_not_whitespace_only(self):
        """Test sequences strings are not whitespace-only."""
        toml_data = self.get_valid_config()
        toml_data['config']['sequences'] = ["setup", "   "]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("sequences", str(err.exception).lower())
        self.assertIn("non-empty", str(err.exception).lower())


class TestControlPatternValidation(BaseConfigTest):
    """Test control_pattern validation rules."""

    def test_missing_control_pattern(self):
        """Test control_pattern is present."""
        toml_data = self.get_valid_config()
        del toml_data['config']['control_pattern']

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("control_pattern", str(err.exception).lower())
        self.assertIn("missing", str(err.exception).lower())

    def test_control_pattern_is_string(self):
        """Test control_pattern is a string."""
        toml_data = self.get_valid_config()
        toml_data['config']['control_pattern'] = 123

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("control_pattern", str(err.exception).lower())
        self.assertIn("string", str(err.exception).lower())

    def test_control_pattern_nonempty(self):
        """Test control_pattern is not empty."""
        toml_data = self.get_valid_config()
        toml_data['config']['control_pattern'] = ""

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("control_pattern", str(err.exception).lower())
        self.assertIn("empty", str(err.exception).lower())

    def test_control_pattern_not_whitespace_only(self):
        """Test control_pattern is not whitespace-only."""
        toml_data = self.get_valid_config()
        toml_data['config']['control_pattern'] = "   "

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("control_pattern", str(err.exception).lower())
        self.assertIn("empty", str(err.exception).lower())


class TestEscapePatternsValidation(BaseConfigTest):
    """Test escape_patterns validation rules."""

    def test_missing_escape_patterns(self):
        """Test escape_patterns is present."""
        toml_data = self.get_valid_config()
        del toml_data['config']['escape_patterns']

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("escape_patterns", str(err.exception).lower())
        self.assertIn("missing", str(err.exception).lower())

    def test_escape_patterns_is_list(self):
        """Test escape_patterns is a list."""
        toml_data = self.get_valid_config()
        toml_data['config']['escape_patterns'] = "[Escape]"

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("escape_patterns", str(err.exception).lower())
        self.assertIn("list", str(err.exception).lower())

    def test_escape_patterns_exactly_two_elements(self):
        """Test escape_patterns contains exactly two elements."""
        toml_data = self.get_valid_config()
        toml_data['config']['escape_patterns'] = ["[Escape]"]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("escape_patterns", str(err.exception).lower())
        self.assertIn("two", str(err.exception).lower())

    def test_escape_patterns_exactly_two_elements_too_many(self):
        """Test escape_patterns with more than two elements fails."""
        toml_data = self.get_valid_config()
        toml_data['config']['escape_patterns'] = ["[Escape]", "[EndEscape]", "[Extra]"]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("escape_patterns", str(err.exception).lower())
        self.assertIn("two", str(err.exception).lower())

    def test_escape_patterns_first_element_string(self):
        """Test first escape pattern element is a string."""
        toml_data = self.get_valid_config()
        toml_data['config']['escape_patterns'] = [123, "[EndEscape]"]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("escape_patterns", str(err.exception).lower())
        self.assertIn("string", str(err.exception).lower())

    def test_escape_patterns_second_element_string(self):
        """Test second escape pattern element is a string."""
        toml_data = self.get_valid_config()
        toml_data['config']['escape_patterns'] = ["[Escape]", 456]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("escape_patterns", str(err.exception).lower())
        self.assertIn("string", str(err.exception).lower())

    def test_escape_patterns_first_element_nonempty(self):
        """Test first escape pattern is not empty."""
        toml_data = self.get_valid_config()
        toml_data['config']['escape_patterns'] = ["", "[EndEscape]"]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("escape_patterns", str(err.exception).lower())
        self.assertIn("empty", str(err.exception).lower())

    def test_escape_patterns_second_element_nonempty(self):
        """Test second escape pattern is not empty."""
        toml_data = self.get_valid_config()
        toml_data['config']['escape_patterns'] = ["[Escape]", ""]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("escape_patterns", str(err.exception).lower())
        self.assertIn("empty", str(err.exception).lower())

    def test_escape_patterns_first_element_not_whitespace_only(self):
        """Test first escape pattern is not whitespace-only."""
        toml_data = self.get_valid_config()
        toml_data['config']['escape_patterns'] = ["   ", "[EndEscape]"]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("escape_patterns", str(err.exception).lower())
        self.assertIn("empty", str(err.exception).lower())

    def test_escape_patterns_second_element_not_whitespace_only(self):
        """Test second escape pattern is not whitespace-only."""
        toml_data = self.get_valid_config()
        toml_data['config']['escape_patterns'] = ["[Escape]", "   "]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("escape_patterns", str(err.exception).lower())
        self.assertIn("empty", str(err.exception).lower())


class TestToolsValidation(BaseConfigTest):
    """Test tools validation rules."""

    def test_missing_tools(self):
        """Test tools is present."""
        toml_data = self.get_valid_config()
        del toml_data['config']['tools']

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("tools", str(err.exception).lower())
        self.assertIn("missing", str(err.exception).lower())

    def test_tools_is_list(self):
        """Test tools is a list."""
        toml_data = self.get_valid_config()
        toml_data['config']['tools'] = "tool1"

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("tools", str(err.exception).lower())
        self.assertIn("list", str(err.exception).lower())

    def test_tools_contains_strings(self):
        """Test tools contains only strings."""
        toml_data = self.get_valid_config()
        toml_data['config']['tools'] = ["tool1", 123, "tool3"]

        with self.assertRaises(ConfigParseError) as err:
            parse_config(toml_data)

        self.assertIn("tool", str(err.exception).lower())
        self.assertIn("string", str(err.exception).lower())

    def test_tools_empty_list_allowed(self):
        """Test tools can be an empty list."""
        toml_data = self.get_valid_config()
        toml_data['config']['tools'] = []

        config = parse_config(toml_data)
        self.assertEqual(config.tools, [])

    def test_tools_with_valid_strings(self):
        """Test tools with valid string values."""
        toml_data = self.get_valid_config()
        toml_data['config']['tools'] = ["calculator", "web_search", "file_reader"]

        config = parse_config(toml_data)
        self.assertEqual(config.tools, ["calculator", "web_search", "file_reader"])


class TestValidConfigScenarios(BaseConfigTest):
    """Test various valid configuration scenarios."""

    def test_complete_valid_config(self):
        """Test parsing a completely valid config."""
        toml_data = self.get_valid_config()
        config = parse_config(toml_data)

        # Check all basic fields
        self.assertEqual(config.zone_patterns, ["[Prompt]", "[Answer]", "[EOS]"])
        self.assertEqual(config.required_patterns, ["[Prompt]", "[Answer]"])
        self.assertEqual(config.valid_tags, ["Training", "Correct", "Incorrect"])
        self.assertEqual(config.default_max_token_length, 20000)
        self.assertEqual(config.sequences, ["setup", "loop", "solving", "concluding"])
        self.assertEqual(config.control_pattern, "[Jump]")
        self.assertEqual(config.escape_patterns, ("[Escape]", "[EndEscape]"))
        self.assertEqual(config.tools, [])

        # Check misc contains the full toml data
        self.assertEqual(config.misc, toml_data)

    def test_minimal_valid_config(self):
        """Test a minimal but valid config."""
        toml_data = {
            'config': {
                'zone_patterns': ["[A]", "[B]"],
                'required_patterns': ["[A]"],
                'valid_tags': ["Tag1"],
                'default_max_token_length': 1,
                'sequences': ["seq1"],
                'control_pattern': "[C]",
                'escape_patterns': ["[E]", "[EndE]"],
                'tools': []
            }
        }

        config = parse_config(toml_data)
        self.assertEqual(config.zone_patterns, ["[A]", "[B]"])
        self.assertEqual(config.required_patterns, ["[A]"])
        self.assertEqual(config.valid_tags, ["Tag1"])
        self.assertEqual(config.default_max_token_length, 1)
        self.assertEqual(config.sequences, ["seq1"])
        self.assertEqual(config.control_pattern, "[C]")
        self.assertEqual(config.escape_patterns, ("[E]", "[EndE]"))
        self.assertEqual(config.tools, [])

    def test_config_with_many_patterns_and_sequences(self):
        """Test config with larger lists."""
        toml_data = {
            'config': {
                'zone_patterns': ["[Prompt]", "[Reasoning]", "[Answer]", "[Feedback]", "[EOS]"],
                'required_patterns': ["[Prompt]", "[Reasoning]", "[Answer]"],
                'valid_tags': ["Training", "Correct", "Incorrect1", "Incorrect2", "Feedback", "Audit"],
                'default_max_token_length': 50000,
                'sequences': ["intro", "setup", "reasoning", "solving", "feedback", "conclusion"],
                'control_pattern': "[Jump]",
                'escape_patterns': ["[Escape]", "[EndEscape]"],
                'tools': ["calculator", "web_search"]
            }
        }

        config = parse_config(toml_data)
        self.assertEqual(len(config.zone_patterns), 5)
        self.assertEqual(len(config.required_patterns), 3)
        self.assertEqual(len(config.valid_tags), 6)
        self.assertEqual(len(config.sequences), 6)
        self.assertEqual(len(config.tools), 2)

        # Check all required patterns are in zone patterns
        for pattern in config.required_patterns:
            self.assertIn(pattern, config.zone_patterns)

    def test_config_with_extra_fields_preserved(self):
        """Test that extra fields in config are preserved in misc."""
        toml_data = self.get_valid_config()
        toml_data['config']['custom_field'] = 'custom_value'
        toml_data['config']['nested'] = {'key': 'value'}
        toml_data['config']['number_field'] = 42

        config = parse_config(toml_data)

        # Should still parse successfully
        self.assertEqual(config.control_pattern, "[Jump]")

        # Extra fields should be in misc
        self.assertEqual(config.misc['config']['custom_field'], 'custom_value')
        self.assertEqual(config.misc['config']['nested'], {'key': 'value'})
        self.assertEqual(config.misc['config']['number_field'], 42)

    def test_config_with_empty_valid_tags_but_warning(self):
        """Test config that's valid but generates warning for empty tags."""
        toml_data = self.get_valid_config()
        toml_data['config']['valid_tags'] = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = parse_config(toml_data)

            # Should parse successfully
            self.assertEqual(config.valid_tags, [])

            # Should generate exactly one warning
            self.assertEqual(len(w), 1)
            self.assertIn("valid_tags", str(w[0].message).lower())
            self.assertIn("empty", str(w[0].message).lower())

    def test_num_zones_per_block_calculation(self):
        """Test that num_zones_per_block correctly calculates zones as len(zone_patterns) - 1."""
        toml_data = self.get_valid_config()
        config = parse_config(toml_data)

        # Default config has 3 zone_patterns, so should have 2 zones
        self.assertEqual(config.num_zones_per_block, 2)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)