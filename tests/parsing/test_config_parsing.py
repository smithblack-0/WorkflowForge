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
                'zone_tokens': ["[Prompt]", "[Answer]", "[EOS]"],
                'required_tokens': ["[Prompt]", "[Answer]"],
                'valid_tags': ["Training", "Correct", "Incorrect"],
                'default_max_token_length': 20000,
                'sequences': ["setup", "loop", "solving", "concluding"],
                'control_token': "[Jump]",
                'escape_token': "[Escape]"
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

        with self.assertRaisesRegex(ConfigParseError, "Missing required \\[config\\] section"):
            parse_config(toml_data)


class TestZoneTokensValidation(BaseConfigTest):
    """Test zone_tokens validation rules."""

    def test_missing_zone_tokens(self):
        """Is zone_tokens present?"""
        toml_data = self.get_valid_config()
        del toml_data['config']['zone_tokens']

        with self.assertRaisesRegex(ConfigParseError, "Missing required 'zone_tokens'"):
            parse_config(toml_data)

    def test_zone_tokens_is_list(self):
        """Is it a list of strings?"""
        toml_data = self.get_valid_config()
        toml_data['config']['zone_tokens'] = "[Prompt]"

        with self.assertRaisesRegex(ConfigParseError, "'zone_tokens' must be a list"):
            parse_config(toml_data)

    def test_zone_tokens_contains_strings(self):
        """Is it a list of strings?"""
        toml_data = self.get_valid_config()
        toml_data['config']['zone_tokens'] = ["[Prompt]", 123, "[EOS]"]

        with self.assertRaisesRegex(ConfigParseError, "All 'zone_tokens' must be strings"):
            parse_config(toml_data)

    def test_zone_tokens_nonempty(self):
        """Is it nonempty?"""
        toml_data = self.get_valid_config()
        toml_data['config']['zone_tokens'] = []

        with self.assertRaisesRegex(ConfigParseError, "'zone_tokens' cannot be empty"):
            parse_config(toml_data)

    def test_zone_tokens_length_at_least_two(self):
        """Is it of length at least two?"""
        toml_data = self.get_valid_config()
        toml_data['config']['zone_tokens'] = ["[Prompt]"]

        with self.assertRaisesRegex(ConfigParseError, "'zone_tokens' must contain at least 2 tokens"):
            parse_config(toml_data)


class TestRequiredTokensValidation(BaseConfigTest):
    """Test required_tokens validation rules."""

    def test_missing_required_tokens(self):
        """Is required_tokens present?"""
        toml_data = self.get_valid_config()
        del toml_data['config']['required_tokens']

        with self.assertRaisesRegex(ConfigParseError, "Missing requirement 'required_tokens'"):
            parse_config(toml_data)

    def test_required_tokens_is_list(self):
        """Is it a list of strings?"""
        toml_data = self.get_valid_config()
        toml_data['config']['required_tokens'] = "[Prompt]"

        with self.assertRaisesRegex(ConfigParseError, "'required_tokens' must be a list"):
            parse_config(toml_data)

    def test_required_tokens_contains_strings(self):
        """Is it a list of strings?"""
        toml_data = self.get_valid_config()
        toml_data['config']['required_tokens'] = ["[Prompt]", 456]

        with self.assertRaisesRegex(ConfigParseError, "All 'required_tokens' must be strings"):
            parse_config(toml_data)

    def test_required_tokens_nonempty(self):
        """Is it nonempty?"""
        toml_data = self.get_valid_config()
        toml_data['config']['required_tokens'] = []

        with self.assertRaisesRegex(ConfigParseError, "'required_tokens' cannot be empty"):
            parse_config(toml_data)

    def test_required_tokens_in_zone_tokens(self):
        """Are all required tokens in zone tokens?"""
        toml_data = self.get_valid_config()
        toml_data['config']['required_tokens'] = ["[Prompt]", "[Missing]"]

        with self.assertRaisesRegex(ConfigParseError, "Required token '\\[Missing\\]' not found in zone_tokens"):
            parse_config(toml_data)


class TestValidTagsValidation(BaseConfigTest):
    """Test valid_tags validation rules."""

    def test_missing_valid_tags(self):
        """Is valid_tags present?"""
        toml_data = self.get_valid_config()
        del toml_data['config']['valid_tags']

        with self.assertRaisesRegex(ConfigParseError, "Missing required 'valid_tags'"):
            parse_config(toml_data)

    def test_valid_tags_is_list(self):
        """Is it a list?"""
        toml_data = self.get_valid_config()
        toml_data['config']['valid_tags'] = "Training"

        with self.assertRaisesRegex(ConfigParseError, "'valid_tags' must be a list"):
            parse_config(toml_data)

    def test_valid_tags_contains_strings(self):
        """Are all elements strings?"""
        toml_data = self.get_valid_config()
        toml_data['config']['valid_tags'] = ["Training", 789]

        with self.assertRaisesRegex(ConfigParseError, "All 'valid_tags' must be strings"):
            parse_config(toml_data)

    def test_valid_tags_empty_warning(self):
        """Warn if empty"""
        toml_data = self.get_valid_config()
        toml_data['config']['valid_tags'] = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = parse_config(toml_data)
            self.assertEqual(len(w), 1)
            self.assertIn("valid_tags' is empty", str(w[0].message))
            self.assertEqual(config.valid_tags, [])


class TestDefaultMaxTokenLengthValidation(BaseConfigTest):
    """Test default_max_token_length validation rules."""

    def test_missing_default_max_token_length(self):
        """Is default_max_token_length present?"""
        toml_data = self.get_valid_config()
        del toml_data['config']['default_max_token_length']

        with self.assertRaisesRegex(ConfigParseError, "Missing required 'default_max_token_length'"):
            parse_config(toml_data)

    def test_default_max_token_length_is_integer(self):
        """Is it an integer > 0?"""
        toml_data = self.get_valid_config()
        toml_data['config']['default_max_token_length'] = "20000"

        with self.assertRaisesRegex(ConfigParseError, "'default_max_token_length' must be an integer"):
            parse_config(toml_data)

    def test_default_max_token_length_greater_than_zero(self):
        """Is it an integer > 0?"""
        toml_data = self.get_valid_config()
        toml_data['config']['default_max_token_length'] = 0

        with self.assertRaisesRegex(ConfigParseError, "'default_max_token_length' must be greater than 0"):
            parse_config(toml_data)

    def test_default_max_token_length_not_negative(self):
        """Is it an integer > 0?"""
        toml_data = self.get_valid_config()
        toml_data['config']['default_max_token_length'] = -100

        with self.assertRaisesRegex(ConfigParseError, "'default_max_token_length' must be greater than 0"):
            parse_config(toml_data)


class TestSequencesValidation(BaseConfigTest):
    """Test sequences validation rules."""

    def test_missing_sequences(self):
        """Is sequences present?"""
        toml_data = self.get_valid_config()
        del toml_data['config']['sequences']

        with self.assertRaisesRegex(ConfigParseError, "Missing required 'sequences'"):
            parse_config(toml_data)

    def test_sequences_is_list(self):
        """Is it a list?"""
        toml_data = self.get_valid_config()
        toml_data['config']['sequences'] = "setup"

        with self.assertRaisesRegex(ConfigParseError, "'sequences' must be a list"):
            parse_config(toml_data)

    def test_sequences_contains_strings(self):
        """Does the list contain strings?"""
        toml_data = self.get_valid_config()
        toml_data['config']['sequences'] = ["setup", 123]

        with self.assertRaisesRegex(ConfigParseError, "All 'sequences' must be strings"):
            parse_config(toml_data)

    def test_sequences_nonempty(self):
        """Is it nonempty?"""
        toml_data = self.get_valid_config()
        toml_data['config']['sequences'] = []

        with self.assertRaisesRegex(ConfigParseError, "'sequences' cannot be empty"):
            parse_config(toml_data)

    def test_sequences_strings_nonempty(self):
        """Are the strings non-empty?"""
        toml_data = self.get_valid_config()
        toml_data['config']['sequences'] = ["setup", ""]

        with self.assertRaisesRegex(ConfigParseError, "All 'sequences' must be non-empty strings"):
            parse_config(toml_data)

    def test_sequences_strings_not_whitespace_only(self):
        """Are the strings non-empty (not just whitespace)?"""
        toml_data = self.get_valid_config()
        toml_data['config']['sequences'] = ["setup", "   "]

        with self.assertRaisesRegex(ConfigParseError, "All 'sequences' must be non-empty strings"):
            parse_config(toml_data)


class TestControlTokenValidation(BaseConfigTest):
    """Test control_token validation rules."""

    def test_missing_control_token(self):
        """Is control_token present?"""
        toml_data = self.get_valid_config()
        del toml_data['config']['control_token']

        with self.assertRaisesRegex(ConfigParseError, "Missing required 'control_token'"):
            parse_config(toml_data)

    def test_control_token_is_string(self):
        """Does it contain a string?"""
        toml_data = self.get_valid_config()
        toml_data['config']['control_token'] = 123

        with self.assertRaisesRegex(ConfigParseError, "'control_token' must be a string"):
            parse_config(toml_data)

    def test_control_token_nonempty(self):
        """Is that string nonempty?"""
        toml_data = self.get_valid_config()
        toml_data['config']['control_token'] = ""

        with self.assertRaisesRegex(ConfigParseError, "'control_token' cannot be empty"):
            parse_config(toml_data)

    def test_control_token_not_whitespace_only(self):
        """Is that string nonempty (not just whitespace)?"""
        toml_data = self.get_valid_config()
        toml_data['config']['control_token'] = "   "

        with self.assertRaisesRegex(ConfigParseError, "'control_token' cannot be empty"):
            parse_config(toml_data)


class TestEscapeTokenValidation(BaseConfigTest):
    """Test escape_token validation rules."""

    def test_missing_escape_token(self):
        """Is escape_token present?"""
        toml_data = self.get_valid_config()
        del toml_data['config']['escape_token']

        with self.assertRaisesRegex(ConfigParseError, "Missing required 'escape_token'"):
            parse_config(toml_data)

    def test_escape_token_is_string(self):
        """Does it contain a string?"""
        toml_data = self.get_valid_config()
        toml_data['config']['escape_token'] = 456

        with self.assertRaisesRegex(ConfigParseError, "'escape_token' must be a string"):
            parse_config(toml_data)

    def test_escape_token_nonempty(self):
        """Is that string nonempty?"""
        toml_data = self.get_valid_config()
        toml_data['config']['escape_token'] = ""

        with self.assertRaisesRegex(ConfigParseError, "'escape_token' cannot be empty"):
            parse_config(toml_data)

    def test_escape_token_not_whitespace_only(self):
        """Is that string nonempty (not just whitespace)?"""
        toml_data = self.get_valid_config()
        toml_data['config']['escape_token'] = "   "

        with self.assertRaisesRegex(ConfigParseError, "'escape_token' cannot be empty"):
            parse_config(toml_data)


class TestValidConfigSequences(BaseConfigTest):
    """Test various valid configuration scenarios."""

    def test_complete_valid_config(self):
        """Test parsing a completely valid config from the example."""
        toml_data = self.get_valid_config()
        config = parse_config(toml_data)

        # Check all basic fields
        self.assertEqual(config.zone_tokens, ["[Prompt]", "[Answer]", "[EOS]"])
        self.assertEqual(config.required_tokens, ["[Prompt]", "[Answer]"])
        self.assertEqual(config.valid_tags, ["Training", "Correct", "Incorrect"])
        self.assertEqual(config.default_max_token_length, 20000)
        self.assertEqual(config.sequences, ["setup", "loop", "solving", "concluding"])
        self.assertEqual(config.control_token, "[Jump]")
        self.assertEqual(config.escape_token, "[Escape]")

        # Check computed special_tokens
        expected_special = {"[Prompt]", "[Answer]", "[EOS]", "[Jump]", "[Escape]"}
        self.assertEqual(set(config.special_tokens), expected_special)

        # Check misc contains the full toml data
        self.assertEqual(config.misc, toml_data)

    def test_minimal_valid_config(self):
        """Test a minimal but valid config."""
        toml_data = {
            'config': {
                'zone_tokens': ["[A]", "[B]"],
                'required_tokens': ["[A]"],
                'valid_tags': ["Tag1"],
                'default_max_token_length': 1,
                'sequences': ["seq1"],
                'control_token': "[C]",
                'escape_token': "[E]"
            }
        }

        config = parse_config(toml_data)
        self.assertEqual(config.zone_tokens, ["[A]", "[B]"])
        self.assertEqual(config.required_tokens, ["[A]"])
        self.assertEqual(config.valid_tags, ["Tag1"])
        self.assertEqual(config.default_max_token_length, 1)
        self.assertEqual(config.sequences, ["seq1"])
        self.assertEqual(config.control_token, "[C]")
        self.assertEqual(config.escape_token, "[E]")

        expected_special = {"[A]", "[B]", "[C]", "[E]"}
        self.assertEqual(set(config.special_tokens), expected_special)

    def test_config_with_many_tokens_and_sequences(self):
        """Test config with larger lists."""
        toml_data = {
            'config': {
                'zone_tokens': ["[Prompt]", "[Reasoning]", "[Answer]", "[Feedback]", "[EOS]"],
                'required_tokens': ["[Prompt]", "[Reasoning]", "[Answer]"],
                'valid_tags': ["Training", "Correct", "Incorrect1", "Incorrect2", "Feedback", "Audit"],
                'default_max_token_length': 50000,
                'sequences': ["intro", "setup", "reasoning", "solving", "feedback", "conclusion"],
                'control_token': "[Jump]",
                'escape_token': "[Escape]"
            }
        }

        config = parse_config(toml_data)
        self.assertEqual(len(config.zone_tokens), 5)
        self.assertEqual(len(config.required_tokens), 3)
        self.assertEqual(len(config.valid_tags), 6)
        self.assertEqual(len(config.sequences), 6)

        # Check all required tokens are in zone tokens
        for token in config.required_tokens:
            self.assertIn(token, config.zone_tokens)

    def test_special_tokens_no_duplicates(self):
        """Test that special_tokens doesn't contain duplicates when tokens overlap."""
        toml_data = self.get_valid_config()
        # Set control_token to something already in zone_tokens
        toml_data['config']['control_token'] = "[Prompt]"

        config = parse_config(toml_data)

        # Should not duplicate [Prompt]
        self.assertEqual(config.special_tokens.count("[Prompt]"), 1)
        self.assertIn("[Answer]", config.special_tokens)
        self.assertIn("[EOS]", config.special_tokens)
        self.assertIn("[Escape]", config.special_tokens)

    def test_config_with_extra_fields_preserved(self):
        """Test that extra fields in config are preserved in misc."""
        toml_data = self.get_valid_config()
        toml_data['config']['custom_field'] = 'custom_value'
        toml_data['config']['nested'] = {'key': 'value'}
        toml_data['config']['number_field'] = 42

        config = parse_config(toml_data)

        # Should still parse successfully
        self.assertEqual(config.control_token, "[Jump]")

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
            self.assertIn("valid_tags' is empty", str(w[0].message))
            self.assertTrue(issubclass(w[0].category, UserWarning))


    def test_num_zones_per_block_calculation(self):
        """Test that num_zones_per_block correctly calculates zones as len(zone_tokens) - 1."""
        toml_data = self.get_valid_config()
        config = parse_config(toml_data)

        # Default config has 3 zone_tokens, so should have 2 zones
        self.assertEqual(config.num_zones_per_block, 2)

if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)