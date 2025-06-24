"""
Unit tests for UDPL Zone Parser

Tests cover:
1. Basic functionality - core features working correctly
2. Validation error tests - input validation catching problems
3. Construction callback tests - testing callback function in isolation
4. Error handling tests - ensuring failures are reported usefully
"""

import unittest
import warnings
from unittest.mock import Mock
from typing import Dict, Any, Optional

# Import the modules under test
from src.workflow_forge.parsing.zone_parsing import (
    parse_zone, validate_flow_control_safety, extract_placeholders,
    build_resource_specs, create_construction_callback, ZoneParseError
)
from src.workflow_forge.parsing.config_parsing import Config
from src.workflow_forge.parsing.block_parsing import ZoneInfo
from src.workflow_forge.zcp.nodes import ZCPNode
from src.workflow_forge.resources import AbstractResource


class BaseZoneParserTest(unittest.TestCase):
    """Base test class with common setup and helper methods."""

    def setUp(self):
        """Set up common test fixtures."""
        pass  # No complex mocks needed for zone parser

    def get_valid_config(self, **overrides) -> Config:
        """
        Return valid config for testing, with optional field overrides.

        Args:
            **overrides: Fields to override in the base config

        Returns:
            Config object with valid settings
        """
        base_config = {
            'zone_patterns': ["[Prompt]", "[Answer]", "[EOS]"],
            'required_patterns': ["[Prompt]", "[Answer]"],
            'valid_tags': ["Training", "Correct"],
            'default_max_token_length': 1000,
            'sequences': ["test_sequence"],
            'control_pattern': "[Jump]",
            'escape_patterns': ("[Escape]", "[EndEscape]"),
            'tools': ["search", "calculator"],
            'misc': {}
        }
        base_config.update(overrides)
        return Config(**base_config)

    def get_valid_zone_info(self, **overrides) -> ZoneInfo:
        """
        Return valid zone info for testing, with optional field overrides.

        Args:
            **overrides: Fields to override in the base zone info

        Returns:
            ZoneInfo object with valid settings
        """
        base_zone_info = {
            'advance_token': "[Answer]",
            'zone_text': "What is AI?",
            'tags': ["Training"],
            'sequence_name': "test_sequence",
            'block_index': 0,
            'zone_index': 1,
            'max_gen_tokens': 500,
            'block_data': {}
        }
        base_zone_info.update(overrides)
        return ZoneInfo(**base_zone_info)

    def create_zone_info(self, **overrides) -> ZoneInfo:
        """
        Create a ZoneInfo with valid data and optional overrides.

        Args:
            **overrides: Fields to override in the base zone info

        Returns:
            Configured ZoneInfo instance
        """
        return self.get_valid_zone_info(**overrides)

    def create_zone_with_placeholders(self, placeholders: Dict[str, Dict[str, Any]],
                                    zone_text: Optional[str] = None, **overrides) -> ZoneInfo:
        """
        Create ZoneInfo with placeholder configuration.

        Args:
            placeholders: Dictionary mapping placeholder names to their resource specs
            zone_text: Custom zone text (auto-generated if None)
            **overrides: Additional ZoneInfo field overrides

        Returns:
            ZoneInfo configured with placeholders
        """
        if zone_text is None:
            placeholder_names = list(placeholders.keys())
            zone_text = "Text with " + " and ".join(f"{{{name}}}" for name in placeholder_names)

        zone_overrides = {
            'zone_text': zone_text,
            'block_data': placeholders
        }
        zone_overrides.update(overrides)
        return self.create_zone_info(**zone_overrides)

    def create_mock_resource(self, return_value: str = "mock_result") -> Mock:
        """
        Create a mock resource with proper spec.

        Args:
            return_value: Value the mock resource should return

        Returns:
            Mock resource object
        """
        mock_resource = Mock(spec=AbstractResource)
        mock_resource.return_value = return_value
        return mock_resource

    def assert_zone_parse_error_context(self, context_manager, expected_sequence: str,
                                      expected_block: int, expected_zone: int):
        """
        Assert that a ZoneParseError has the expected context information.

        Args:
            context_manager: The exception context manager
            expected_sequence: Expected sequence name in error
            expected_block: Expected block number in error
            expected_zone: Expected zone number in error
        """
        error_msg = str(context_manager.exception).lower()
        self.assertIn(f"zone {expected_zone}", error_msg)
        self.assertIn(f"sequence", error_msg)
        self.assertIn(expected_sequence.lower(), error_msg)
        self.assertIn(f"block {expected_block}", error_msg)

    def assert_resource_specs_equal(self, actual: Dict[str, Dict[str, Any]],
                                  expected: Dict[str, Dict[str, Any]], msg: str = None):
        """
        Assert resource specs are equal with better error messages.

        Args:
            actual: Actual resource specs
            expected: Expected resource specs
            msg: Optional message prefix
        """
        self.assertEqual(actual, expected, msg)

    def assert_construction_callback_works(self, callback, resources: Dict[str, AbstractResource],
                                         expected_result: str):
        """
        Assert that a construction callback works correctly.

        Args:
            callback: The construction callback to test
            resources: Resources to pass to callback
            expected_result: Expected callback result
        """
        result = callback(resources)
        self.assertEqual(result, expected_result)


class TestBasicFunctionality(BaseZoneParserTest):
    """Test core zone parser features working correctly."""

    def test_zone_with_no_placeholders(self):
        """Test basic zone with no placeholders."""
        config = self.get_valid_config()
        zone_info = self.create_zone_info()

        result = parse_zone(zone_info, config)

        # Verify ZCPNode structure
        self.assertIsInstance(result, ZCPNode)
        self.assertEqual(result.sequence, "test_sequence")
        self.assertEqual(result.block, 0)
        self.assertEqual(result.raw_text, "What is AI?")
        self.assertEqual(result.zone_advance_str, "[Answer]")
        self.assertEqual(result.tags, ["Training"])
        self.assertEqual(result.timeout, 500)
        self.assertEqual(result.resource_specs, {})
        self.assertIsNone(result.next_zone)

        # Test construction callback with no placeholders
        final_text = result.construction_callback({})
        self.assertEqual(final_text, "What is AI?")

    def test_zone_with_single_placeholder(self):
        """Test zone with single placeholder and resource spec."""
        config = self.get_valid_config()
        placeholders = {
            "feedback": {
                "name": "feedback_resource",
                "arguments": {"num_samples": 3}
            }
        }
        zone_info = self.create_zone_with_placeholders(
            placeholders,
            zone_text="Consider this feedback: {feedback}"
        )

        result = parse_zone(zone_info, config)

        # Verify resource specs
        expected_spec = {
            "feedback": {
                "name": "feedback_resource",
                "arguments": {"num_samples": 3},
                "type": "default"
            }
        }
        self.assert_resource_specs_equal(result.resource_specs, expected_spec)

        # Test construction callback
        mock_resource = self.create_mock_resource("Great work!")
        resources = {"feedback_resource": mock_resource}

        self.assert_construction_callback_works(
            result.construction_callback,
            resources,
            "Consider this feedback: Great work!"
        )
        mock_resource.assert_called_once_with(num_samples=3)

    def test_zone_with_multiple_placeholders(self):
        """Test zone with multiple placeholders."""
        config = self.get_valid_config()
        placeholders = {
            "principle": {
                "name": "constitution_overview"
            },
            "details": {
                "name": "constitution_details",
                "arguments": {"num_samples": 2}
            }
        }
        zone_info = self.create_zone_with_placeholders(
            placeholders,
            zone_text="Follow {principle} and consider {details}"
        )

        result = parse_zone(zone_info, config)

        # Verify both resource specs
        expected_specs = {
            "principle": {
                "name": "constitution_overview",
                "arguments": None,
                "type": "default"
            },
            "details": {
                "name": "constitution_details",
                "arguments": {"num_samples": 2},
                "type": "default"
            }
        }
        self.assert_resource_specs_equal(result.resource_specs, expected_specs)

    def test_zone_with_custom_resource_type(self):
        """Test zone with custom resource type."""
        config = self.get_valid_config()
        placeholders = {
            "min_value": {
                "name": "min_control",
                "type": "flow_control"
            }
        }
        zone_info = self.create_zone_with_placeholders(
            placeholders,
            zone_text="Repeat at least {min_value} times"
        )

        result = parse_zone(zone_info, config)

        # Verify custom type is preserved
        expected_spec = {
            "min_value": {
                "name": "min_control",
                "arguments": None,
                "type": "flow_control"
            }
        }
        self.assert_resource_specs_equal(result.resource_specs, expected_spec)


class TestValidationErrors(BaseZoneParserTest):
    """Test input validation catching problems."""

    def test_malformed_placeholder_syntax(self):
        """Test error for malformed placeholder syntax."""
        config = self.get_valid_config()
        zone_info = self.create_zone_info(zone_text="Bad placeholder {unclosed syntax")

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, config)

        error_msg = str(context.exception).lower()
        self.assertIn("malformed", error_msg)
        self.assertIn("placeholder", error_msg)

    def test_missing_resource_specification(self):
        """Test error when placeholder has no resource spec."""
        config = self.get_valid_config()
        zone_info = self.create_zone_info(
            zone_text="Missing spec: {missing_placeholder}",
            block_data={}  # No spec for missing_placeholder
        )

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, config)

        error_msg = str(context.exception).lower()
        self.assertIn("missing", error_msg)
        self.assertIn("resource", error_msg)
        self.assertIn("placeholder", error_msg)
        self.assertIn("missing_placeholder", error_msg)

    def test_invalid_resource_spec_structure(self):
        """Test error when resource spec is not a dictionary."""
        config = self.get_valid_config()
        zone_info = self.create_zone_info(
            zone_text="Bad spec: {bad_spec}",
            block_data={"bad_spec": "not_a_dict"}  # Should be a dictionary
        )

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, config)

        error_msg = str(context.exception).lower()
        self.assertIn("resource", error_msg)
        self.assertIn("dictionary", error_msg)
        self.assertIn("bad_spec", error_msg)

    def test_missing_name_field(self):
        """Test error when resource spec missing name field."""
        config = self.get_valid_config()
        placeholders = {
            "no_name": {
                "arguments": {"test": "value"}
                # Missing 'name' field
            }
        }
        zone_info = self.create_zone_with_placeholders(
            placeholders,
            zone_text="No name: {no_name}"
        )

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, config)

        error_msg = str(context.exception).lower()
        self.assertIn("missing", error_msg)
        self.assertIn("name", error_msg)
        self.assertIn("no_name", error_msg)

    def test_invalid_field_types(self):
        """Test errors for invalid field types in resource spec."""
        config = self.get_valid_config()
        placeholders = {
            "bad_name": {
                "name": 123  # Should be string
            }
        }
        zone_info = self.create_zone_with_placeholders(
            placeholders,
            zone_text="Bad name: {bad_name}"
        )

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, config)

        error_msg = str(context.exception).lower()
        self.assertIn("name", error_msg)
        self.assertIn("string", error_msg)
        self.assertIn("bad_name", error_msg)

    def test_unescaped_flow_control_token(self):
        """Test warning for unescaped flow control token."""
        config = self.get_valid_config()
        zone_info = self.create_zone_info(
            zone_text="This will [Jump] without escape",
            sequence_name="test_sequence",
            block_index=2
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = parse_zone(zone_info, config)  # Should succeed with warning

            # Verify it still creates a valid ZCPNode
            self.assertIsInstance(result, ZCPNode)

            # Verify warning was issued with key concepts
            self.assertTrue(len(w) > 0, "Expected warning to be issued")
            warning_msg = str(w[0].message).lower()
            self.assertIn("jump", warning_msg)
            self.assertIn("test_sequence", warning_msg)
            self.assertIn("block", warning_msg)

    def test_properly_escaped_flow_control_token(self):
        """Test that properly escaped flow control token passes validation."""
        config = self.get_valid_config()
        zone_info = self.create_zone_info(
            zone_text="Use [Escape] [Jump] [EndEscape] to jump"
        )

        # Should not raise exception
        result = parse_zone(zone_info, config)
        self.assertIsInstance(result, ZCPNode)


class TestConstructionCallback(BaseZoneParserTest):
    """Test the construction callback function in isolation."""

    def test_callback_resolves_placeholders_correctly(self):
        """Test callback resolves all placeholders with mock resources."""
        resource_specs = {
            "principle": {
                "name": "constitution",
                "arguments": None,
                "type": "default"
            },
            "count": {
                "name": "counter",
                "arguments": {"start": 5},
                "type": "default"
            }
        }

        callback = create_construction_callback(
            "Follow {principle} and repeat {count} times",
            resource_specs
        )

        # Create mock resources
        mock_constitution = self.create_mock_resource("safety first")
        mock_counter = self.create_mock_resource("3")

        resources = {
            "constitution": mock_constitution,
            "counter": mock_counter
        }

        self.assert_construction_callback_works(
            callback,
            resources,
            "Follow safety first and repeat 3 times"
        )
        mock_constitution.assert_called_once_with()
        mock_counter.assert_called_once_with(start=5)

    def test_callback_handles_missing_resources(self):
        """Test callback raises ValueError for missing resources."""
        resource_specs = {
            "missing": {
                "name": "missing_resource",
                "arguments": None,
                "type": "default"
            }
        }

        callback = create_construction_callback(
            "Need {missing} resource",
            resource_specs
        )

        with self.assertRaises(ValueError) as context:
            callback({})  # Empty resources dict

        error_msg = str(context.exception).lower()
        self.assertIn("required", error_msg)
        self.assertIn("resource", error_msg)
        self.assertIn("missing_resource", error_msg)
        self.assertIn("not found", error_msg)

    def test_callback_handles_resource_call_failures(self):
        """Test callback handles exceptions from resource calls."""
        resource_specs = {
            "failing": {
                "name": "failing_resource",
                "arguments": {"param": "value"},
                "type": "default"
            }
        }

        callback = create_construction_callback(
            "This {failing} will fail",
            resource_specs
        )

        mock_resource = Mock(spec=AbstractResource)
        mock_resource.side_effect = RuntimeError("Resource failed!")

        resources = {"failing_resource": mock_resource}

        with self.assertRaises(ValueError) as context:
            callback(resources)

        error_msg = str(context.exception).lower()
        self.assertIn("error", error_msg)
        self.assertIn("resource", error_msg)
        self.assertIn("failing_resource", error_msg)

    def test_callback_handles_extra_placeholders_in_text(self):
        """Test callback handles placeholders in text but not in specs."""
        resource_specs = {}  # No specs provided

        callback = create_construction_callback(
            "This has {unexpected} placeholder",
            resource_specs
        )

        with self.assertRaises(ValueError) as context:
            callback({})

        error_msg = str(context.exception).lower()
        self.assertIn("placeholder", error_msg)
        self.assertIn("unexpected", error_msg)
        self.assertIn("found", error_msg)


class TestErrorHandling(BaseZoneParserTest):
    """Test that failures are reported usefully."""

    def test_error_messages_include_context(self):
        """Test that error messages include sequence/block/zone context."""
        config = self.get_valid_config()
        zone_info = self.create_zone_info(
            zone_text="Bad placeholder {unclosed",
            sequence_name="my_sequence",
            block_index=3,
            zone_index=7
        )

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, config)

        self.assert_zone_parse_error_context(context, "my_sequence", 3, 7)

    def test_exception_chaining_works(self):
        """Test that exception chaining preserves original errors."""
        config = self.get_valid_config()
        zone_info = self.create_zone_info(zone_text="Bad placeholder {unclosed")

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, config)

        # Check that original exception is chained
        self.assertIsNotNone(context.exception.__cause__)

    def test_validation_error_context(self):
        """Test that validation errors include proper context."""
        config = self.get_valid_config()
        zone_info = self.create_zone_info(
            zone_text="This will [Jump] without escape",
            sequence_name="error_sequence",
            block_index=5,
            zone_index=2
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = parse_zone(zone_info, config)  # Should succeed with warning

            # Verify warning contains context
            self.assertTrue(len(w) > 0, "Expected warning to be issued")
            warning_msg = str(w[0].message).lower()
            self.assertIn("error_sequence", warning_msg)
            self.assertIn("block", warning_msg)


class TestHelperFunctions(BaseZoneParserTest):
    """Test individual helper functions."""

    def test_extract_placeholders(self):
        """Test placeholder extraction from text."""
        # Test normal placeholders
        placeholders = extract_placeholders("Text with {placeholder1} and {placeholder2}")
        self.assertEqual(set(placeholders), {"placeholder1", "placeholder2"})

        # Test no placeholders
        placeholders = extract_placeholders("Text with no placeholders")
        self.assertEqual(placeholders, [])

        # Test duplicate placeholders
        placeholders = extract_placeholders("Text {same} and {same} again")
        self.assertEqual(placeholders, ["same"])

    def test_build_resource_specs(self):
        """Test resource spec building from block data."""
        placeholders = ["feedback", "count"]
        block_data = {
            "feedback": {
                "name": "feedback_resource",
                "arguments": {"num": 3}
            },
            "count": {
                "name": "counter_resource",
                "type": "flow_control"
            }
        }

        specs = build_resource_specs(placeholders, block_data, "test_seq")

        expected = {
            "feedback": {
                "name": "feedback_resource",
                "arguments": {"num": 3},
                "type": "default"
            },
            "count": {
                "name": "counter_resource",
                "arguments": None,
                "type": "flow_control"
            }
        }
        self.assert_resource_specs_equal(specs, expected)

    def test_validate_flow_control_safety(self):
        """Test flow control validation function."""
        config = self.get_valid_config()

        # Test safe case - no control pattern
        zone_info = self.create_zone_info(zone_text="Safe text")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_flow_control_safety("Safe text", config, zone_info)  # Should not warn
            self.assertEqual(len(w), 0, "No warnings expected for safe text")

        # Test safe case - properly escaped with both patterns
        escaped_text = "Use [Escape] [Jump] [EndEscape] to jump"
        zone_info = self.create_zone_info(zone_text=escaped_text)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_flow_control_safety(escaped_text, config, zone_info)  # Should not warn
            self.assertEqual(len(w), 0, "No warnings expected for properly escaped text")

        # Test unsafe case - unescaped control pattern
        unsafe_text = "This will [Jump] immediately"
        zone_info = self.create_zone_info(zone_text=unsafe_text)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_flow_control_safety(unsafe_text, config, zone_info)  # Should warn
            self.assertTrue(len(w) > 0, "Expected warning for unescaped control pattern")
            warning_msg = str(w[0].message).lower()
            self.assertIn("jump", warning_msg)


if __name__ == "__main__":
    unittest.main()