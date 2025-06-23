"""
Unit tests for UDPL Zone Parser

Tests cover:
1. Basic functionality - core features working correctly
2. Validation error tests - input validation catching problems
3. Construction callback tests - testing callback function in isolation
4. Error handling tests - ensuring failures are reported usefully
"""

import unittest
from unittest.mock import Mock
from typing import Dict, Any

# Import the modules under test
from src.workflow_forge.parsing.zone_parsing import (
    parse_zone, validate_flow_control_safety, extract_placeholders,
    build_resource_specs, create_construction_callback, ZoneParseError
)
from src.workflow_forge.parsing.config_parsing import Config
from src.workflow_forge.parsing.block_parsing import ZoneInfo
from src.workflow_forge.zcp.nodes import ZCPNode
from src.workflow_forge.resources import AbstractResource


class TestBasicFunctionality(unittest.TestCase):
    """Test core zone parser features working correctly."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            zone_patterns=["[Prompt]", "[Answer]", "[EOS]"],
            required_patterns=["[Prompt]", "[Answer]"],
            valid_tags=["Training", "Correct"],
            default_max_token_length=1000,
            sequences=["test_sequence"],
            control_pattern="[Jump]",
            escape_token="[Escape]",
            special_patterns=["[Prompt]", "[Answer]", "[EOS]", "[Jump]", "[Escape]"],
            misc={}
        )

    def test_zone_with_no_placeholders(self):
        """Test basic zone with no placeholders."""
        zone_info = ZoneInfo(
            advance_token="[Answer]",
            zone_text="What is AI?",
            tags=["Training"],
            sequence_name="test_sequence",
            block_index=0,
            zone_index=1,
            max_gen_tokens=500,
            block_data={}
        )

        result = parse_zone(zone_info, self.config)

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
        block_data = {
            "feedback": {
                "name": "feedback_resource",
                "arguments": {"num_samples": 3}
            }
        }

        zone_info = ZoneInfo(
            advance_token="[Answer]",
            zone_text="Consider this feedback: {feedback}",
            tags=["Training"],
            sequence_name="test_sequence",
            block_index=0,
            zone_index=1,
            max_gen_tokens=500,
            block_data=block_data
        )

        result = parse_zone(zone_info, self.config)

        # Verify resource specs
        expected_spec = {
            "feedback": {
                "name": "feedback_resource",
                "arguments": {"num_samples": 3},
                "type": "default"
            }
        }
        self.assertEqual(result.resource_specs, expected_spec)

        # Test construction callback
        mock_resource = Mock(spec=AbstractResource)
        mock_resource.return_value = "Great work!"
        resources = {"feedback_resource": mock_resource}

        final_text = result.construction_callback(resources)
        self.assertEqual(final_text, "Consider this feedback: Great work!")
        mock_resource.assert_called_once_with(num_samples=3)

    def test_zone_with_multiple_placeholders(self):
        """Test zone with multiple placeholders."""
        block_data = {
            "principle": {
                "name": "constitution_overview"
            },
            "details": {
                "name": "constitution_details",
                "arguments": {"num_samples": 2}
            }
        }

        zone_info = ZoneInfo(
            advance_token="[Answer]",
            zone_text="Follow {principle} and consider {details}",
            tags=["Training"],
            sequence_name="test_sequence",
            block_index=0,
            zone_index=1,
            max_gen_tokens=500,
            block_data=block_data
        )

        result = parse_zone(zone_info, self.config)

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
        self.assertEqual(result.resource_specs, expected_specs)

    def test_zone_with_custom_resource_type(self):
        """Test zone with custom resource type."""
        block_data = {
            "min_value": {
                "name": "min_control",
                "type": "flow_control"
            }
        }

        zone_info = ZoneInfo(
            advance_token="[Answer]",
            zone_text="Repeat at least {min_value} times",
            tags=[],
            sequence_name="test_sequence",
            block_index=0,
            zone_index=1,
            max_gen_tokens=500,
            block_data=block_data
        )

        result = parse_zone(zone_info, self.config)

        # Verify custom type is preserved
        expected_spec = {
            "min_value": {
                "name": "min_control",
                "arguments": None,
                "type": "flow_control"
            }
        }
        self.assertEqual(result.resource_specs, expected_spec)


class TestValidationErrors(unittest.TestCase):
    """Test input validation catching problems."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            zone_patterns=["[Prompt]", "[Answer]", "[EOS]"],
            required_patterns=["[Prompt]", "[Answer]"],
            valid_tags=["Training"],
            default_max_token_length=1000,
            sequences=["test_sequence"],
            control_pattern="[Jump]",
            escape_token="[Escape]",
            special_patterns=["[Prompt]", "[Answer]", "[EOS]", "[Jump]", "[Escape]"],
            misc={}
        )

    def test_malformed_placeholder_syntax(self):
        """Test error for malformed placeholder syntax."""
        zone_info = ZoneInfo(
            advance_token="[Answer]",
            zone_text="Bad placeholder {unclosed syntax",
            tags=[],
            sequence_name="test_sequence",
            block_index=0,
            zone_index=1,
            max_gen_tokens=500,
            block_data={}
        )

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, self.config)

        self.assertIn("Malformed placeholder syntax", str(context.exception))

    def test_missing_resource_specification(self):
        """Test error when placeholder has no resource spec."""
        zone_info = ZoneInfo(
            advance_token="[Answer]",
            zone_text="Missing spec: {missing_placeholder}",
            tags=[],
            sequence_name="test_sequence",
            block_index=0,
            zone_index=1,
            max_gen_tokens=500,
            block_data={}  # No spec for missing_placeholder
        )

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, self.config)

        self.assertIn("Missing resource specification for placeholder 'missing_placeholder'", str(context.exception))
        self.assertIn("test_sequence.missing_placeholder", str(context.exception))

    def test_invalid_resource_spec_structure(self):
        """Test error when resource spec is not a dictionary."""
        block_data = {
            "bad_spec": "not_a_dict"  # Should be a dictionary
        }

        zone_info = ZoneInfo(
            advance_token="[Answer]",
            zone_text="Bad spec: {bad_spec}",
            tags=[],
            sequence_name="test_sequence",
            block_index=0,
            zone_index=1,
            max_gen_tokens=500,
            block_data=block_data
        )

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, self.config)

        self.assertIn("Resource specification for placeholder 'bad_spec' must be a dictionary", str(context.exception))

    def test_missing_name_field(self):
        """Test error when resource spec missing name field."""
        block_data = {
            "no_name": {
                "arguments": {"test": "value"}
                # Missing 'name' field
            }
        }

        zone_info = ZoneInfo(
            advance_token="[Answer]",
            zone_text="No name: {no_name}",
            tags=[],
            sequence_name="test_sequence",
            block_index=0,
            zone_index=1,
            max_gen_tokens=500,
            block_data=block_data
        )

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, self.config)

        self.assertIn("Resource specification for placeholder 'no_name' missing required 'name' field", str(context.exception))

    def test_invalid_field_types(self):
        """Test errors for invalid field types in resource spec."""
        # Test non-string name
        block_data = {
            "bad_name": {
                "name": 123  # Should be string
            }
        }

        zone_info = ZoneInfo(
            advance_token="[Answer]",
            zone_text="Bad name: {bad_name}",
            tags=[],
            sequence_name="test_sequence",
            block_index=0,
            zone_index=1,
            max_gen_tokens=500,
            block_data=block_data
        )

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, self.config)

        self.assertIn("Resource 'name' for placeholder 'bad_name' must be a string", str(context.exception))

    def test_unescaped_flow_control_token(self):
        """Test warning for unescaped flow control token."""
        zone_info = ZoneInfo(
            advance_token="[Answer]",
            zone_text="This will [Jump] without escape",
            tags=[],
            sequence_name="test_sequence",
            block_index=2,
            zone_index=1,
            max_gen_tokens=500,
            block_data={}
        )

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, self.config)

        error_msg = str(context.exception)
        self.assertIn("In block 2 of sequence 'test_sequence'", error_msg)
        self.assertIn("[Jump]", error_msg)
        self.assertIn("[Escape]", error_msg)
        self.assertIn("Do you mean to teacher-force flow control?", error_msg)

    def test_properly_escaped_flow_control_token(self):
        """Test that properly escaped flow control token passes validation."""
        zone_info = ZoneInfo(
            advance_token="[Answer]",
            zone_text="Use [Escape] [Jump] to jump",
            tags=[],
            sequence_name="test_sequence",
            block_index=0,
            zone_index=1,
            max_gen_tokens=500,
            block_data={}
        )

        # Should not raise exception
        result = parse_zone(zone_info, self.config)
        self.assertIsInstance(result, ZCPNode)


class TestConstructionCallback(unittest.TestCase):
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
        mock_constitution = Mock(spec=AbstractResource)
        mock_constitution.return_value = "safety first"
        mock_counter = Mock(spec=AbstractResource)
        mock_counter.return_value = "3"

        resources = {
            "constitution": mock_constitution,
            "counter": mock_counter
        }

        result = callback(resources)

        self.assertEqual(result, "Follow safety first and repeat 3 times")
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

        self.assertIn("Required resource 'missing_resource' for placeholder 'missing' not found", str(context.exception))

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

        self.assertIn("Error calling resource 'failing_resource' for placeholder 'failing'", str(context.exception))
        self.assertIn("Resource failed!", str(context.exception))

    def test_callback_handles_extra_placeholders_in_text(self):
        """Test callback handles placeholders in text but not in specs."""
        resource_specs = {}  # No specs provided

        callback = create_construction_callback(
            "This has {unexpected} placeholder",
            resource_specs
        )

        with self.assertRaises(ValueError) as context:
            callback({})

        self.assertIn("Placeholder 'unexpected' found in text but no resource specification provided", str(context.exception))


class TestErrorHandling(unittest.TestCase):
    """Test that failures are reported usefully."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            zone_patterns=["[Prompt]", "[Answer]", "[EOS]"],
            required_patterns=["[Prompt]", "[Answer]"],
            valid_tags=["Training"],
            default_max_token_length=1000,
            sequences=["test_sequence"],
            control_pattern="[Jump]",
            escape_token="[Escape]",
            special_patterns=["[Prompt]", "[Answer]", "[EOS]", "[Jump]", "[Escape]"],
            misc={}
        )

    def test_error_messages_include_context(self):
        """Test that error messages include sequence/block/zone context."""
        zone_info = ZoneInfo(
            advance_token="[Answer]",
            zone_text="Bad placeholder {unclosed",
            tags=[],
            sequence_name="my_sequence",
            block_index=3,
            zone_index=7,
            max_gen_tokens=500,
            block_data={}
        )

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, self.config)

        error_msg = str(context.exception)
        self.assertIn("Error parsing zone 7", error_msg)
        self.assertIn("sequence 'my_sequence'", error_msg)
        self.assertIn("block 3", error_msg)

    def test_exception_chaining_works(self):
        """Test that exception chaining preserves original errors."""
        zone_info = ZoneInfo(
            advance_token="[Answer]",
            zone_text="Bad placeholder {unclosed",
            tags=[],
            sequence_name="test_sequence",
            block_index=0,
            zone_index=1,
            max_gen_tokens=500,
            block_data={}
        )

        with self.assertRaises(ZoneParseError) as context:
            parse_zone(zone_info, self.config)

        # Check that original exception is chained
        self.assertIsNotNone(context.exception.__cause__)


class TestHelperFunctions(unittest.TestCase):
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
        self.assertEqual(specs, expected)


if __name__ == "__main__":
    unittest.main()