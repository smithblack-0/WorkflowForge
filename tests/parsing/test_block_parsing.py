"""
Unit tests for UDPL Block Parser

Tests cover:
1. Block structure validation
2. Text parsing into zones
3. Tags/tagset/repeats validation and processing
4. Zone orchestration and chaining
5. Error handling and exception chaining
"""

import unittest
import warnings
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

# Import the modules under test
from src.workflow_forge.parsing.block_parsing import (
    parse_block, validate_block_structure, parse_text_into_zones,
    resolve_and_validate_tags, BlockParseError, ZoneInfo
)
from src.workflow_forge.parsing.config_parsing import Config
from src.workflow_forge.zcp.nodes import ZCPNode


class TestBlockStructureValidation(unittest.TestCase):
    """Test block-level structure validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            zone_tokens=["[Prompt]", "[Answer]", "[EOS]"],
            required_tokens=["[Prompt]", "[Answer]"],
            valid_tags=["Training", "Correct", "Incorrect"],
            default_max_token_length=1000,
            sequences=["test_sequence"],
            control_token="[Jump]",
            escape_token="[Escape]",
            special_tokens=["[Prompt]", "[Answer]", "[EOS]", "[Jump]", "[Escape]"],
            misc={}
        )

    def test_missing_text_field(self):
        """Test error when text field is missing."""
        block_data = {
            "tags": [[], []]
        }

        with self.assertRaises(BlockParseError) as context:
            validate_block_structure(block_data, self.config)

        self.assertIn("missing required 'text' field", str(context.exception))

    def test_missing_tags_and_tagset(self):
        """Test error when both tags and tagset are missing."""
        block_data = {
            "text": "[Prompt] test [Answer]"
        }

        with self.assertRaises(BlockParseError) as context:
            validate_block_structure(block_data, self.config)

        self.assertIn("must have either 'tags' or 'tagset' field", str(context.exception))

    def test_wrong_text_type(self):
        """Test error when text field is not a string."""
        block_data = {
            "text": 123,
            "tags": [[], []]
        }

        with self.assertRaises(BlockParseError) as context:
            validate_block_structure(block_data, self.config)

        self.assertIn("'text' field must be a string", str(context.exception))

    def test_both_tags_and_tagset(self):
        """Test error when both tags and tagset are present."""
        block_data = {
            "text": "[Prompt] test [Answer]",
            "tags": [[], []],
            "tagset": [[[], []], [[], []]]
        }

        with self.assertRaises(BlockParseError) as context:
            validate_block_structure(block_data, self.config)

        self.assertIn("cannot have both 'tags' and 'tagset' fields", str(context.exception))

    def test_tagset_and_repeats(self):
        """Test error when both tagset and repeats are present."""
        block_data = {
            "text": "[Prompt] test [Answer]",
            "tagset": [[[], []], [[], []]],
            "repeats": 3
        }

        with self.assertRaises(BlockParseError) as context:
            validate_block_structure(block_data, self.config)

        self.assertIn("cannot have both 'tagset' and 'repeats' fields", str(context.exception))

    def test_invalid_repeats_type(self):
        """Test error when repeats is not an integer."""
        block_data = {
            "text": "[Prompt] test [Answer]",
            "tags": [[], []],
            "repeats": "3"
        }

        with self.assertRaises(BlockParseError) as context:
            validate_block_structure(block_data, self.config)

        self.assertIn("'repeats' field must be an integer", str(context.exception))

    def test_repeats_zero(self):
        """Test error when repeats is zero."""
        block_data = {
            "text": "[Prompt] test [Answer]",
            "tags": [[], []],
            "repeats": 0
        }

        with self.assertRaises(BlockParseError) as context:
            validate_block_structure(block_data, self.config)

        self.assertIn("'repeats' field must be greater than 0", str(context.exception))

    def test_large_repeats_warning(self):
        """Test warning for large number of repeats."""
        block_data = {
            "text": "[Prompt] test [Answer]",
            "tags": [[], []],
            "repeats": 150
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_block_structure(block_data, self.config)

            self.assertEqual(len(w), 1)
            self.assertIn("Large number of repetitions", str(w[0].message))
            self.assertIn("without flow control", str(w[0].message))

    def test_invalid_max_gen_tokens(self):
        """Test error when max_gen_tokens is invalid."""
        block_data = {
            "text": "[Prompt] test [Answer]",
            "tags": [[], []],
            "max_gen_tokens": -10
        }

        with self.assertRaises(BlockParseError) as context:
            validate_block_structure(block_data, self.config)

        self.assertIn("'max_gen_tokens' field must be greater than 0", str(context.exception))

    def test_valid_structure(self):
        """Test successful validation of valid block structure."""
        block_data = {
            "text": "[Prompt] test [Answer]",
            "tags": [[], []],
            "repeats": 2,
            "max_gen_tokens": 500
        }

        # Should not raise any exception
        validate_block_structure(block_data, self.config)


class TestTextParsing(unittest.TestCase):
    """Test text parsing into zone structures."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            zone_tokens=["[Prompt]", "[Answer]", "[EOS]"],
            required_tokens=["[Prompt]", "[Answer]"],
            valid_tags=["Training"],
            default_max_token_length=1000,
            sequences=["test"],
            control_token="[Jump]",
            escape_token="[Escape]",
            special_tokens=["[Prompt]", "[Answer]", "[EOS]", "[Jump]", "[Escape]"],
            misc={}
        )

    def test_simple_zone_parsing(self):
        """Test parsing text with all zones present."""
        text = "[Prompt] What is AI? [Answer] AI is artificial intelligence. [EOS]"

        result = parse_text_into_zones(text, self.config)

        self.assertEqual(len(result), 3)  # 3 zones total for 3 zone tokens

        # Check first zone (triggered by [Prompt])
        self.assertEqual(result[0]["advance_token"], "[Prompt]")
        self.assertEqual(result[0]["zone_text"], "[Prompt]")  # Initial zone is empty

        # Check second zone ([Prompt] -> [Answer])
        self.assertEqual(result[1]["advance_token"], "[Answer]")
        self.assertEqual(result[1]["zone_text"], " What is AI? [Answer]")

        # Check third zone ([Answer] -> [EOS])
        self.assertEqual(result[2]["advance_token"], "[EOS]")
        self.assertEqual(result[2]["zone_text"], " AI is artificial intelligence. [EOS]")

    def test_partial_zones_valid(self):
        """Test parsing text with all required tokens but missing optional ones."""
        text = "[Prompt] What is AI? [Answer] AI is intelligence."  # Has required tokens, missing [EOS]

        result = parse_text_into_zones(text, self.config)

        self.assertEqual(len(result), 3)  # Still 3 zones total
        self.assertEqual(result[0]["advance_token"], "[Prompt]")
        self.assertEqual(result[0]["zone_text"], "[Prompt]")
        self.assertEqual(result[1]["advance_token"], "[Answer]")
        self.assertEqual(result[1]["zone_text"], " What is AI? [Answer]")
        self.assertEqual(result[2]["advance_token"], "[EOS]")
        self.assertEqual(result[2]["zone_text"], " AI is intelligence.")  # Gets remaining text

    def test_missing_required_tokens(self):
        """Test error when required tokens are missing."""
        # Missing [Answer] which is required
        text = "[Prompt] What is AI?"

        with self.assertRaises(BlockParseError) as context:
            parse_text_into_zones(text, self.config)

        self.assertIn("Not enough zone tokens", str(context.exception))

    def test_wrong_token_order(self):
        """Test error when tokens are in wrong order."""
        text = "[Answer] AI is artificial intelligence. [Prompt] What is AI?"

        with self.assertRaises(BlockParseError) as context:
            parse_text_into_zones(text, self.config)

        self.assertIn("does not match zone token", str(context.exception))

    def test_too_many_tokens(self):
        """Test error when too many zone tokens are present."""
        # Add extra tokens beyond what config allows
        config_with_few_tokens = Config(
            zone_tokens=["[Prompt]", "[Answer]", "[EOS]"],
            required_tokens=["[Prompt]"],
            valid_tags=["Training"],
            default_max_token_length=1000,
            sequences=["test"],
            control_token="[Jump]",
            escape_token="[Escape]",
            special_tokens=["[Prompt]", "[Answer]", "[EOS]", "[Jump]", "[Escape]"],
            misc={}
        )

        text = "[Prompt] test [Answer] response [EOS] [Answer]"  # [EOS] not in config

        with self.assertRaises(BlockParseError) as context:
            parse_text_into_zones(text, config_with_few_tokens)

        self.assertIn("Too many zone tokens", str(context.exception))

    def test_no_zone_tokens(self):
        """Test error when no zone tokens are present."""
        text = "Just some text without any zone tokens"

        with self.assertRaises(BlockParseError) as context:
            parse_text_into_zones(text, self.config)

        self.assertIn("must contain at least one zone token", str(context.exception))

    def test_empty_zones(self):
        """Test handling of empty zones."""
        text = "[Prompt][Answer][EOS]"

        result = parse_text_into_zones(text, self.config)

        self.assertEqual(len(result), 3)  # 3 zones for 3 tokens
        self.assertEqual(result[0]["zone_text"], "[Prompt]")  # Initial zone
        self.assertEqual(result[1]["zone_text"], "[Answer]")  # Empty between [Prompt] and [Answer]
        self.assertEqual(result[2]["zone_text"], "[EOS]")  # Empty between [Answer] and [EOS]

    def test_escaped_zone_token(self):
        """Test that escaped zone tokens become literal text instead of zone boundaries."""
        text = "[Prompt] This has [Escape] [Answer] in the middle [Answer] Real answer here."

        result = parse_text_into_zones(text, self.config)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["zone_text"], "[Prompt]")
        self.assertEqual(result[1]["zone_text"], " This has [Escape] [Answer] in the middle [Answer]")
        self.assertEqual(result[2]["zone_text"], " Real answer here.")


    def test_escaped_flow_control_token(self):
        """Test that escaped flow control tokens become literal text."""
        text = "[Prompt] Use [Escape] [Jump] to escape, then [Answer] Real answer."

        result = parse_text_into_zones(text, self.config)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["zone_text"], "[Prompt]")
        self.assertEqual(result[1]["zone_text"], " Use [Escape] [Jump] to escape, then [Answer]")
        self.assertEqual(result[2]["zone_text"], " Real answer.")

class TestTagsValidation(unittest.TestCase):
    """Test tags, tagset, and repeats validation and processing."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            zone_tokens=["[Prompt]", "[Answer]", "[EOS]"],
            required_tokens=["[Prompt]"],
            valid_tags=["Training", "Correct", "Incorrect"],
            default_max_token_length=1000,
            sequences=["test"],
            control_token="[Jump]",
            escape_token="[Escape]",
            special_tokens=["[Prompt]", "[Answer]", "[EOS]", "[Jump]", "[Escape]"],
            misc={}
        )

    def test_simple_tags_processing(self):
        """Test processing simple tags field."""
        block_data = {
            "tags": [["Training"], ["Correct"]]
        }

        result = resolve_and_validate_tags(block_data, self.config)

        # Should return one repetition with extra initial zone
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 3)  # [] + 2 user tag lists
        self.assertEqual(result[0], [[], ["Training"], ["Correct"]])

    def test_tags_with_repeats(self):
        """Test processing tags with repeats."""
        block_data = {
            "tags": [["Training"], ["Correct"]],
            "repeats": 3
        }

        result = resolve_and_validate_tags(block_data, self.config)

        # Should return 3 repetitions, each with extra initial zone
        self.assertEqual(len(result), 3)
        for rep in result:
            self.assertEqual(rep, [[], ["Training"], ["Correct"]])

    def test_tagset_processing(self):
        """Test processing tagset field."""
        block_data = {
            "tagset": [
                [["Training"], ["Correct"]],
                [["Training"], ["Incorrect"]]
            ]
        }

        result = resolve_and_validate_tags(block_data, self.config)

        # Should return 2 repetitions with extra initial zone
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [[], ["Training"], ["Correct"]])
        self.assertEqual(result[1], [[], ["Training"], ["Incorrect"]])

    def test_invalid_tag_name(self):
        """Test error for invalid tag names."""
        block_data = {
            "tags": [["InvalidTag"], ["Correct"]]
        }

        with self.assertRaises(BlockParseError) as context:
            resolve_and_validate_tags(block_data, self.config)

        self.assertIn("Invalid tag 'InvalidTag' not in config.valid_tags", str(context.exception))

    def test_wrong_tags_length(self):
        """Test error when tags length doesn't match zones."""
        block_data = {
            "tags": [["Training"]]  # Should be 2 lists for 2 zones
        }

        with self.assertRaises(BlockParseError) as context:
            resolve_and_validate_tags(block_data, self.config)

        self.assertIn("must have 2 sublists, got 1", str(context.exception))

    def test_non_list_tags(self):
        """Test error when tags is not a list."""
        block_data = {
            "tags": "not a list"
        }

        with self.assertRaises(BlockParseError) as context:
            resolve_and_validate_tags(block_data, self.config)

        self.assertIn("'tags' field must be a list", str(context.exception))

    def test_empty_tagset(self):
        """Test error when tagset is empty."""
        block_data = {
            "tagset": []
        }

        with self.assertRaises(BlockParseError) as context:
            resolve_and_validate_tags(block_data, self.config)

        self.assertIn("'tagset' cannot be empty", str(context.exception))

    def test_tagset_wrong_length(self):
        """Test error when tagset repetition has wrong length."""
        block_data = {
            "tagset": [
                [["Training"]]  # Should have 2 sublists for 2 zones
            ]
        }

        with self.assertRaises(BlockParseError) as context:
            resolve_and_validate_tags(block_data, self.config)

        self.assertIn("must have 2 sublists, got 1", str(context.exception))


class TestZoneOrchestration(unittest.TestCase):
    """Test the main block parsing orchestration and zone chaining."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            zone_tokens=["[Prompt]", "[Answer]", "[EOS]"],
            required_tokens=["[Prompt]"],
            valid_tags=["Training", "Correct"],
            default_max_token_length=1000,
            sequences=["test_sequence"],
            control_token="[Jump]",
            escape_token="[Escape]",
            special_tokens=["[Prompt]", "[Answer]", "[EOS]", "[Jump]", "[Escape]"],
            misc={}
        )

        # Create mock zcp nodes
        self.mock_nodes = []
        for i in range(6):  # Enough for testing multiple zones/repetitions
            node = Mock(spec=ZCPNode)
            node.next_zone = None
            self.mock_nodes.append(node)

    def test_single_zone_processing(self):
        """Test processing a simple block with one repetition."""
        block_data = {
            "text": "[Prompt] What is AI? [Answer] AI is intelligence.",
            "tags": [["Training"], ["Correct"]]
        }

        # Mock zone parser to return our test nodes
        mock_zone_parser = Mock(side_effect=self.mock_nodes)

        result = parse_block(
            block_data, self.config, "test_sequence", 0, mock_zone_parser
        )

        # Should return head of chain
        self.assertEqual(result, self.mock_nodes[0])

        # Verify zone parser was called correctly
        self.assertEqual(mock_zone_parser.call_count, 3)  # 2 zones + 1 initial

        # Check ZoneInfo structure for first call (initial zone)
        first_call = mock_zone_parser.call_args_list[0]
        zone_info = first_call[0][0]  # First positional argument

        self.assertEqual(zone_info["advance_token"], "[Prompt]")
        self.assertEqual(zone_info["zone_text"], "")
        self.assertEqual(zone_info["tags"], [])
        self.assertEqual(zone_info["sequence_name"], "test_sequence")
        self.assertEqual(zone_info["block_index"], 0)
        self.assertEqual(zone_info["zone_index"], 0)

        # Verify chaining
        self.assertEqual(self.mock_nodes[0].next_zone, self.mock_nodes[1])
        self.assertEqual(self.mock_nodes[1].next_zone, self.mock_nodes[2])
        self.assertIsNone(self.mock_nodes[2].next_zone)

    def test_multiple_repetitions(self):
        """Test processing block with multiple repetitions."""
        block_data = {
            "text": "[Prompt] Test [Answer]",
            "tags": [["Training"], ["Correct"]],
            "repeats": 2
        }

        mock_zone_parser = Mock(side_effect=self.mock_nodes)

        result = parse_block(
            block_data, self.config, "test_sequence", 0, mock_zone_parser
        )

        # Should call zone parser 6 times (3 zones × 2 repetitions)
        self.assertEqual(mock_zone_parser.call_count, 6)

        # Verify overall zone indexing
        zone_indices = [call[0][0]["zone_index"] for call in mock_zone_parser.call_args_list]
        self.assertEqual(zone_indices, [0, 1, 2, 3, 4, 5])

        # Verify chaining across repetitions
        for i in range(5):
            self.assertEqual(self.mock_nodes[i].next_zone, self.mock_nodes[i + 1])

    def test_zone_parser_wrong_return_type(self):
        """Test error when zone parser returns wrong type."""
        block_data = {
            "text": "[Prompt] Test [Answer]",
            "tags": [[], []]
        }

        # Mock zone parser that returns wrong type
        mock_zone_parser = Mock(return_value="not a ZCPNode")

        with self.assertRaises(BlockParseError) as context:
            parse_block(
                block_data, self.config, "test_sequence", 0, mock_zone_parser
            )

        self.assertIn("Zone parser must return ZCPNode", str(context.exception))
        self.assertIn("test_sequence", str(context.exception))
        self.assertIn("block 0", str(context.exception))

    def test_tags_zones_mismatch(self):
        """Test error when tags don't match number of zones."""
        block_data = {
            "text": "[Prompt] Test [Answer]",
            "tags": [[], [], []]  # 3 tag lists but only 2 zones + 1 initial = 3 total
        }

        mock_zone_parser = Mock(return_value=self.mock_nodes[0])

        # This should actually work since we have 3 zones total (including initial)
        # Let me fix this test
        block_data["tags"] = [[], [], [], []]  # 4 tag lists for 3 zones

        with self.assertRaises(BlockParseError) as context:
            parse_block(
                block_data, self.config, "test_sequence", 0, mock_zone_parser
            )

        self.assertIn("sublists", str(context.exception))

class TestErrorHandling(unittest.TestCase):
    """Test error handling and exception chaining."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            zone_tokens=["[Prompt]", "[Answer]", "[EOS]"],
            required_tokens=["[Prompt]"],
            valid_tags=["Training"],
            default_max_token_length=1000,
            sequences=["test_sequence"],
            control_token="[Jump]",
            escape_token="[Escape]",
            special_tokens=["[Prompt]", "[Answer]", "[EOS]", "[Jump]", "[Escape]"],
            misc={}
        )

    def test_exception_chaining(self):
        """Test that exceptions are properly chained with context."""
        block_data = {
            "text": "[Prompt] Test [Answer]",
            "tags": [[], []]
        }

        # Mock zone parser that raises an exception
        def failing_zone_parser(zone_info, config):
            raise ValueError("Something went wrong in zone processing")

        with self.assertRaises(BlockParseError) as context:
            parse_block(
                block_data, self.config, "test_sequence", 5, failing_zone_parser
            )

        # Check that context is added
        self.assertIn("Error parsing block 5 in sequence 'test_sequence'", str(context.exception))
        self.assertIn("Something went wrong in zone processing", str(context.exception))

        # Check that exception is chained
        self.assertIsInstance(context.exception.__cause__, ValueError)

    def test_error_context_in_messages(self):
        """Test that error messages include helpful context."""
        block_data = {
            "text": "[Prompt] Test [Answer]",
            "tags": [[], []]
        }

        # Test with zone parser returning wrong type
        mock_zone_parser = Mock(return_value="wrong type")

        with self.assertRaises(BlockParseError) as context:
            parse_block(
                block_data, self.config, "my_sequence", 3, mock_zone_parser
            )

        error_msg = str(context.exception)
        self.assertIn("my_sequence", error_msg)
        self.assertIn("block 3", error_msg)
        self.assertIn("zone 0", error_msg)
        self.assertIn("repetition 0", error_msg)


if __name__ == "__main__":
    unittest.main()