"""
Unit tests for UDPL sequence parsing functionality.

Tests focus on:
1. Sequence parsing validation and error handling
2. Chain construction and linked list attachment verification
3. Integration with block parser (updated for new signature)
"""

import unittest
from unittest.mock import Mock
from typing import Dict, Any

# Import the modules under test (adjust imports based on actual project structure)
from src.workflow_forge.parsing.sequence_parsing import parse_sequences, SequenceParseError
from src.workflow_forge.ZCP.nodes import ZCPNode
from src.workflow_forge.parsing.config_parsing import Config


class TestSequenceParser(unittest.TestCase):
    """Test cases for sequence parsing functionality."""

    def setUp(self):
        """Set up common test fixtures."""
        # Create a basic valid config for testing
        self.config = Config(
            zone_tokens=["[Prompt]", "[Answer]", "[EOS]"],
            required_tokens=["[Prompt]", "[Answer]"],
            valid_tags=["Training", "Correct"],
            default_max_token_length=1000,
            sequences=["setup", "main", "conclude"],
            control_token="[Jump]",
            escape_token="[Escape]",
            special_tokens=["[Prompt]", "[Answer]", "[EOS]", "[Jump]", "[Escape]"],
            misc={}
        )

        # Create mock ZCP nodes for testing chain construction
        self.mock_node1 = Mock(spec=ZCPNode)
        self.mock_node1.next_zone = None
        self.mock_node1.get_last_node.return_value = self.mock_node1

        self.mock_node2 = Mock(spec=ZCPNode)
        self.mock_node2.next_zone = None
        self.mock_node2.get_last_node.return_value = self.mock_node2

        self.mock_node3 = Mock(spec=ZCPNode)
        self.mock_node3.next_zone = None
        self.mock_node3.get_last_node.return_value = self.mock_node3

    def test_missing_sequence_in_toml(self):
        """Test error when sequence declared in config but missing from TOML."""
        toml_data = {
            "setup": [{"text": "test"}],
            "main": [{"text": "test"}]
            # Missing "conclude" sequence
        }

        mock_block_parser = Mock(return_value=self.mock_node1)

        with self.assertRaises(SequenceParseError) as context:
            parse_sequences(toml_data, self.config, mock_block_parser)

        self.assertIn("Sequence 'conclude' declared in config but not found", str(context.exception))

    def test_sequence_not_a_list(self):
        """Test error when sequence doesn't resolve to a list."""
        toml_data = {
            "setup": [{"text": "test"}],
            "main": "not_a_list",  # Should be a list
            "conclude": [{"text": "test"}]
        }

        mock_block_parser = Mock(return_value=self.mock_node1)

        with self.assertRaises(SequenceParseError) as context:
            parse_sequences(toml_data, self.config, mock_block_parser)

        self.assertIn("Sequence 'main' must resolve to a list", str(context.exception))

    def test_empty_sequence(self):
        """Test error when sequence is an empty list."""
        toml_data = {
            "setup": [{"text": "test"}],
            "main": [],  # Empty sequence
            "conclude": [{"text": "test"}]
        }

        mock_block_parser = Mock(return_value=self.mock_node1)

        with self.assertRaises(SequenceParseError) as context:
            parse_sequences(toml_data, self.config, mock_block_parser)

        self.assertIn("Sequence 'main' cannot be empty", str(context.exception))

    def test_block_not_dictionary(self):
        """Test error when block is not a dictionary."""
        toml_data = {
            "setup": [{"text": "test"}],
            "main": [{"text": "test"}, "not_a_dict"],  # Second block is not a dict
            "conclude": [{"text": "test"}]
        }

        mock_block_parser = Mock(return_value=self.mock_node1)

        with self.assertRaises(SequenceParseError) as context:
            parse_sequences(toml_data, self.config, mock_block_parser)

        self.assertIn("Block 1 in sequence 'main' must be a dictionary", str(context.exception))

    def test_block_parser_returns_non_zcp_node(self):
        """Test error when block parser doesn't return a ZCPNode."""
        toml_data = {
            "setup": [{"text": "test"}],
            "main": [{"text": "test"}],
            "conclude": [{"text": "test"}]
        }

        # Mock block parser that returns wrong type
        mock_block_parser = Mock(return_value="not_a_zcp_node")

        with self.assertRaises(SequenceParseError) as context:
            parse_sequences(toml_data, self.config, mock_block_parser)

        self.assertIn("Block parser must return a ZCPNode", str(context.exception))

    def test_single_block_sequence_success(self):
        """Test successful parsing of sequences with single blocks."""
        toml_data = {
            "setup": [{"text": "setup block"}],
            "main": [{"text": "main block"}],
            "conclude": [{"text": "conclude block"}]
        }

        # Mock block parser returns different nodes for each call
        mock_block_parser = Mock(side_effect=[self.mock_node1, self.mock_node2, self.mock_node3])

        result = parse_sequences(toml_data, self.config, mock_block_parser)

        # Verify all sequences are present
        self.assertEqual(set(result.keys()), {"setup", "main", "conclude"})

        # Verify each sequence points to the correct head node
        self.assertEqual(result["setup"], self.mock_node1)
        self.assertEqual(result["main"], self.mock_node2)
        self.assertEqual(result["conclude"], self.mock_node3)

        # Verify block parser was called correctly with sequence name and block index
        self.assertEqual(mock_block_parser.call_count, 3)
        mock_block_parser.assert_any_call({"text": "setup block"}, self.config, "setup", 0)
        mock_block_parser.assert_any_call({"text": "main block"}, self.config, "main", 0)
        mock_block_parser.assert_any_call({"text": "conclude block"}, self.config, "conclude", 0)

    def test_multi_block_sequence_linking(self):
        """Test that multi-block sequences are properly linked into chains."""
        toml_data = {
            "setup": [{"text": "block1"}, {"text": "block2"}, {"text": "block3"}],
            "main": [{"text": "single block"}],
            "conclude": [{"text": "final block"}]
        }

        # Create additional mock nodes for the multi-block sequence
        mock_node1_tail = Mock(spec=ZCPNode)
        mock_node1_tail.next_zone = None

        mock_node2_tail = Mock(spec=ZCPNode)
        mock_node2_tail.next_zone = None

        mock_node3_tail = Mock(spec=ZCPNode)
        mock_node3_tail.next_zone = None

        # Set up get_last_node to return the tail nodes
        self.mock_node1.get_last_node.return_value = mock_node1_tail
        self.mock_node2.get_last_node.return_value = mock_node2_tail
        self.mock_node3.get_last_node.return_value = mock_node3_tail

        # Mock block parser returns nodes in sequence
        mock_block_parser = Mock(side_effect=[
            self.mock_node1,  # setup block 1
            self.mock_node2,  # setup block 2
            self.mock_node3,  # setup block 3
            Mock(spec=ZCPNode),  # main block
            Mock(spec=ZCPNode)  # conclude block
        ])

        result = parse_sequences(toml_data, self.config, mock_block_parser)

        # Verify the setup sequence starts with the first node
        self.assertEqual(result["setup"], self.mock_node1)

        # Verify the chain linking in setup sequence
        # Block 1 tail should point to block 2 head
        self.assertEqual(mock_node1_tail.next_zone, self.mock_node2)
        # Block 2 tail should point to block 3 head
        self.assertEqual(mock_node2_tail.next_zone, self.mock_node3)
        # Block 3 tail should remain None (end of chain)
        self.assertIsNone(mock_node3_tail.next_zone)

        # Verify get_last_node was called correctly during chaining
        self.mock_node1.get_last_node.assert_called()
        self.mock_node2.get_last_node.assert_called()
        self.mock_node3.get_last_node.assert_called()

        # Verify block parser was called with correct parameters
        expected_calls = [
            ({"text": "block1"}, self.config, "setup", 0),
            ({"text": "block2"}, self.config, "setup", 1),
            ({"text": "block3"}, self.config, "setup", 2),
            ({"text": "single block"}, self.config, "main", 0),
            ({"text": "final block"}, self.config, "conclude", 0)
        ]
        for i, expected_call in enumerate(expected_calls):
            actual_call = mock_block_parser.call_args_list[i]
            self.assertEqual(actual_call[0], expected_call)

    def test_block_parser_exception_propagation(self):
        """Test that exceptions from block parser are properly wrapped."""
        toml_data = {
            "setup": [{"text": "good block"}],
            "main": [{"text": "bad block"}],  # This will cause block parser to fail
            "conclude": [{"text": "good block"}]
        }

        def failing_block_parser(block_data, config, sequence_name, block_index):
            if block_data["text"] == "bad block":
                raise ValueError("Block parsing failed")
            return self.mock_node1

        with self.assertRaises(SequenceParseError) as context:
            parse_sequences(toml_data, self.config, failing_block_parser)

        # Should include context about which block and sequence failed
        self.assertIn("Error parsing block 0 in sequence 'main'", str(context.exception))
        self.assertIn("Block parsing failed", str(context.exception))

    def test_complex_chain_construction(self):
        """Test construction of complex chains with multiple blocks per sequence."""
        toml_data = {
            "setup": [{"text": "s1"}, {"text": "s2"}],
            "main": [{"text": "m1"}, {"text": "m2"}, {"text": "m3"}],
            "conclude": [{"text": "c1"}]
        }

        # Create a more complex set of mock nodes to track linking
        nodes = []
        for i in range(6):  # Total blocks across all sequences
            node = Mock(spec=ZCPNode)
            node.next_zone = None
            node.get_last_node.return_value = node
            nodes.append(node)

        mock_block_parser = Mock(side_effect=nodes)

        result = parse_sequences(toml_data, self.config, mock_block_parser)

        # Verify sequence heads
        self.assertEqual(result["setup"], nodes[0])  # s1
        self.assertEqual(result["main"], nodes[2])  # m1
        self.assertEqual(result["conclude"], nodes[5])  # c1

        # Verify setup chain: s1 -> s2
        self.assertEqual(nodes[0].next_zone, nodes[1])
        self.assertIsNone(nodes[1].next_zone)

        # Verify main chain: m1 -> m2 -> m3
        self.assertEqual(nodes[2].next_zone, nodes[3])
        self.assertEqual(nodes[3].next_zone, nodes[4])
        self.assertIsNone(nodes[4].next_zone)

        # Verify conclude chain: c1 (single node)
        self.assertIsNone(nodes[5].next_zone)

        # Verify block parser was called with correct parameters
        expected_calls = [
            ({"text": "s1"}, self.config, "setup", 0),
            ({"text": "s2"}, self.config, "setup", 1),
            ({"text": "m1"}, self.config, "main", 0),
            ({"text": "m2"}, self.config, "main", 1),
            ({"text": "m3"}, self.config, "main", 2),
            ({"text": "c1"}, self.config, "conclude", 0)
        ]
        for i, expected_call in enumerate(expected_calls):
            actual_call = mock_block_parser.call_args_list[i]
            self.assertEqual(actual_call[0], expected_call)

    def test_all_sequences_processed(self):
        """Test that all sequences in config are processed, none skipped."""
        toml_data = {
            "setup": [{"text": "setup"}],
            "main": [{"text": "main"}],
            "conclude": [{"text": "conclude"}],
            "extra": [{"text": "extra"}]  # Extra sequence not in config should be ignored
        }

        mock_block_parser = Mock(return_value=self.mock_node1)

        result = parse_sequences(toml_data, self.config, mock_block_parser)

        # Should only contain sequences from config
        self.assertEqual(set(result.keys()), {"setup", "main", "conclude"})
        self.assertNotIn("extra", result)

        # Should have called block parser exactly 3 times (once per configured sequence)
        self.assertEqual(mock_block_parser.call_count, 3)

        # Verify correct parameters were passed
        expected_calls = [
            ({"text": "setup"}, self.config, "setup", 0),
            ({"text": "main"}, self.config, "main", 0),
            ({"text": "conclude"}, self.config, "conclude", 0)
        ]
        for i, expected_call in enumerate(expected_calls):
            actual_call = mock_block_parser.call_args_list[i]
            self.assertEqual(actual_call[0], expected_call)


if __name__ == "__main__":
    unittest.main()