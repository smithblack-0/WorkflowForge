"""
Unit tests for UDPL sequence parsing functionality.

Tests focus on:
1. Sequence parsing validation and error handling
2. Chain construction and linked list attachment verification
3. Integration with block parser (updated for new signature)
"""

import unittest
from unittest.mock import Mock
from typing import Dict, Any, Optional, List

# Import the modules under test (adjust imports based on actual project structure)
from src.workflow_forge.parsing.sequence_parsing import parse_sequences, SequenceParseError
from src.workflow_forge.zcp.nodes import ZCPNode
from src.workflow_forge.parsing.config_parsing import Config


class BaseSequenceParserTest(unittest.TestCase):
    """Base test class with common setup and helper methods."""

    def setUp(self):
        """Set up common test fixtures."""
        pass  # No complex mocks needed for sequence parser

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
            'sequences': ["setup", "main", "conclude"],
            'control_pattern': "[Jump]",
            'escape_patterns': ("[Escape]", "[EndEscape]"),
            'tools': ["search", "calculator"],
            'misc': {}
        }
        base_config.update(overrides)
        return Config(**base_config)

    def create_mock_node(self, node_id: Optional[str] = None) -> Mock:
        """
        Create a mock ZCPNode with proper spec and default behavior.

        Args:
            node_id: Optional identifier for debugging

        Returns:
            Mock ZCPNode with proper configuration
        """
        mock_node = Mock(spec=ZCPNode)
        mock_node.next_zone = None
        mock_node.get_last_node.return_value = mock_node
        if node_id:
            mock_node._test_id = node_id  # For debugging
        return mock_node

    def create_mock_nodes(self, count: int) -> List[Mock]:
        """
        Create multiple mock nodes for complex chain testing.

        Args:
            count: Number of nodes to create

        Returns:
            List of mock ZCPNodes
        """
        return [self.create_mock_node(f"node_{i}") for i in range(count)]

    def create_mock_block_parser(self, return_values: List[Mock]) -> Mock:
        """
        Create a mock block parser that returns specified nodes in sequence.

        Args:
            return_values: List of mock nodes to return in order

        Returns:
            Mock block parser function
        """
        mock_parser = Mock(side_effect=return_values)
        return mock_parser

    def create_failing_block_parser(self, fail_on_text: str, error_msg: str = "Block parsing failed") -> Mock:
        """
        Create a mock block parser that fails on specific text.

        Args:
            fail_on_text: Text that triggers failure
            error_msg: Error message to raise

        Returns:
            Mock block parser that fails conditionally
        """
        def failing_parser(block_data, config, sequence_name, block_index):
            if block_data.get("text") == fail_on_text:
                raise ValueError(error_msg)
            return self.create_mock_node()

        return Mock(side_effect=failing_parser)

    def create_toml_data(self, **sequences) -> Dict[str, Any]:
        """
        Create TOML data with specified sequences.

        Args:
            **sequences: Mapping of sequence names to their block lists

        Returns:
            Dictionary representing TOML data
        """
        return dict(sequences)



    def assert_chain_linking(self, head_node: Mock, expected_chain: List[Mock]):
        """
        Assert that nodes are properly linked in a chain.

        Args:
            head_node: First node in the chain
            expected_chain: Expected sequence of nodes
        """
        current = head_node
        for i, expected_node in enumerate(expected_chain):
            self.assertEqual(current, expected_node, f"Chain mismatch at position {i}")
            if i < len(expected_chain) - 1:
                # Not the last node, should have next_zone
                current = current.next_zone
                self.assertIsNotNone(current, f"Chain broken at position {i}")
            else:
                # Last node, should have no next_zone
                self.assertIsNone(current.next_zone, "Chain should end with None")

    def assert_block_parser_calls(self, mock_parser: Mock, expected_calls: List[tuple]):
        """
        Assert that block parser was called with expected parameters.

        Args:
            mock_parser: Mock block parser
            expected_calls: List of (block_data, config, sequence_name, block_index) tuples
        """
        self.assertEqual(mock_parser.call_count, len(expected_calls))
        for i, expected_call in enumerate(expected_calls):
            actual_call = mock_parser.call_args_list[i]
            self.assertEqual(actual_call[0], expected_call)


class TestSequenceParserValidation(BaseSequenceParserTest):
    """Test sequence parsing validation and error handling."""

    def test_missing_sequence_in_toml(self):
        """Test error when sequence declared in config but missing from TOML."""
        config = self.get_valid_config()
        toml_data = self.create_toml_data(
            setup=[{"text": "test"}],
            main=[{"text": "test"}]
            # Missing "conclude" sequence
        )

        # Need enough nodes for: setup(1) + main(1) before it discovers conclude is missing
        mock_block_parser = self.create_mock_block_parser([self.create_mock_node(), self.create_mock_node()])

        with self.assertRaises(SequenceParseError) as context:
            parse_sequences(toml_data, config, mock_block_parser)

        error_msg = str(context.exception).lower()
        self.assertIn("sequence", error_msg)
        self.assertIn("conclude", error_msg)
        self.assertIn("declared", error_msg)
        self.assertIn("not found", error_msg)

    def test_sequence_not_a_list(self):
        """Test error when sequence doesn't resolve to a list."""
        config = self.get_valid_config()
        toml_data = self.create_toml_data(
            setup=[{"text": "test"}],
            main="not_a_list",  # Should be a list
            conclude=[{"text": "test"}]
        )

        mock_block_parser = self.create_mock_block_parser([self.create_mock_node()])

        with self.assertRaises(SequenceParseError) as context:
            parse_sequences(toml_data, config, mock_block_parser)

        error_msg = str(context.exception).lower()
        self.assertIn("sequence", error_msg)
        self.assertIn("main", error_msg)
        self.assertIn("list", error_msg)

    def test_empty_sequence(self):
        """Test error when sequence is an empty list."""
        config = self.get_valid_config()
        toml_data = self.create_toml_data(
            setup=[{"text": "test"}],
            main=[],  # Empty sequence
            conclude=[{"text": "test"}]
        )

        mock_block_parser = self.create_mock_block_parser([self.create_mock_node()])

        with self.assertRaises(SequenceParseError) as context:
            parse_sequences(toml_data, config, mock_block_parser)

        error_msg = str(context.exception).lower()
        self.assertIn("sequence", error_msg)
        self.assertIn("main", error_msg)
        self.assertIn("empty", error_msg)

    def test_block_not_dictionary(self):
        """Test error when block is not a dictionary."""
        config = self.get_valid_config()
        toml_data = self.create_toml_data(
            setup=[{"text": "test"}],
            main=[{"text": "test"}, "not_a_dict"],  # Second block is not a dict
            conclude=[{"text": "test"}]
        )

        # Need enough nodes for: setup(1) + main(1 valid block before error)
        mock_block_parser = self.create_mock_block_parser([self.create_mock_node(), self.create_mock_node()])

        with self.assertRaises(SequenceParseError) as context:
            parse_sequences(toml_data, config, mock_block_parser)

        error_msg = str(context.exception).lower()
        self.assertIn("block", error_msg)
        self.assertIn("1", error_msg)
        self.assertIn("main", error_msg)
        self.assertIn("dictionary", error_msg)

    def test_block_parser_returns_non_zcp_node(self):
        """Test error when block parser doesn't return a ZCPNode."""
        config = self.get_valid_config()
        toml_data = self.create_toml_data(
            setup=[{"text": "test"}],
            main=[{"text": "test"}],
            conclude=[{"text": "test"}]
        )

        # Mock block parser that returns wrong type
        mock_block_parser = Mock(return_value="not_a_zcp_node")

        with self.assertRaises(SequenceParseError) as context:
            parse_sequences(toml_data, config, mock_block_parser)

        error_msg = str(context.exception).lower()
        self.assertIn("block parser", error_msg)
        self.assertIn("zcpnode", error_msg)

    def test_block_parser_exception_propagation(self):
        """Test that exceptions from block parser are properly wrapped."""
        config = self.get_valid_config()
        toml_data = self.create_toml_data(
            setup=[{"text": "good block"}],
            main=[{"text": "bad block"}],  # This will cause block parser to fail
            conclude=[{"text": "good block"}]
        )

        failing_parser = self.create_failing_block_parser("bad block", "Block parsing failed")

        with self.assertRaises(SequenceParseError) as context:
            parse_sequences(toml_data, config, failing_parser)

        # Should include context about which block and sequence failed
        error_msg = str(context.exception).lower()
        self.assertIn("error", error_msg)
        self.assertIn("block", error_msg)
        self.assertIn("0", error_msg)
        self.assertIn("main", error_msg)
        self.assertIn("block parsing failed", error_msg)


class TestSequenceParserChainConstruction(BaseSequenceParserTest):
    """Test chain construction and linked list attachment verification."""

    def test_single_block_sequence_success(self):
        """Test successful parsing of sequences with single blocks."""
        config = self.get_valid_config()
        toml_data = self.create_toml_data(
            setup=[{"text": "setup block"}],
            main=[{"text": "main block"}],
            conclude=[{"text": "conclude block"}]
        )

        # Create mock nodes for each sequence
        nodes = self.create_mock_nodes(3)
        mock_block_parser = self.create_mock_block_parser(nodes)

        result = parse_sequences(toml_data, config, mock_block_parser)

        # Verify all sequences are present
        self.assertEqual(set(result.keys()), {"setup", "main", "conclude"})

        # Verify each sequence points to the correct head node
        self.assertEqual(result["setup"], nodes[0])
        self.assertEqual(result["main"], nodes[1])
        self.assertEqual(result["conclude"], nodes[2])

        # Verify block parser was called correctly
        expected_calls = [
            ({"text": "setup block"}, config, "setup", 0),
            ({"text": "main block"}, config, "main", 0),
            ({"text": "conclude block"}, config, "conclude", 0)
        ]
        self.assert_block_parser_calls(mock_block_parser, expected_calls)

    def test_multi_block_sequence_linking(self):
        """Test that multi-block sequences are properly linked into chains."""
        config = self.get_valid_config()
        toml_data = self.create_toml_data(
            setup=[{"text": "block1"}, {"text": "block2"}, {"text": "block3"}],
            main=[{"text": "single block"}],
            conclude=[{"text": "final block"}]
        )

        # Create nodes and set up proper tail behavior
        nodes = self.create_mock_nodes(5)

        # Set up get_last_node behavior for chain construction
        nodes[0].get_last_node.return_value = nodes[0]  # block1 tail is itself
        nodes[1].get_last_node.return_value = nodes[1]  # block2 tail is itself
        nodes[2].get_last_node.return_value = nodes[2]  # block3 tail is itself

        mock_block_parser = self.create_mock_block_parser(nodes)

        result = parse_sequences(toml_data, config, mock_block_parser)

        # Verify the setup sequence starts with the first node
        self.assertEqual(result["setup"], nodes[0])

        # Verify the chain linking in setup sequence
        self.assertEqual(nodes[0].next_zone, nodes[1])  # block1 -> block2
        self.assertEqual(nodes[1].next_zone, nodes[2])  # block2 -> block3
        self.assertIsNone(nodes[2].next_zone)          # block3 -> None

        # Verify get_last_node was called correctly during chaining
        nodes[0].get_last_node.assert_called()
        nodes[1].get_last_node.assert_called()

        # Verify block parser was called with correct parameters
        expected_calls = [
            ({"text": "block1"}, config, "setup", 0),
            ({"text": "block2"}, config, "setup", 1),
            ({"text": "block3"}, config, "setup", 2),
            ({"text": "single block"}, config, "main", 0),
            ({"text": "final block"}, config, "conclude", 0)
        ]
        self.assert_block_parser_calls(mock_block_parser, expected_calls)

    def test_complex_chain_construction(self):
        """Test construction of complex chains with multiple blocks per sequence."""
        config = self.get_valid_config()
        toml_data = self.create_toml_data(
            setup=[{"text": "s1"}, {"text": "s2"}],
            main=[{"text": "m1"}, {"text": "m2"}, {"text": "m3"}],
            conclude=[{"text": "c1"}]
        )

        # Create nodes for all blocks
        nodes = self.create_mock_nodes(6)
        mock_block_parser = self.create_mock_block_parser(nodes)

        result = parse_sequences(toml_data, config, mock_block_parser)

        # Verify sequence heads
        self.assertEqual(result["setup"], nodes[0])    # s1
        self.assertEqual(result["main"], nodes[2])     # m1
        self.assertEqual(result["conclude"], nodes[5]) # c1

        # Verify setup chain: s1 -> s2
        self.assert_chain_linking(result["setup"], [nodes[0], nodes[1]])

        # Verify main chain: m1 -> m2 -> m3
        self.assert_chain_linking(result["main"], [nodes[2], nodes[3], nodes[4]])

        # Verify conclude chain: c1 (single node)
        self.assert_chain_linking(result["conclude"], [nodes[5]])

        # Verify block parser was called with correct parameters
        expected_calls = [
            ({"text": "s1"}, config, "setup", 0),
            ({"text": "s2"}, config, "setup", 1),
            ({"text": "m1"}, config, "main", 0),
            ({"text": "m2"}, config, "main", 1),
            ({"text": "m3"}, config, "main", 2),
            ({"text": "c1"}, config, "conclude", 0)
        ]
        self.assert_block_parser_calls(mock_block_parser, expected_calls)


class TestSequenceParserIntegration(BaseSequenceParserTest):
    """Test integration with block parser and overall behavior."""

    def test_all_sequences_processed(self):
        """Test that all sequences in config are processed, none skipped."""
        config = self.get_valid_config()
        toml_data = self.create_toml_data(
            setup=[{"text": "setup"}],
            main=[{"text": "main"}],
            conclude=[{"text": "conclude"}],
            extra=[{"text": "extra"}]  # Extra sequence not in config should be ignored
        )

        nodes = self.create_mock_nodes(3)
        mock_block_parser = self.create_mock_block_parser(nodes)

        result = parse_sequences(toml_data, config, mock_block_parser)

        # Should only contain sequences from config
        self.assertEqual(set(result.keys()), {"setup", "main", "conclude"})
        self.assertNotIn("extra", result)

        # Should have called block parser exactly 3 times (once per configured sequence)
        self.assertEqual(mock_block_parser.call_count, 3)

        # Verify correct parameters were passed
        expected_calls = [
            ({"text": "setup"}, config, "setup", 0),
            ({"text": "main"}, config, "main", 0),
            ({"text": "conclude"}, config, "conclude", 0)
        ]
        self.assert_block_parser_calls(mock_block_parser, expected_calls)

    def test_different_config_sequences(self):
        """Test with different sequence configurations."""
        config = self.get_valid_config(sequences=["alpha", "beta"])
        toml_data = self.create_toml_data(
            alpha=[{"text": "first"}],
            beta=[{"text": "second"}, {"text": "third"}],
            gamma=[{"text": "ignored"}]  # Not in config
        )

        nodes = self.create_mock_nodes(3)
        mock_block_parser = self.create_mock_block_parser(nodes)

        result = parse_sequences(toml_data, config, mock_block_parser)

        # Should only process alpha and beta
        self.assertEqual(set(result.keys()), {"alpha", "beta"})
        self.assertEqual(result["alpha"], nodes[0])
        self.assertEqual(result["beta"], nodes[1])

        # Verify beta chain linking
        self.assertEqual(nodes[1].next_zone, nodes[2])
        self.assertIsNone(nodes[2].next_zone)

        expected_calls = [
            ({"text": "first"}, config, "alpha", 0),
            ({"text": "second"}, config, "beta", 0),
            ({"text": "third"}, config, "beta", 1)
        ]
        self.assert_block_parser_calls(mock_block_parser, expected_calls)

    def test_block_parser_gets_correct_context(self):
        """Test that block parser receives correct context information."""
        config = self.get_valid_config(sequences=["test_seq"])
        toml_data = self.create_toml_data(
            test_seq=[
                {"text": "block_0", "extra": "data"},
                {"text": "block_1", "more": "info"}
            ]
        )

        # Use a mock that captures all call arguments
        captured_calls = []
        def capturing_parser(block_data, config_obj, sequence_name, block_index):
            captured_calls.append((block_data, config_obj, sequence_name, block_index))
            return self.create_mock_node()

        result = parse_sequences(toml_data, config, capturing_parser)

        # Verify context was passed correctly
        self.assertEqual(len(captured_calls), 2)

        # First block call
        block_data, config_obj, seq_name, block_idx = captured_calls[0]
        self.assertEqual(block_data, {"text": "block_0", "extra": "data"})
        self.assertEqual(config_obj, config)
        self.assertEqual(seq_name, "test_seq")
        self.assertEqual(block_idx, 0)

        # Second block call
        block_data, config_obj, seq_name, block_idx = captured_calls[1]
        self.assertEqual(block_data, {"text": "block_1", "more": "info"})
        self.assertEqual(config_obj, config)
        self.assertEqual(seq_name, "test_seq")
        self.assertEqual(block_idx, 1)


if __name__ == "__main__":
    unittest.main()