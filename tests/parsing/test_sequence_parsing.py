import unittest
from unittest.mock import MagicMock
from typing import Dict, Any

from src.workflow_forge.parsing.sequence_parsing import (
    parse_sequences,
    SequenceParseError
)
from src.workflow_forge.ZCP.nodes import ZCPNode
from src.workflow_forge.parsing.config_parsing import Config


class BaseSequenceTest(unittest.TestCase):
    """Base test class with helper methods."""

    def get_valid_config(self) -> Config:
        """Return a valid config for testing."""
        return Config(
            zone_tokens=["[Prompt]", "[Answer]", "[EOS]"],
            required_tokens=["[Prompt]", "[Answer]"],
            valid_tags=["Training", "Correct"],
            default_max_token_length=1000,
            sequences=["setup", "loop"],
            control_token="[Jump]",
            escape_token="[Escape]",
            special_tokens=["[Prompt]", "[Answer]", "[EOS]", "[Jump]", "[Escape]"],
            misc={}
        )

    def create_mock_zcp_node(self, sequence: str = "test", block: int = 0) -> ZCPNode:
        """Create a mock ZCP node for testing."""
        return ZCPNode(
            sequence=sequence,
            block=block,
            sampling_callbacks=lambda resources: {},
            zone_advance_token="[Answer]",
            tags=["Training"],
            timeout=100,
            input=True,
            output=False
        )

    def create_mock_block_parser(self, return_node: ZCPNode = None):
        """Create a mock block parser that returns a single ZCP node."""
        if return_node is None:
            return_node = self.create_mock_zcp_node()

        mock_parser = MagicMock()
        mock_parser.return_value = return_node
        return mock_parser


class TestSequenceExistenceValidation(BaseSequenceTest):
    """Test that sequences declared in config exist in TOML."""

    def test_missing_sequence_in_toml(self):
        """Test error when sequence declared in config but missing from TOML."""
        config = self.get_valid_config()
        toml_data = {
            'setup': [{'text': 'test'}],
            # 'loop' is missing
        }
        block_parser = self.create_mock_block_parser()

        with self.assertRaisesRegex(
                SequenceParseError,
                "Sequence 'loop' declared in config but not found in TOML data"
        ):
            parse_sequences(toml_data, config, block_parser)


class TestSequenceListValidation(BaseSequenceTest):
    """Test that sequences resolve to lists."""

    def test_sequence_not_list(self):
        """Test error when sequence is not a list."""
        config = self.get_valid_config()
        toml_data = {
            'setup': [{'text': 'test'}],
            'loop': {'text': 'not a list'}  # Should be a list
        }
        block_parser = self.create_mock_block_parser()

        with self.assertRaisesRegex(
                SequenceParseError,
                "Sequence 'loop' must resolve to a list, got dict"
        ):
            parse_sequences(toml_data, config, block_parser)

    def test_sequence_empty_list(self):
        """Test error when sequence is an empty list."""
        config = self.get_valid_config()
        toml_data = {
            'setup': [{'text': 'test'}],
            'loop': []  # Empty list
        }
        block_parser = self.create_mock_block_parser()

        with self.assertRaisesRegex(
                SequenceParseError,
                "Sequence 'loop' cannot be empty"
        ):
            parse_sequences(toml_data, config, block_parser)


class TestBlockDataValidation(BaseSequenceTest):
    """Test validation of individual block data."""

    def test_block_not_dict(self):
        """Test error when block is not a dictionary."""
        config = self.get_valid_config()
        toml_data = {
            'setup': [{'text': 'test'}, 'not a dict'],  # Second block invalid
            'loop': [{'text': 'test'}]
        }
        block_parser = self.create_mock_block_parser()

        with self.assertRaisesRegex(
                SequenceParseError,
                "Block 1 in sequence 'setup' must be a dictionary, got str"
        ):
            parse_sequences(toml_data, config, block_parser)


class TestBlockParserIntegration(BaseSequenceTest):
    """Test integration with the block parser."""

    def test_block_parser_not_returning_zcp_node(self):
        """Test error when block parser doesn't return a ZCP node."""
        config = self.get_valid_config()
        toml_data = {
            'setup': [{'text': 'test'}],
            'loop': [{'text': 'test'}]
        }

        # Mock block parser that returns a string instead of ZCP node
        mock_parser = MagicMock()
        mock_parser.return_value = "not a zcp node"

        with self.assertRaisesRegex(
                SequenceParseError,
                "Block parser must return a ZCPNode, got str"
        ):
            parse_sequences(toml_data, config, mock_parser)

    def test_block_parser_called_with_correct_args(self):
        """Test that block parser is called with correct arguments."""
        config = self.get_valid_config()
        block_data = {'text': 'test content'}
        toml_data = {
            'setup': [block_data],
            'loop': [{'text': 'loop content'}]
        }

        mock_parser = self.create_mock_block_parser()

        parse_sequences(toml_data, config, mock_parser)

        # Check that block parser was called with correct args
        self.assertEqual(mock_parser.call_count, 2)

        # First call should be for setup block
        first_call_args = mock_parser.call_args_list[0]
        self.assertEqual(first_call_args[0][0], block_data)
        self.assertEqual(first_call_args[0][1], config)


class TestZCPChainConstruction(BaseSequenceTest):
    """Test construction of ZCP node chains."""

    def test_single_block_single_node(self):
        """Test chain construction with one block containing one node."""
        config = self.get_valid_config()
        toml_data = {
            'setup': [{'text': 'test'}],
            'loop': [{'text': 'test'}]
        }

        # Create separate nodes for each call
        setup_node = self.create_mock_zcp_node(sequence="setup")
        loop_node = self.create_mock_zcp_node(sequence="loop")

        call_count = 0

        def mock_parser_func(block_data, config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return setup_node
            else:
                return loop_node

        mock_parser = MagicMock(side_effect=mock_parser_func)

        result = parse_sequences(toml_data, config, mock_parser)

        # Should have chains for both sequences
        self.assertEqual(len(result), 2)
        self.assertIn('setup', result)
        self.assertIn('loop', result)

        # Each chain should point to the respective nodes
        self.assertEqual(result['setup'], setup_node)
        self.assertEqual(result['loop'], loop_node)

    def test_single_block_with_internal_chain(self):
        """Test when block parser returns a node that's already part of a chain."""
        config = self.get_valid_config()
        toml_data = {
            'setup': [{'text': 'test'}],
            'loop': [{'text': 'test'}]
        }

        # Create a chain within a single block
        node1 = self.create_mock_zcp_node(sequence="setup", block=0)
        node2 = self.create_mock_zcp_node(sequence="setup", block=0)
        node3 = self.create_mock_zcp_node(sequence="setup", block=0)

        # Link them together
        node1.next_zone = node2
        node2.next_zone = node3

        # Block parser returns the head of this internal chain
        setup_parser = self.create_mock_block_parser(node1)
        loop_node = self.create_mock_zcp_node(sequence="loop")

        call_count = 0

        def mock_parser_func(block_data, config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return node1  # Return head of chain
            else:
                return loop_node

        mock_parser = MagicMock(side_effect=mock_parser_func)

        result = parse_sequences(toml_data, config, mock_parser)

        # Chain head should be first node
        self.assertEqual(result['setup'], node1)

        # Should be able to traverse the internal chain
        self.assertEqual(result['setup'].next_zone, node2)
        self.assertEqual(result['setup'].next_zone.next_zone, node3)
        self.assertIsNone(result['setup'].next_zone.next_zone.next_zone)

    def test_multiple_blocks_chaining(self):
        """Test that multiple blocks get chained together correctly."""
        config = self.get_valid_config()
        toml_data = {
            'setup': [{'text': 'block1'}, {'text': 'block2'}],
            'loop': [{'text': 'test'}]
        }

        # Each block returns a single node
        block1_node = self.create_mock_zcp_node(sequence="setup", block=0)
        block2_node = self.create_mock_zcp_node(sequence="setup", block=1)
        loop_node = self.create_mock_zcp_node(sequence="loop", block=0)

        call_count = 0

        def mock_parser_func(block_data, config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return block1_node
            elif call_count == 2:
                return block2_node
            else:
                return loop_node

        mock_parser = MagicMock(side_effect=mock_parser_func)

        result = parse_sequences(toml_data, config, mock_parser)

        # Setup chain should start with block1_node
        self.assertEqual(result['setup'], block1_node)

        # block1_node should link to block2_node
        self.assertEqual(result['setup'].next_zone, block2_node)
        self.assertIsNone(result['setup'].next_zone.next_zone)

        # Loop should be separate
        self.assertEqual(result['loop'], loop_node)

    def test_complex_chaining_multiple_blocks_with_internal_chains(self):
        """Test complex chaining where each block has its own internal chain."""
        config = self.get_valid_config()
        toml_data = {
            'setup': [{'text': 'block1'}, {'text': 'block2'}],
            'loop': [{'text': 'test'}]
        }

        # Block 1: internal chain of 2 nodes
        b1_n1 = self.create_mock_zcp_node(sequence="setup", block=0)
        b1_n2 = self.create_mock_zcp_node(sequence="setup", block=0)
        b1_n1.next_zone = b1_n2

        # Block 2: internal chain of 2 nodes
        b2_n1 = self.create_mock_zcp_node(sequence="setup", block=1)
        b2_n2 = self.create_mock_zcp_node(sequence="setup", block=1)
        b2_n1.next_zone = b2_n2

        # Loop: single node
        loop_node = self.create_mock_zcp_node(sequence="loop", block=0)

        call_count = 0

        def mock_parser_func(block_data, config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return b1_n1  # Head of block 1 chain
            elif call_count == 2:
                return b2_n1  # Head of block 2 chain
            else:
                return loop_node

        mock_parser = MagicMock(side_effect=mock_parser_func)

        result = parse_sequences(toml_data, config, mock_parser)

        # Setup chain should start with first block's head
        self.assertEqual(result['setup'], b1_n1)

        # Should traverse block 1's internal chain
        self.assertEqual(result['setup'].next_zone, b1_n2)

        # Block 1's tail should link to block 2's head
        self.assertEqual(result['setup'].next_zone.next_zone, b2_n1)

        # Should traverse block 2's internal chain
        self.assertEqual(result['setup'].next_zone.next_zone.next_zone, b2_n2)

        # Block 2's tail should be the end
        self.assertIsNone(result['setup'].next_zone.next_zone.next_zone.next_zone)

        # Loop should be separate
        self.assertEqual(result['loop'], loop_node)


class TestErrorHandling(BaseSequenceTest):
    """Test error handling and edge cases."""

    def test_block_parser_exception_propagation(self):
        """Test that exceptions from block parser are properly wrapped."""
        config = self.get_valid_config()
        toml_data = {
            'setup': [{'text': 'test'}],
            'loop': [{'text': 'test'}]
        }

        mock_parser = MagicMock()
        mock_parser.side_effect = ValueError("Block parser error")

        with self.assertRaisesRegex(
                SequenceParseError,
                "Error parsing block 0 in sequence 'setup': Block parser error"
        ):
            parse_sequences(toml_data, config, mock_parser)

    def test_empty_config_sequences(self):
        """Test behavior with empty sequences list in config."""
        config = self.get_valid_config()
        config.sequences = []  # Empty sequences

        toml_data = {'setup': [{'text': 'test'}]}
        mock_parser = self.create_mock_block_parser()

        result = parse_sequences(toml_data, config, mock_parser)

        # Should return empty dictionary
        self.assertEqual(result, {})

        # Block parser should never be called
        mock_parser.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)