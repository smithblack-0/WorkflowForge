"""
Unit tests for LZCPNode class.

Tests cover:
1. Constructor and extensive validation (__post_init__)
2. State query methods (has_jump, is_terminal, etc.)
3. Array validation (dtypes, shapes, consistency)
4. Linked list operations (get_last_node)
5. Utility methods (num_tokens, get_active_tags)
6. Error handling and exception propagation
"""

import unittest
import numpy as np
from unittest.mock import Mock
from typing import Dict, Any, Optional

# Import the modules under test
from src.workflow_forge.zcp.nodes import LZCPNode, GraphError


class TestLZCPNodeConstruction(unittest.TestCase):
    """Test LZCPNode creation and validation."""

    def setUp(self):
        """Set up test fixtures."""
        # Basic valid node data
        self.valid_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'tokens': np.array([1, 2, 3], dtype=np.int32),
            'zone_advance_tokens': np.array([10], dtype=np.int32),
            'tags': np.array([True, False], dtype=np.bool_),
            'timeout': 1000,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_tokens': None,
            'jump_zone': None,
            'tool_callback': None
        }

    def test_valid_node_creation(self):
        """Test creating a valid LZCPNode with all required fields."""
        node = LZCPNode(**self.valid_node_data)

        self.assertEqual(node.sequence, 'test_sequence')
        self.assertEqual(node.block, 0)
        self.assertEqual(node.timeout, 1000)
        self.assertFalse(node.input)
        self.assertFalse(node.output)
        self.assertIsNone(node.next_zone)
        self.assertIsNone(node.jump_tokens)
        self.assertIsNone(node.jump_zone)
        self.assertIsNone(node.tool_callback)

        # Verify arrays
        np.testing.assert_array_equal(node.tokens, np.array([1, 2, 3]))
        np.testing.assert_array_equal(node.zone_advance_tokens, np.array([10]))
        np.testing.assert_array_equal(node.tags, np.array([True, False]))

    def test_node_with_jump_control(self):
        """Test creating node with jump flow control."""
        target_node = LZCPNode(**self.valid_node_data)

        node_data = self.valid_node_data.copy()
        node_data['jump_tokens'] = np.array([20, 21], dtype=np.int32)
        node_data['jump_zone'] = target_node

        node = LZCPNode(**node_data)
        self.assertIsNotNone(node.jump_tokens)
        self.assertIsNotNone(node.jump_zone)
        np.testing.assert_array_equal(node.jump_tokens, np.array([20, 21]))

    def test_input_output_flags(self):
        """Test nodes with input/output flags set."""
        # Test input node
        input_data = self.valid_node_data.copy()
        input_data['input'] = True
        input_node = LZCPNode(**input_data)
        self.assertTrue(input_node.input)

        # Test output node
        output_data = self.valid_node_data.copy()
        output_data['output'] = True
        output_node = LZCPNode(**output_data)
        self.assertTrue(output_node.output)

    def test_tool_callback_assignment(self):
        """Test node with tool callback."""
        mock_tool_callback = Mock()

        node_data = self.valid_node_data.copy()
        node_data['tool_callback'] = mock_tool_callback

        node = LZCPNode(**node_data)
        self.assertEqual(node.tool_callback, mock_tool_callback)


class TestLZCPNodeValidation(unittest.TestCase):
    """Test __post_init__ validation logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_node_data = {
            'sequence': 'validation_test',
            'block': 0,
            'tokens': np.array([1, 2, 3], dtype=np.int32),
            'zone_advance_tokens': np.array([10], dtype=np.int32),
            'tags': np.array([True, False], dtype=np.bool_),
            'timeout': 1000,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_tokens': None,
            'jump_zone': None
        }

    def test_jump_consistency_both_present(self):
        """Test validation when both jump_tokens and jump_zone are present."""
        target_node = LZCPNode(**self.valid_node_data)

        node_data = self.valid_node_data.copy()
        node_data['jump_tokens'] = np.array([20], dtype=np.int32)
        node_data['jump_zone'] = target_node

        # Should not raise exception
        node = LZCPNode(**node_data)
        self.assertIsNotNone(node.jump_tokens)
        self.assertIsNotNone(node.jump_zone)

    def test_jump_consistency_mismatch_tokens_only(self):
        """Test validation fails when only jump_tokens is present."""
        node_data = self.valid_node_data.copy()
        node_data['jump_tokens'] = np.array([20], dtype=np.int32)
        # jump_zone remains None

        with self.assertRaises(GraphError) as context:
            LZCPNode(**node_data)

        self.assertEqual(context.exception.sequence, 'validation_test')
        self.assertEqual(context.exception.block, 0)

    def test_jump_consistency_mismatch_zone_only(self):
        """Test validation fails when only jump_zone is present."""
        target_node = LZCPNode(**self.valid_node_data)

        node_data = self.valid_node_data.copy()
        node_data['jump_zone'] = target_node
        # jump_tokens remains None

        with self.assertRaises(GraphError) as context:
            LZCPNode(**node_data)

        self.assertEqual(context.exception.sequence, 'validation_test')
        self.assertEqual(context.exception.block, 0)

    def test_zone_advance_tokens_validation(self):
        """Test validation of zone_advance_tokens array."""
        # Test wrong type
        node_data = self.valid_node_data.copy()
        node_data['zone_advance_tokens'] = [10, 20]  # List instead of numpy array

        with self.assertRaises(GraphError):
            LZCPNode(**node_data)

        # Test wrong dimensions
        node_data = self.valid_node_data.copy()
        node_data['zone_advance_tokens'] = np.array([[10], [20]], dtype=np.int32)  # 2D

        with self.assertRaises(GraphError):
            LZCPNode(**node_data)

        # Test wrong dtype
        node_data = self.valid_node_data.copy()
        node_data['zone_advance_tokens'] = np.array([10.5], dtype=np.float32)  # Float

        with self.assertRaises(GraphError):
            LZCPNode(**node_data)

    def test_jump_tokens_validation(self):
        """Test validation of jump_tokens array when present."""
        target_node = LZCPNode(**self.valid_node_data)

        # Test wrong type
        node_data = self.valid_node_data.copy()
        node_data['jump_tokens'] = [20, 21]  # List instead of numpy array
        node_data['jump_zone'] = target_node

        with self.assertRaises(GraphError):
            LZCPNode(**node_data)

        # Test wrong dimensions
        node_data = self.valid_node_data.copy()
        node_data['jump_tokens'] = np.array([[20], [21]], dtype=np.int32)  # 2D
        node_data['jump_zone'] = target_node

        with self.assertRaises(GraphError):
            LZCPNode(**node_data)

        # Test wrong dtype
        node_data = self.valid_node_data.copy()
        node_data['jump_tokens'] = np.array([20.5], dtype=np.float32)  # Float
        node_data['jump_zone'] = target_node

        with self.assertRaises(GraphError):
            LZCPNode(**node_data)

    def test_tags_array_validation(self):
        """Test validation of tags array."""
        # Test wrong type
        node_data = self.valid_node_data.copy()
        node_data['tags'] = [True, False]  # List instead of numpy array

        with self.assertRaises(GraphError):
            LZCPNode(**node_data)

        # Test wrong dimensions
        node_data = self.valid_node_data.copy()
        node_data['tags'] = np.array([[True], [False]], dtype=np.bool_)  # 2D

        with self.assertRaises(GraphError):
            LZCPNode(**node_data)

        # Test wrong dtype
        node_data = self.valid_node_data.copy()
        node_data['tags'] = np.array([1, 0], dtype=np.int32)  # Integer

        with self.assertRaises(GraphError):
            LZCPNode(**node_data)

    def test_tokens_array_validation(self):
        """Test validation of tokens array."""
        # Test wrong type
        node_data = self.valid_node_data.copy()
        node_data['tokens'] = [1, 2, 3]  # List instead of numpy array

        with self.assertRaises(GraphError):
            LZCPNode(**node_data)

        # Test wrong dimensions
        node_data = self.valid_node_data.copy()
        node_data['tokens'] = np.array([[1], [2], [3]], dtype=np.int32)  # 2D

        with self.assertRaises(GraphError):
            LZCPNode(**node_data)

        # Test wrong dtype
        node_data = self.valid_node_data.copy()
        node_data['tokens'] = np.array([1.5, 2.5, 3.5], dtype=np.float32)  # Float

        with self.assertRaises(GraphError):
            LZCPNode(**node_data)


class TestLZCPNodeStateQueries(unittest.TestCase):
    """Test state query methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'tokens': np.array([1, 2, 3], dtype=np.int32),
            'zone_advance_tokens': np.array([10], dtype=np.int32),
            'tags': np.array([True, False, True], dtype=np.bool_),
            'timeout': 1000,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_tokens': None,
            'jump_zone': None
        }

    def test_has_jump_false(self):
        """Test has_jump returns False when no jump capability."""
        node = LZCPNode(**self.base_node_data)
        self.assertFalse(node.has_jump())

    def test_has_jump_true(self):
        """Test has_jump returns True when jump capability exists."""
        target_node = LZCPNode(**self.base_node_data)

        node_data = self.base_node_data.copy()
        node_data['jump_tokens'] = np.array([20], dtype=np.int32)
        node_data['jump_zone'] = target_node

        node = LZCPNode(**node_data)
        self.assertTrue(node.has_jump())

    def test_is_terminal_true(self):
        """Test is_terminal returns True for terminal nodes."""
        node = LZCPNode(**self.base_node_data)
        # No next_zone, no jump_zone
        self.assertTrue(node.is_terminal())

    def test_is_terminal_false_has_next(self):
        """Test is_terminal returns False when node has next_zone."""
        next_node = LZCPNode(**self.base_node_data)

        node_data = self.base_node_data.copy()
        node_data['next_zone'] = next_node

        node = LZCPNode(**node_data)
        self.assertFalse(node.is_terminal())

    def test_is_terminal_false_has_jump(self):
        """Test is_terminal returns False when node has jump capability."""
        target_node = LZCPNode(**self.base_node_data)

        node_data = self.base_node_data.copy()
        node_data['jump_tokens'] = np.array([20], dtype=np.int32)
        node_data['jump_zone'] = target_node

        node = LZCPNode(**node_data)
        self.assertFalse(node.is_terminal())

    def test_is_input_zone(self):
        """Test is_input_zone reflects input flag."""
        # Test False
        node = LZCPNode(**self.base_node_data)
        self.assertFalse(node.is_input_zone())

        # Test True
        node_data = self.base_node_data.copy()
        node_data['input'] = True
        input_node = LZCPNode(**node_data)
        self.assertTrue(input_node.is_input_zone())

    def test_is_output_zone(self):
        """Test is_output_zone reflects output flag."""
        # Test False
        node = LZCPNode(**self.base_node_data)
        self.assertFalse(node.is_output_zone())

        # Test True
        node_data = self.base_node_data.copy()
        node_data['output'] = True
        output_node = LZCPNode(**node_data)
        self.assertTrue(output_node.is_output_zone())

    def test_num_tokens(self):
        """Test num_tokens returns correct token count."""
        node = LZCPNode(**self.base_node_data)
        self.assertEqual(node.num_tokens(), 3)

        # Test with different token array
        node_data = self.base_node_data.copy()
        node_data['tokens'] = np.array([10, 20, 30, 40, 50], dtype=np.int32)

        node = LZCPNode(**node_data)
        self.assertEqual(node.num_tokens(), 5)

    def test_get_active_tags(self):
        """Test get_active_tags returns correct tag names."""
        node = LZCPNode(**self.base_node_data)
        tag_names = ["Training", "Correct", "Feedback"]

        # Should return names for True positions: index 0 and 2
        active_tags = node.get_active_tags(tag_names)
        self.assertEqual(active_tags, ["Training", "Feedback"])

    def test_get_active_tags_all_false(self):
        """Test get_active_tags with no active tags."""
        node_data = self.base_node_data.copy()
        node_data['tags'] = np.array([False, False, False], dtype=np.bool_)

        node = LZCPNode(**node_data)
        tag_names = ["Training", "Correct", "Feedback"]

        active_tags = node.get_active_tags(tag_names)
        self.assertEqual(active_tags, [])

    def test_get_active_tags_all_true(self):
        """Test get_active_tags with all tags active."""
        node_data = self.base_node_data.copy()
        node_data['tags'] = np.array([True, True, True], dtype=np.bool_)

        node = LZCPNode(**node_data)
        tag_names = ["Training", "Correct", "Feedback"]

        active_tags = node.get_active_tags(tag_names)
        self.assertEqual(active_tags, ["Training", "Correct", "Feedback"])

    def test_get_active_tags_length_mismatch(self):
        """Test get_active_tags raises error for mismatched lengths."""
        node = LZCPNode(**self.base_node_data)
        tag_names = ["Training", "Correct"]  # Only 2 names but 3 tags

        with self.assertRaises(ValueError) as context:
            node.get_active_tags(tag_names)

        self.assertIn("tag_names length must match tags array length", str(context.exception))


class TestLZCPNodeLinkedList(unittest.TestCase):
    """Test linked list operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'tokens': np.array([1, 2, 3], dtype=np.int32),
            'zone_advance_tokens': np.array([10], dtype=np.int32),
            'tags': np.array([True], dtype=np.bool_),
            'timeout': 1000,
            'input': False,
            'output': False
        }

    def test_get_last_node_single(self):
        """Test get_last_node with single node returns self."""
        node = LZCPNode(**self.base_node_data)

        last_node = node.get_last_node()
        self.assertEqual(last_node, node)

    def test_get_last_node_chain(self):
        """Test get_last_node traverses to end of chain."""
        # Create three nodes
        node1 = LZCPNode(**self.base_node_data)

        node2_data = self.base_node_data.copy()
        node2_data['block'] = 1
        node2 = LZCPNode(**node2_data)

        node3_data = self.base_node_data.copy()
        node3_data['block'] = 2
        node3 = LZCPNode(**node3_data)

        # Link them: node1 -> node2 -> node3
        node1.next_zone = node2
        node2.next_zone = node3

        # get_last_node should return node3 from any starting point
        self.assertEqual(node1.get_last_node(), node3)
        self.assertEqual(node2.get_last_node(), node3)
        self.assertEqual(node3.get_last_node(), node3)

    def test_chain_traversal(self):
        """Test manual traversal through a chain."""
        # Create 4-node chain
        nodes = []
        for i in range(4):
            node_data = self.base_node_data.copy()
            node_data['block'] = i
            nodes.append(LZCPNode(**node_data))

        # Link them
        for i in range(3):
            nodes[i].next_zone = nodes[i + 1]

        # Traverse and verify order
        current = nodes[0]
        visited_blocks = []

        while current is not None:
            visited_blocks.append(current.block)
            current = current.next_zone

        self.assertEqual(visited_blocks, [0, 1, 2, 3])


class TestLZCPNodeErrorHandling(unittest.TestCase):
    """Test error handling and exception propagation."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_node_data = {
            'sequence': 'error_sequence',
            'block': 5,
            'tokens': np.array([1, 2, 3], dtype=np.int32),
            'zone_advance_tokens': np.array([10], dtype=np.int32),
            'tags': np.array([True], dtype=np.bool_),
            'timeout': 1000,
            'input': False,
            'output': False
        }

    def test_validation_error_context(self):
        """Test that validation errors include proper context."""
        node_data = self.base_node_data.copy()
        node_data['tokens'] = [1, 2, 3]  # List instead of numpy array

        with self.assertRaises(GraphError) as context:
            LZCPNode(**node_data)

        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 5)

    def test_exception_chaining_preserved(self):
        """Test that original exceptions are preserved in the chain."""
        # This test verifies that the GraphError wraps the original TypeError
        node_data = self.base_node_data.copy()
        node_data['zone_advance_tokens'] = "not an array"  # Wrong type

        with self.assertRaises(GraphError) as context:
            LZCPNode(**node_data)

        # Should have proper context
        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 5)
        # Original exception should be chained
        self.assertIsNotNone(context.exception.__cause__)

    def test_multiple_validation_errors(self):
        """Test that first validation error is caught and reported."""
        node_data = self.base_node_data.copy()
        node_data['tokens'] = "not an array"  # Will fail first
        node_data['tags'] = [True, False]  # Would also fail

        with self.assertRaises(GraphError) as context:
            LZCPNode(**node_data)

        # Should report the first error encountered
        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 5)


if __name__ == "__main__":
    unittest.main()