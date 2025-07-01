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
from typing import Dict, Any, Optional, Tuple

# Import the modules under test
from workflow_forge.zcp.nodes import LZCPNode, GraphError


class BaseLZCPNodeTest(unittest.TestCase):
    """Base test class with common setup and helper methods."""

    def setUp(self):
        """Set up common test fixtures."""
        pass  # No complex mocks needed for LZCPNode

    def get_valid_tokens(self, tokens: Optional[list] = None) -> np.ndarray:
        """Create valid tokens array with proper dtype."""
        if tokens is None:
            tokens = [1, 2, 3]
        return np.array(tokens, dtype=np.int32)

    def get_valid_zone_advance_tokens(self, tokens: Optional[list] = None) -> np.ndarray:
        """Create valid zone_advance_tokens array with proper dtype."""
        if tokens is None:
            tokens = [10]
        return np.array(tokens, dtype=np.int32)

    def get_valid_jump_tokens(self, tokens: Optional[list] = None) -> np.ndarray:
        """Create valid jump_tokens array with proper dtype."""
        if tokens is None:
            tokens = [20, 21]
        return np.array(tokens, dtype=np.int32)

    def get_valid_tags(self, tags: Optional[list] = None) -> np.ndarray:
        """Create valid tags array with proper dtype."""
        if tags is None:
            tags = [True, False]
        return np.array(tags, dtype=np.bool_)

    def get_valid_escape_tokens(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create valid escape_tokens tuple with proper dtypes."""
        escape_start = np.array([100, 101], dtype=np.int32)
        escape_end = np.array([102, 103], dtype=np.int32)
        return (escape_start, escape_end)

    def get_valid_node_data(self, **overrides) -> Dict[str, Any]:
        """
        Return valid node data for testing, with optional field overrides.

        Args:
            **overrides: Fields to override in the base node data

        Returns:
            Dictionary of valid LZCPNode constructor arguments
        """
        base_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'tokens': self.get_valid_tokens(),
            'zone_advance_tokens': self.get_valid_zone_advance_tokens(),
            'escape_tokens': self.get_valid_escape_tokens(),
            'tags': self.get_valid_tags(),
            'timeout': 1000,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_tokens': None,
            'jump_zone': None,
            'tool_callback': None
        }
        base_data.update(overrides)
        return base_data

    def create_node(self, **overrides) -> LZCPNode:
        """
        Create an LZCPNode with valid data and optional overrides.

        Args:
            **overrides: Fields to override in the base node data

        Returns:
            Configured LZCPNode instance
        """
        return LZCPNode(**self.get_valid_node_data(**overrides))

    def create_node_chain(self, length: int, **base_overrides) -> LZCPNode:
        """
        Create a chain of linked LZCPNodes.

        Args:
            length: Number of nodes in the chain
            **base_overrides: Common overrides to apply to all nodes

        Returns:
            Head node of the chain
        """
        if length < 1:
            raise ValueError("Chain length must be at least 1")

        # Create nodes
        nodes = []
        for i in range(length):
            node_overrides = base_overrides.copy()
            node_overrides.update({'block': i})
            # Create unique token arrays for each node to avoid shared references
            node_overrides.update({
                'tokens': self.get_valid_tokens([i*10 + j for j in range(3)])
            })
            node = self.create_node(**node_overrides)
            nodes.append(node)

        # Link them
        for i in range(length - 1):
            nodes[i].next_zone = nodes[i + 1]

        return nodes[0]

    def create_jump_node(self, target_node: LZCPNode, jump_tokens: Optional[list] = None, **overrides) -> LZCPNode:
        """
        Create an LZCPNode with jump capability.

        Args:
            target_node: The node to jump to
            jump_tokens: Custom jump tokens (uses default if None)
            **overrides: Additional field overrides

        Returns:
            LZCPNode with jump capability configured
        """
        jump_overrides = {
            'jump_tokens': self.get_valid_jump_tokens(jump_tokens),
            'jump_zone': target_node
        }
        jump_overrides.update(overrides)
        return self.create_node(**jump_overrides)

    def assert_graph_error_context(self, context_manager, expected_sequence: str, expected_block: int):
        """
        Assert that a GraphError has the expected context information.

        Args:
            context_manager: The exception context manager
            expected_sequence: Expected sequence name in error
            expected_block: Expected block number in error
        """
        self.assertEqual(context_manager.exception.sequence, expected_sequence)
        self.assertEqual(context_manager.exception.block, expected_block)

    def assert_arrays_equal(self, actual: np.ndarray, expected: np.ndarray, msg: str = None):
        """Helper to assert numpy arrays are equal with better error messages."""
        np.testing.assert_array_equal(actual, expected, err_msg=msg)

    def assert_array_properties(self, array: np.ndarray, expected_dtype: np.dtype,
                              expected_ndim: int, array_name: str = "array"):
        """
        Assert array has expected dtype and dimensions.

        Args:
            array: The numpy array to check
            expected_dtype: Expected data type
            expected_ndim: Expected number of dimensions
            array_name: Name for error messages
        """
        self.assertIsInstance(array, np.ndarray, f"{array_name} must be numpy array")
        self.assertEqual(array.dtype, expected_dtype, f"{array_name} has wrong dtype")
        self.assertEqual(array.ndim, expected_ndim, f"{array_name} has wrong dimensions")


class TestLZCPNodeConstruction(BaseLZCPNodeTest):
    """Test LZCPNode creation and validation."""

    def test_valid_node_creation(self):
        """Test creating a valid LZCPNode with all required fields."""
        node_data = self.get_valid_node_data()
        node = LZCPNode(**node_data)

        self.assertEqual(node.sequence, 'test_sequence')
        self.assertEqual(node.block, 0)
        self.assertEqual(node.timeout, 1000)
        self.assertFalse(node.input)
        self.assertFalse(node.output)
        self.assertIsNone(node.next_zone)
        self.assertIsNone(node.jump_tokens)
        self.assertIsNone(node.jump_zone)
        self.assertIsNone(node.tool_callback)

        # Verify arrays with proper assertions
        self.assert_arrays_equal(node.tokens, np.array([1, 2, 3]))
        self.assert_arrays_equal(node.zone_advance_tokens, np.array([10]))
        self.assert_arrays_equal(node.tags, np.array([True, False]))

        # Verify escape_tokens tuple
        self.assertIsInstance(node.escape_tokens, tuple)
        self.assertEqual(len(node.escape_tokens), 2)
        self.assert_arrays_equal(node.escape_tokens[0], np.array([100, 101]))
        self.assert_arrays_equal(node.escape_tokens[1], np.array([102, 103]))

    def test_node_with_jump_control(self):
        """Test creating node with jump flow control."""
        target_node = self.create_node()
        jump_node = self.create_jump_node(target_node)

        self.assertIsNotNone(jump_node.jump_tokens)
        self.assertIsNotNone(jump_node.jump_zone)
        self.assert_arrays_equal(jump_node.jump_tokens, np.array([20, 21]))

    def test_input_output_flags(self):
        """Test nodes with input/output flags set."""
        # Test input node
        input_node = self.create_node(input=True)
        self.assertTrue(input_node.input)

        # Test output node
        output_node = self.create_node(output=True)
        self.assertTrue(output_node.output)

    def test_tool_callback_assignment(self):
        """Test node with tool callback."""
        mock_tool_callback = Mock()
        tool_node = self.create_node(tool_callback=mock_tool_callback)
        self.assertEqual(tool_node.tool_callback, mock_tool_callback)

    def test_escape_tokens_assignment(self):
        """Test that escape_tokens field is properly assigned."""
        node = self.create_node()

        # Verify escape_tokens structure
        self.assertIsInstance(node.escape_tokens, tuple)
        self.assertEqual(len(node.escape_tokens), 2)

        # Verify both arrays have correct values and types
        escape_start, escape_end = node.escape_tokens
        self.assert_array_properties(escape_start, np.int32, 1, "escape_start")
        self.assert_array_properties(escape_end, np.int32, 1, "escape_end")
        self.assert_arrays_equal(escape_start, np.array([100, 101]))
        self.assert_arrays_equal(escape_end, np.array([102, 103]))

        # Test with custom escape tokens
        custom_escape_tokens = (
            np.array([200, 201], dtype=np.int32),
            np.array([202, 203], dtype=np.int32)
        )
        custom_node = self.create_node(escape_tokens=custom_escape_tokens)
        self.assert_arrays_equal(custom_node.escape_tokens[0], np.array([200, 201]))
        self.assert_arrays_equal(custom_node.escape_tokens[1], np.array([202, 203]))


class TestLZCPNodeValidation(BaseLZCPNodeTest):
    """Test __post_init__ validation logic."""

    def test_jump_consistency_both_present(self):
        """Test validation when both jump_tokens and jump_zone are present."""
        target_node = self.create_node()
        jump_node = self.create_jump_node(target_node)

        # Should not raise exception
        self.assertIsNotNone(jump_node.jump_tokens)
        self.assertIsNotNone(jump_node.jump_zone)

    def test_jump_consistency_mismatch_tokens_only(self):
        """Test validation fails when only jump_tokens is present."""
        with self.assertRaises(GraphError) as context:
            self.create_node(
                sequence='validation_test',
                jump_tokens=self.get_valid_jump_tokens()
                # jump_zone remains None
            )

        self.assert_graph_error_context(context, 'validation_test', 0)

    def test_jump_consistency_mismatch_zone_only(self):
        """Test validation fails when only jump_zone is present."""
        target_node = self.create_node()

        with self.assertRaises(GraphError) as context:
            self.create_node(
                sequence='validation_test',
                jump_zone=target_node
                # jump_tokens remains None
            )

        self.assert_graph_error_context(context, 'validation_test', 0)

    def test_zone_advance_tokens_validation(self):
        """Test validation of zone_advance_tokens array."""
        # Test wrong type
        with self.assertRaises(GraphError):
            self.create_node(
                sequence='validation_test',
                zone_advance_tokens=[10, 20]  # List instead of numpy array
            )

        # Test wrong dimensions
        with self.assertRaises(GraphError):
            self.create_node(
                sequence='validation_test',
                zone_advance_tokens=np.array([[10], [20]], dtype=np.int32)  # 2D
            )

        # Test wrong dtype
        with self.assertRaises(GraphError):
            self.create_node(
                sequence='validation_test',
                zone_advance_tokens=np.array([10.5], dtype=np.float32)  # Float
            )

    def test_jump_tokens_validation(self):
        """Test validation of jump_tokens array when present."""
        target_node = self.create_node()

        # Test wrong type
        with self.assertRaises(GraphError):
            self.create_node(
                jump_tokens=[20, 21],  # List instead of numpy array
                jump_zone=target_node
            )

        # Test wrong dimensions
        with self.assertRaises(GraphError):
            self.create_node(
                jump_tokens=np.array([[20], [21]], dtype=np.int32),  # 2D
                jump_zone=target_node
            )

        # Test wrong dtype
        with self.assertRaises(GraphError):
            self.create_node(
                jump_tokens=np.array([20.5], dtype=np.float32),  # Float
                jump_zone=target_node
            )

    def test_tags_array_validation(self):
        """Test validation of tags array."""
        # Test wrong type
        with self.assertRaises(GraphError):
            self.create_node(tags=[True, False])  # List instead of numpy array

        # Test wrong dimensions
        with self.assertRaises(GraphError):
            self.create_node(
                tags=np.array([[True], [False]], dtype=np.bool_)  # 2D
            )

        # Test wrong dtype
        with self.assertRaises(GraphError):
            self.create_node(
                tags=np.array([1, 0], dtype=np.int32)  # Integer
            )

    def test_tokens_array_validation(self):
        """Test validation of tokens array."""
        # Test wrong type
        with self.assertRaises(GraphError):
            self.create_node(tokens=[1, 2, 3])  # List instead of numpy array

        # Test wrong dimensions
        with self.assertRaises(GraphError):
            self.create_node(
                tokens=np.array([[1], [2], [3]], dtype=np.int32)  # 2D
            )

        # Test wrong dtype
        with self.assertRaises(GraphError):
            self.create_node(
                tokens=np.array([1.5, 2.5, 3.5], dtype=np.float32)  # Float
            )

    def test_escape_tokens_validation(self):
        """Test validation of escape_tokens tuple and arrays."""
        # Test wrong type (not tuple)
        with self.assertRaises(GraphError):
            self.create_node(
                escape_tokens=[np.array([100]), np.array([101])]  # List instead of tuple
            )

        # Test wrong tuple length
        with self.assertRaises(GraphError):
            self.create_node(
                escape_tokens=(np.array([100], dtype=np.int32),)  # Only one element
            )

        # Test first element wrong type
        with self.assertRaises(GraphError):
            self.create_node(
                escape_tokens=([100, 101], np.array([102, 103], dtype=np.int32))
            )

        # Test second element wrong type
        with self.assertRaises(GraphError):
            self.create_node(
                escape_tokens=(np.array([100, 101], dtype=np.int32), [102, 103])
            )



class TestLZCPNodeStateQueries(BaseLZCPNodeTest):
    """Test state query methods."""

    def test_has_jump_false(self):
        """Test has_jump returns False when no jump capability."""
        node = self.create_node()
        self.assertFalse(node.has_jump())

    def test_has_jump_true(self):
        """Test has_jump returns True when jump capability exists."""
        target_node = self.create_node()
        jump_node = self.create_jump_node(target_node)
        self.assertTrue(jump_node.has_jump())

    def test_is_terminal_true(self):
        """Test is_terminal returns True for terminal nodes."""
        node = self.create_node()
        # No next_zone, no jump_zone
        self.assertTrue(node.is_terminal())

    def test_is_terminal_false_has_next(self):
        """Test is_terminal returns False when node has next_zone."""
        next_node = self.create_node()
        node = self.create_node(next_zone=next_node)
        self.assertFalse(node.is_terminal())

    def test_is_terminal_false_has_jump(self):
        """Test is_terminal returns False when node has jump capability."""
        target_node = self.create_node()
        jump_node = self.create_jump_node(target_node)
        self.assertFalse(jump_node.is_terminal())

    def test_is_input_zone(self):
        """Test is_input_zone reflects input flag."""
        # Test False
        node = self.create_node()
        self.assertFalse(node.is_input_zone())

        # Test True
        input_node = self.create_node(input=True)
        self.assertTrue(input_node.is_input_zone())

    def test_is_output_zone(self):
        """Test is_output_zone reflects output flag."""
        # Test False
        node = self.create_node()
        self.assertFalse(node.is_output_zone())

        # Test True
        output_node = self.create_node(output=True)
        self.assertTrue(output_node.is_output_zone())

    def test_num_tokens(self):
        """Test num_tokens returns correct token count."""
        node = self.create_node()
        self.assertEqual(node.num_tokens(), 3)

        # Test with different token array
        large_tokens_node = self.create_node(
            tokens=self.get_valid_tokens([10, 20, 30, 40, 50])
        )
        self.assertEqual(large_tokens_node.num_tokens(), 5)

    def test_get_active_tags(self):
        """Test get_active_tags returns correct tag names."""
        # Test with [True, False, True] pattern
        node = self.create_node(tags=self.get_valid_tags([True, False, True]))
        tag_names = ["Training", "Correct", "Feedback"]

        # Should return names for True positions: index 0 and 2
        active_tags = node.get_active_tags(tag_names)
        self.assertEqual(active_tags, ["Training", "Feedback"])

    def test_get_active_tags_all_false(self):
        """Test get_active_tags with no active tags."""
        node = self.create_node(tags=self.get_valid_tags([False, False, False]))
        tag_names = ["Training", "Correct", "Feedback"]

        active_tags = node.get_active_tags(tag_names)
        self.assertEqual(active_tags, [])

    def test_get_active_tags_all_true(self):
        """Test get_active_tags with all tags active."""
        node = self.create_node(tags=self.get_valid_tags([True, True, True]))
        tag_names = ["Training", "Correct", "Feedback"]

        active_tags = node.get_active_tags(tag_names)
        self.assertEqual(active_tags, ["Training", "Correct", "Feedback"])

    def test_get_active_tags_length_mismatch(self):
        """Test get_active_tags raises error for mismatched lengths."""
        node = self.create_node(tags=self.get_valid_tags([True, False, True]))
        tag_names = ["Training", "Correct"]  # Only 2 names but 3 tags

        with self.assertRaises(ValueError) as context:
            node.get_active_tags(tag_names)

        self.assertIn("tag_names length must match tags array length", str(context.exception))


class TestLZCPNodeLinkedList(BaseLZCPNodeTest):
    """Test linked list operations."""

    def test_get_last_node_single(self):
        """Test get_last_node with single node returns self."""
        node = self.create_node()
        last_node = node.get_last_node()
        self.assertEqual(last_node, node)

    def test_get_last_node_chain(self):
        """Test get_last_node traverses to end of chain."""
        # Create three-node chain
        head_node = self.create_node_chain(3)

        # Navigate to get individual nodes
        node1 = head_node
        node2 = head_node.next_zone
        node3 = head_node.next_zone.next_zone

        # get_last_node should return node3 from any starting point
        self.assertEqual(node1.get_last_node(), node3)
        self.assertEqual(node2.get_last_node(), node3)
        self.assertEqual(node3.get_last_node(), node3)

    def test_chain_traversal(self):
        """Test manual traversal through a chain."""
        # Create 4-node chain
        head_node = self.create_node_chain(4)

        # Traverse and verify order
        current = head_node
        visited_blocks = []

        while current is not None:
            visited_blocks.append(current.block)
            current = current.next_zone

        self.assertEqual(visited_blocks, [0, 1, 2, 3])


class TestLZCPNodeErrorHandling(BaseLZCPNodeTest):
    """Test error handling and exception propagation."""

    def test_validation_error_context(self):
        """Test that validation errors include proper context."""
        with self.assertRaises(GraphError) as context:
            self.create_node(
                sequence='error_sequence',
                block=5,
                tokens=[1, 2, 3]  # List instead of numpy array
            )

        self.assert_graph_error_context(context, "error_sequence", 5)

    def test_exception_chaining_preserved(self):
        """Test that original exceptions are preserved in the chain."""
        # This test verifies that the GraphError wraps the original TypeError
        with self.assertRaises(GraphError) as context:
            self.create_node(
                sequence='error_sequence',
                block=5,
                zone_advance_tokens="not an array"  # Wrong type
            )

        # Should have proper context
        self.assert_graph_error_context(context, "error_sequence", 5)
        # Original exception should be chained
        self.assertIsNotNone(context.exception.__cause__)

    def test_multiple_validation_errors(self):
        """Test that first validation error is caught and reported."""
        with self.assertRaises(GraphError) as context:
            self.create_node(
                sequence='error_sequence',
                block=5,
                tokens="not an array",  # Will fail first
                tags=[True, False]  # Would also fail
            )

        # Should report the first error encountered
        self.assert_graph_error_context(context, "error_sequence", 5)

    def test_escape_tokens_validation_error_context(self):
        """Test that escape_tokens validation errors include proper context."""
        with self.assertRaises(GraphError) as context:
            self.create_node(
                sequence='error_sequence',
                block=5,
                escape_tokens="not a tuple"  # Wrong type
            )

        self.assert_graph_error_context(context, "error_sequence", 5)


if __name__ == "__main__":
    unittest.main()