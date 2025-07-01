"""
Unit tests for SZCPNode class.

Tests cover:
1. Constructor and validation (__post_init__)
2. State query methods (has_jump, is_terminal, etc.)
3. Linked list operations (get_last_node, chain traversal)
4. Basic lowering operations (lower method)
5. Graph topology lowering (mathematical DCG-IO focus)
6. Error handling and exception propagation
7. Serialization and deserialization functionality
"""

import unittest
import numpy as np
import msgpack
from unittest.mock import Mock
from typing import Dict, Any

# Import the modules under test
from workflow_forge.zcp.nodes import SZCPNode, LZCPNode, GraphLoweringError
from workflow_forge.zcp.tag_converter import TagConverter
from workflow_forge.tokenizer_interface import TokenizerInterface


class BaseSZCPNodeTest(unittest.TestCase):
    """Base test class with common setup and helper methods."""

    def setUp(self):
        """Set up common test fixtures."""
        # Create mock tokenizer
        self.mock_tokenizer = Mock(spec=TokenizerInterface)
        self.mock_tokenizer.tokenize = Mock()
        self.mock_tokenizer.tokenize.side_effect = lambda text: np.array([hash(text) % 1000], dtype=np.int32)

        # Create mock tag converter
        self.mock_tag_converter = Mock(spec=TagConverter)
        self.mock_tag_converter.tensorize = Mock()
        self.mock_tag_converter.tensorize.side_effect = lambda tags: np.array([True] * len(tags), dtype=np.bool_)

        # Create mock tool registry
        self.mock_tool_callback = Mock()
        self.mock_tool_callback.return_value = np.array([99], dtype=np.int32)
        self.tool_registry = {'calculator': self.mock_tool_callback}

    def get_valid_node_data(self, **overrides) -> Dict[str, Any]:
        """
        Return valid node data for testing, with optional field overrides.

        Args:
            **overrides: Fields to override in the base node data

        Returns:
            Dictionary of valid SZCPNode constructor arguments
        """
        base_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'text': 'Test resolved text',
            'zone_advance_str': '[Answer]',
            'escape_strs': ('[Escape]', '[EndEscape]'),
            'tags': ['Training'],
            'timeout': 1000,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_advance_str': None,
            'jump_zone': None,
            'tool_name': None
        }
        base_data.update(overrides)
        return base_data

    def create_node(self, **overrides) -> SZCPNode:
        """
        Create an SZCPNode with valid data and optional overrides.

        Args:
            **overrides: Fields to override in the base node data

        Returns:
            Configured SZCPNode instance
        """
        return SZCPNode(**self.get_valid_node_data(**overrides))

    def create_node_chain(self, length: int, **base_overrides) -> SZCPNode:
        """
        Create a chain of linked SZCPNodes.

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
            node_overrides.update({'block': i, 'text': f'Test text {i}'})
            node = self.create_node(**node_overrides)
            nodes.append(node)

        # Link them
        for i in range(length - 1):
            nodes[i].next_zone = nodes[i + 1]

        return nodes[0]

    def create_jump_node(self, target_node: SZCPNode, jump_str: str = '[Jump]', **overrides) -> SZCPNode:
        """
        Create an SZCPNode with jump capability.

        Args:
            target_node: The node to jump to
            jump_str: The jump advance string
            **overrides: Additional field overrides

        Returns:
            SZCPNode with jump capability configured
        """
        jump_overrides = {
            'jump_advance_str': jump_str,
            'jump_zone': target_node
        }
        jump_overrides.update(overrides)
        return self.create_node(**jump_overrides)

    def create_topology_node(self, block: int, **overrides) -> SZCPNode:
        """Helper to create nodes for topology tests with unique text."""
        overrides.update({'block': block, 'text': f'text_{block}'})
        return self.create_node(**overrides)

    def assert_lzcp_node_properties(self, lzcp_node: LZCPNode, expected_sequence: str,
                                  expected_block: int, expected_timeout: int = 1000):
        """
        Assert common properties of an LZCPNode.

        Args:
            lzcp_node: The LZCPNode to validate
            expected_sequence: Expected sequence name
            expected_block: Expected block number
            expected_timeout: Expected timeout value
        """
        self.assertIsInstance(lzcp_node, LZCPNode)
        self.assertEqual(lzcp_node.sequence, expected_sequence)
        self.assertEqual(lzcp_node.block, expected_block)
        self.assertEqual(lzcp_node.timeout, expected_timeout)

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

    def nodes_data_equal(self, node1: SZCPNode, node2: SZCPNode) -> bool:
        """Helper to compare data equality between two nodes."""
        return (node1.sequence == node2.sequence and
                node1.block == node2.block and
                node1.text == node2.text and
                node1.zone_advance_str == node2.zone_advance_str and
                node1.escape_strs == node2.escape_strs and  # Added escape_strs check
                node1.tags == node2.tags and
                node1.timeout == node2.timeout and
                node1.input == node2.input and
                node1.output == node2.output and
                node1.jump_advance_str == node2.jump_advance_str and
                node1.tool_name == node2.tool_name)


class TestSZCPNodeConstruction(BaseSZCPNodeTest):
    """Test SZCPNode creation and validation."""

    def test_valid_node_creation(self):
        """Test creating a valid SZCPNode with all required fields."""
        node_data = self.get_valid_node_data()
        node = SZCPNode(**node_data)

        self.assertEqual(node.sequence, 'test_sequence')
        self.assertEqual(node.block, 0)
        self.assertEqual(node.text, 'Test resolved text')
        self.assertEqual(node.zone_advance_str, '[Answer]')
        self.assertEqual(node.escape_strs, ('[Escape]', '[EndEscape]'))
        self.assertEqual(node.tags, ['Training'])
        self.assertEqual(node.timeout, 1000)
        self.assertFalse(node.input)
        self.assertFalse(node.output)
        self.assertIsNone(node.next_zone)
        self.assertIsNone(node.jump_advance_str)
        self.assertIsNone(node.jump_zone)
        self.assertIsNone(node.tool_name)

    def test_input_output_flags(self):
        """Test nodes with input/output flags set."""
        # Test input node
        input_node = self.create_node(input=True)
        self.assertTrue(input_node.input)

        # Test output node
        output_node = self.create_node(output=True)
        self.assertTrue(output_node.output)

    def test_tool_name_assignment(self):
        """Test node with tool name."""
        tool_node = self.create_node(tool_name='calculator')
        self.assertEqual(tool_node.tool_name, 'calculator')

    def test_escape_strs_assignment(self):
        """Test that escape_strs field is properly assigned."""
        node = self.create_node()
        self.assertEqual(node.escape_strs, ('[Escape]', '[EndEscape]'))

        # Test with custom escape strings
        custom_node = self.create_node(escape_strs=('[Start]', '[End]'))
        self.assertEqual(custom_node.escape_strs, ('[Start]', '[End]'))

    def test_post_init_jump_consistency_both_present(self):
        """Test __post_init__ validation when both jump_advance_str and jump_zone are present."""
        target_node = self.create_node()
        jump_node = self.create_jump_node(target_node)

        # Should not raise exception
        self.assertIsNotNone(jump_node.jump_advance_str)
        self.assertIsNotNone(jump_node.jump_zone)


class TestSZCPNodeStateQueries(BaseSZCPNodeTest):
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

    def test_has_tool_false(self):
        """Test has_tool returns False when no tool name."""
        node = self.create_node()
        self.assertFalse(node.has_tool())

    def test_has_tool_true(self):
        """Test has_tool returns True when tool name exists."""
        tool_node = self.create_node(tool_name='calculator')
        self.assertTrue(tool_node.has_tool())


class TestSZCPNodeLinkedList(BaseSZCPNodeTest):
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

    def test_chain_building(self):
        """Test building chains by setting next_zone."""
        node1 = self.create_node(block=0)
        node2 = self.create_node(block=1)

        # Initially disconnected
        self.assertIsNone(node1.next_zone)
        self.assertIsNone(node2.next_zone)

        # Connect them
        node1.next_zone = node2

        # Verify connection
        self.assertEqual(node1.next_zone, node2)
        self.assertIsNone(node2.next_zone)

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


class TestSZCPNodeBasicLowering(BaseSZCPNodeTest):
    """Test basic lowering operations."""

    def test_lower_node_success(self):
        """Test lower creates valid LZCPNode."""
        node = self.create_node()

        result = node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify result using helper assertion
        self.assert_lzcp_node_properties(result, 'test_sequence', 0)

        # Verify basic properties
        self.assertFalse(result.input)
        self.assertFalse(result.output)
        self.assertIsNone(result.next_zone)
        self.assertIsNone(result.jump_tokens)
        self.assertIsNone(result.jump_zone)
        self.assertIsNone(result.tool_callback)

        # Verify tokenizer was called
        self.mock_tokenizer.tokenize.assert_any_call('Test resolved text')
        self.mock_tokenizer.tokenize.assert_any_call('[Answer]')

        # Verify tag converter was called
        self.mock_tag_converter.tensorize.assert_called_with(['Training'])

    def test_lower_node_field_preservation(self):
        """Test that all fields are properly preserved during lowering."""
        node = self.create_node(
            input=True,
            output=True,
            tool_name='calculator'
        )

        result = node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify flags preserved
        self.assertTrue(result.input)
        self.assertTrue(result.output)
        self.assertEqual(result.tool_callback, self.mock_tool_callback)

    def test_lower_single_node(self):
        """Test lower() method with single node."""
        node = self.create_node()
        result = node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Should return LZCPNode
        self.assertIsInstance(result, LZCPNode)
        self.assertEqual(result.sequence, 'test_sequence')
        self.assertIsNone(result.next_zone)

    def test_lower_chain_of_nodes(self):
        """Test lower() method with chain of nodes."""
        # Create two-node chain
        head_node = self.create_node_chain(2)

        # Lower the chain
        result_head = head_node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify chain structure is preserved
        self.assert_lzcp_node_properties(result_head, 'test_sequence', 0)

        self.assertIsNotNone(result_head.next_zone)
        self.assert_lzcp_node_properties(result_head.next_zone, 'test_sequence', 1)

        self.assertIsNone(result_head.next_zone.next_zone)

    def test_lower_with_jump(self):
        """Test lower() preserves jump information."""
        target_node = self.create_node()
        jump_node = self.create_jump_node(target_node)

        result = jump_node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify jump tokens were tokenized and preserved
        self.assertIsNotNone(result.jump_tokens)
        self.assertIsNotNone(result.jump_zone)
        self.assertIsInstance(result.jump_zone, LZCPNode)

        # Verify jump string was tokenized
        self.mock_tokenizer.tokenize.assert_any_call('[Jump]')

    def test_lower_with_tool(self):
        """Test lower() preserves tool information."""
        tool_node = self.create_node(tool_name='calculator')
        result = tool_node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify tool callback preserved
        self.assertEqual(result.tool_callback, self.mock_tool_callback)

    def test_lower_missing_tool_in_registry(self):
        """Test lower() fails when tool not found in registry."""
        node = self.create_node(tool_name='missing_tool')

        with self.assertRaises(GraphLoweringError) as context:
            node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        self.assertEqual(context.exception.sequence, 'test_sequence')
        self.assertEqual(context.exception.block, 0)


class TestSZCPNodeGraphTopology(BaseSZCPNodeTest):
    """Test graph topology lowering - the mathematical DCG-IO focus."""

    def test_linear_chain_topology(self):
        """Test: A → B → C → Terminal"""
        # Build the graph
        nodeA = self.create_topology_node(0)
        nodeB = self.create_topology_node(1)
        nodeC = self.create_topology_node(2)
        terminal = self.create_topology_node(3)

        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeC.next_zone = terminal

        # Lower from head
        result = nodeA.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify structure preservation
        self.assertEqual(result.block, 0)  # A
        self.assertEqual(result.next_zone.block, 1)  # B
        self.assertEqual(result.next_zone.next_zone.block, 2)  # C
        self.assertEqual(result.next_zone.next_zone.next_zone.block, 3)  # Terminal
        self.assertIsNone(result.next_zone.next_zone.next_zone.next_zone)

    def test_simple_branch_topology(self):
        """Test: A → B (jump to D), A → B → C → D → Terminal"""
        # Build the graph
        nodeA = self.create_topology_node(0)
        nodeB = self.create_topology_node(1)
        nodeC = self.create_topology_node(2)
        nodeD = self.create_topology_node(3)
        terminal = self.create_topology_node(4)

        # Linear path: A → B → C → D → Terminal
        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeC.next_zone = nodeD
        nodeD.next_zone = terminal

        # Jump path: B can jump to D
        nodeB.jump_advance_str = '[Jump]'
        nodeB.jump_zone = nodeD

        # Lower from head
        result = nodeA.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify linear structure preserved
        self.assertEqual(result.block, 0)  # A
        self.assertEqual(result.next_zone.block, 1)  # B
        self.assertEqual(result.next_zone.next_zone.block, 2)  # C
        self.assertEqual(result.next_zone.next_zone.next_zone.block, 3)  # D

        # Verify jump preserved
        nodeB_lowered = result.next_zone
        self.assertIsNotNone(nodeB_lowered.jump_tokens)
        self.assertIsNotNone(nodeB_lowered.jump_zone)
        self.assertEqual(nodeB_lowered.jump_zone.block, 3)  # Points to D

    def test_simple_loop_topology(self):
        """Test: A → B → C (jump back to B) → Terminal"""
        # Build the graph
        nodeA = self.create_topology_node(0)
        nodeB = self.create_topology_node(1)
        nodeC = self.create_topology_node(2)
        terminal = self.create_topology_node(3)

        # Linear path: A → B → C → Terminal
        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeC.next_zone = terminal

        # Loop: C can jump back to B
        nodeC.jump_advance_str = '[Jump]'
        nodeC.jump_zone = nodeB

        # Lower from head (tests cycle handling)
        result = nodeA.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify structure preserved
        self.assertEqual(result.block, 0)  # A
        self.assertEqual(result.next_zone.block, 1)  # B
        self.assertEqual(result.next_zone.next_zone.block, 2)  # C

        # Verify loop preserved
        nodeC_lowered = result.next_zone.next_zone
        self.assertIsNotNone(nodeC_lowered.jump_zone)
        self.assertEqual(nodeC_lowered.jump_zone.block, 1)  # Points back to B

        # Verify no infinite recursion occurred (cycle detection worked)
        nodeB_lowered = result.next_zone
        self.assertEqual(nodeB_lowered.block, 1)

    def test_convergent_paths_topology(self):
        """Test: A → B → D, A → C → D → Terminal"""
        # Build the graph with convergence
        nodeA = self.create_topology_node(0)
        nodeB = self.create_topology_node(1)
        nodeC = self.create_topology_node(2)
        nodeD = self.create_topology_node(3)
        terminal = self.create_topology_node(4)

        # Path 1: A → B → D
        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeD

        # Path 2: A can also jump to C, C → D
        nodeA.jump_advance_str = '[Jump]'
        nodeA.jump_zone = nodeC
        nodeC.next_zone = nodeD

        # Continuation: D → Terminal
        nodeD.next_zone = terminal

        # Lower from head
        result = nodeA.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify both paths lead to same D
        path1_D = result.next_zone.next_zone  # A → B → D
        path2_D = result.jump_zone.next_zone  # A → C → D

        self.assertEqual(path1_D.block, 3)  # D
        self.assertEqual(path2_D.block, 3)  # D

        # Both should point to same terminal
        self.assertEqual(path1_D.next_zone.block, 4)  # Terminal
        self.assertEqual(path2_D.next_zone.block, 4)  # Terminal

    def test_cycle_detection_prevents_infinite_recursion(self):
        """Test that lowering with cycles doesn't cause infinite recursion."""
        # Create a cycle: A → B → C → B
        nodeA = self.create_topology_node(0)
        nodeB = self.create_topology_node(1)
        nodeC = self.create_topology_node(2)

        # Create the cycle
        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeC.next_zone = nodeB  # Cycle back to B

        # This should complete without infinite recursion
        result = nodeA.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify the cycle is preserved in lowered graph
        self.assertEqual(result.block, 0)  # A
        self.assertEqual(result.next_zone.block, 1)  # B
        self.assertEqual(result.next_zone.next_zone.block, 2)  # C
        self.assertEqual(result.next_zone.next_zone.next_zone.block, 1)  # Back to B

        # Verify they're the same instance (proper cycle detection)
        nodeB_lowered = result.next_zone
        nodeB_from_cycle = result.next_zone.next_zone.next_zone
        self.assertEqual(nodeB_lowered, nodeB_from_cycle)


class TestSZCPNodeErrorHandling(BaseSZCPNodeTest):
    """Test error handling and exception propagation."""

    def test_lower_error_propagation(self):
        """Test error propagation when lowering fails."""
        # Make tokenizer fail
        self.mock_tokenizer.tokenize.side_effect = RuntimeError("Tokenization failed")

        node = self.create_node(sequence='error_sequence', block=5)

        with self.assertRaises(GraphLoweringError) as context:
            node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 5)

    def test_exception_chaining_preserved(self):
        """Test that original exceptions are preserved in the chain."""
        # Make tag converter fail
        original_error = ValueError("Tag conversion failed")
        self.mock_tag_converter.tensorize.side_effect = original_error

        node = self.create_node(sequence='error_sequence', block=5)

        with self.assertRaises(GraphLoweringError) as context:
            node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Check that original exception is chained
        self.assertIsInstance(context.exception.__cause__, ValueError)


class TestSZCPSerialization(BaseSZCPNodeTest):
    """Test serialization and deserialization functionality."""

    def test_single_node_round_trip(self):
        """Test serialize→deserialize round-trip for single node."""
        node = self.create_node()

        # Serialize
        serialized = node.serialize()

        # Deserialize
        deserialized = SZCPNode.deserialize(serialized)

        # Verify all properties match (including escape_strs)
        self.assertTrue(self.nodes_data_equal(node, deserialized))
        self.assertIsNone(deserialized.next_zone)
        self.assertIsNone(deserialized.jump_zone)

    def test_node_with_all_fields(self):
        """Test serialization of node with all optional fields populated."""
        target_node = self.create_node(block=1)

        node = self.create_node(
            input=True,
            output=True,
            tool_name='calculator',
            jump_advance_str='[Jump]',
            jump_zone=target_node,
            next_zone=target_node
        )

        # Serialize and deserialize
        serialized = node.serialize()
        deserialized = SZCPNode.deserialize(serialized)

        # Verify all fields preserved
        self.assertTrue(deserialized.input)
        self.assertTrue(deserialized.output)
        self.assertEqual(deserialized.tool_name, 'calculator')
        self.assertEqual(deserialized.jump_advance_str, '[Jump]')
        self.assertEqual(deserialized.escape_strs, ('[Escape]', '[EndEscape]'))  # Verify escape_strs
        self.assertIsNotNone(deserialized.jump_zone)
        self.assertIsNotNone(deserialized.next_zone)

    def test_escape_strs_serialization(self):
        """Test that escape_strs field is properly serialized and deserialized."""
        # Test with default escape_strs
        node1 = self.create_node()
        serialized1 = node1.serialize()
        deserialized1 = SZCPNode.deserialize(serialized1)
        self.assertEqual(deserialized1.escape_strs, ('[Escape]', '[EndEscape]'))

        # Test with custom escape_strs
        node2 = self.create_node(escape_strs=('[Start]', '[End]'))
        serialized2 = node2.serialize()
        deserialized2 = SZCPNode.deserialize(serialized2)
        self.assertEqual(deserialized2.escape_strs, ('[Start]', '[End]'))

    def test_simple_chain_serialization(self):
        """Test serialization of simple A→B→C chain."""
        # Create chain
        head_node = self.create_node_chain(3)

        # Serialize and deserialize
        serialized = head_node.serialize()
        deserialized = SZCPNode.deserialize(serialized)

        # Verify chain structure preserved
        self.assertEqual(deserialized.block, 0)  # A
        self.assertEqual(deserialized.next_zone.block, 1)  # B
        self.assertEqual(deserialized.next_zone.next_zone.block, 2)  # C
        self.assertIsNone(deserialized.next_zone.next_zone.next_zone)

    def test_jump_reference_preservation(self):
        """Test that jump references are preserved correctly."""
        nodeA = self.create_topology_node(0)
        nodeB = self.create_topology_node(1)
        nodeC = self.create_topology_node(2)

        # A → B → C, with B jumping to C
        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeB.jump_advance_str = '[Jump]'
        nodeB.jump_zone = nodeC

        # Serialize and deserialize
        serialized = nodeA.serialize()
        deserialized = SZCPNode.deserialize(serialized)

        # Verify jump preserved
        nodeB_deser = deserialized.next_zone
        nodeC_deser = deserialized.next_zone.next_zone

        self.assertEqual(nodeB_deser.jump_advance_str, '[Jump]')
        self.assertEqual(nodeB_deser.jump_zone, nodeC_deser)  # Same object reference

    def test_cycle_serialization(self):
        """Test serialization of graph with cycles."""
        # Create cycle: A → B → C → B
        nodeA = self.create_topology_node(0)
        nodeB = self.create_topology_node(1)
        nodeC = self.create_topology_node(2)

        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeC.next_zone = nodeB  # Cycle back

        # Serialize and deserialize
        serialized = nodeA.serialize()
        deserialized = SZCPNode.deserialize(serialized)

        # Verify cycle preserved
        nodeA_deser = deserialized
        nodeB_deser = deserialized.next_zone
        nodeC_deser = deserialized.next_zone.next_zone

        self.assertEqual(nodeA_deser.block, 0)
        self.assertEqual(nodeB_deser.block, 1)
        self.assertEqual(nodeC_deser.block, 2)

        # Verify cycle: C should point back to B (same object)
        self.assertEqual(nodeC_deser.next_zone, nodeB_deser)

    def test_convergent_paths_serialization(self):
        """Test serialization where multiple paths converge on same node."""
        # A → B → D, A → C → D
        nodeA = self.create_topology_node(0)
        nodeB = self.create_topology_node(1)
        nodeC = self.create_topology_node(2)
        nodeD = self.create_topology_node(3)

        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeD
        nodeA.jump_advance_str = '[Jump]'
        nodeA.jump_zone = nodeC
        nodeC.next_zone = nodeD

        # Serialize and deserialize
        serialized = nodeA.serialize()
        deserialized = SZCPNode.deserialize(serialized)

        # Verify convergence: both paths should lead to same D object
        path1_D = deserialized.next_zone.next_zone  # A → B → D
        path2_D = deserialized.jump_zone.next_zone  # A → C → D

        self.assertEqual(path1_D.block, 3)
        self.assertEqual(path2_D.block, 3)
        self.assertEqual(path1_D, path2_D)  # Same object reference

    def test_dictionary_structure_validation(self):
        """Test that serialized format has correct structure."""
        node = self.create_node()
        serialized = node.serialize()

        # Should be dict mapping int indices to node data
        self.assertIsInstance(serialized, dict)
        self.assertIn(0, serialized)  # Root node always index 0

        # Each entry should have data and links sections
        node_entry = serialized[0]
        self.assertIn('data', node_entry)
        self.assertIn('links', node_entry)

        # Data section should contain node fields (including escape_strs)
        data = node_entry['data']
        self.assertEqual(data['sequence'], 'test_sequence')
        self.assertEqual(data['block'], 0)
        self.assertEqual(data['text'], 'Test resolved text')
        self.assertEqual(data['escape_strs'], ('[Escape]', '[EndEscape]'))  # Verify escape_strs in serialized data

        # Links section should contain reference indices
        links = node_entry['links']
        self.assertIn('next_zone_index', links)
        self.assertIn('jump_zone_index', links)

    def test_index_assignment_correctness(self):
        """Test that indices are assigned correctly."""
        # Create chain A → B → C
        head_node = self.create_node_chain(3)

        serialized = head_node.serialize()

        # Should have exactly 3 nodes with indices 0, 1, 2
        self.assertEqual(len(serialized), 3)
        self.assertIn(0, serialized)
        self.assertIn(1, serialized)
        self.assertIn(2, serialized)

        # Check linkage
        self.assertEqual(serialized[0]['links']['next_zone_index'], 1)  # A → B
        self.assertEqual(serialized[1]['links']['next_zone_index'], 2)  # B → C
        self.assertIsNone(serialized[2]['links']['next_zone_index'])  # C → None

    def test_round_trip_identity(self):
        """Test that serialize→deserialize produces functionally identical graph."""
        # Create complex graph with jumps and cycles
        nodeA = self.create_topology_node(0)
        nodeB = self.create_topology_node(1)
        nodeC = self.create_topology_node(2)

        # A → B → C, A can jump to C, C cycles back to B
        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeA.jump_advance_str = '[Jump]'
        nodeA.jump_zone = nodeC
        nodeC.jump_advance_str = '[Loop]'
        nodeC.jump_zone = nodeB

        # Also test various field combinations
        nodeB.input = True
        nodeC.output = True
        nodeC.tool_name = 'calculator'

        # Serialize and deserialize
        serialized = nodeA.serialize()
        deserialized = SZCPNode.deserialize(serialized)

        # Verify graph structure identity
        self.verify_graph_identity(nodeA, deserialized)

    def test_reference_integrity(self):
        """Test that all object references are correctly preserved."""
        # Create diamond pattern: A → B → D, A → C → D
        nodeA = self.create_topology_node(0)
        nodeB = self.create_topology_node(1)
        nodeC = self.create_topology_node(2)
        nodeD = self.create_topology_node(3)

        nodeA.next_zone = nodeB
        nodeA.jump_advance_str = '[Jump]'
        nodeA.jump_zone = nodeC
        nodeB.next_zone = nodeD
        nodeC.next_zone = nodeD

        # Serialize and deserialize
        serialized = nodeA.serialize()
        deserialized = SZCPNode.deserialize(serialized)

        # Both paths should converge on exactly the same D object
        path1_D = deserialized.next_zone.next_zone
        path2_D = deserialized.jump_zone.next_zone

        self.assertIs(path1_D, path2_D)  # Same object identity

    def verify_graph_identity(self, original: SZCPNode, deserialized: SZCPNode):
        """Helper to verify two graphs have identical structure and data."""
        visited_orig = {}
        visited_deser = {}

        def traverse_and_compare(orig, deser, visited_o, visited_d):
            if orig is None and deser is None:
                return True
            if orig is None or deser is None:
                return False

            # Check if we've seen these nodes before (cycle detection)
            orig_id = id(orig)
            deser_id = id(deser)

            if orig_id in visited_o:
                return deser_id in visited_d and visited_o[orig_id] == visited_d[deser_id]

            # Mark as visited
            visit_count = len(visited_o)
            visited_o[orig_id] = visit_count
            visited_d[deser_id] = visit_count

            # Compare node data (using our updated helper that includes escape_strs)
            if not self.nodes_data_equal(orig, deser):
                return False

            # Recursively compare linked nodes
            return (traverse_and_compare(orig.next_zone, deser.next_zone, visited_o, visited_d) and
                    traverse_and_compare(orig.jump_zone, deser.jump_zone, visited_o, visited_d))

        self.assertTrue(traverse_and_compare(original, deserialized, visited_orig, visited_deser))

    # Add this test method to the TestSZCPSerialization class in test_szcp_node.py

    def test_msgpack_round_trip_serialization(self):
        """Test that SZCP serialization survives JSON round-trip (string key conversion)."""
        # Create a simple chain
        head_node = self.create_node_chain(2)

        # Serialize to dict
        serialized_dict = head_node.serialize()

        # Simulate workflow round-trip
        pack_str = msgpack.packb(serialized_dict)
        parsed_dict = msgpack.unpackb(pack_str, strict_map_key=False)

        # Verify JSON converted keys to strings
        self.assertIn(0, parsed_dict)
        self.assertIn(1, parsed_dict)

        # This SHOULD succeed - deserialize should handle string keys
        deserialized = SZCPNode.deserialize(parsed_dict)

        # Verify the deserialization worked correctly
        self.assertIsInstance(deserialized, SZCPNode)
        self.assertEqual(deserialized.block, 0)
        self.assertIsNotNone(deserialized.next_zone)
        self.assertEqual(deserialized.next_zone.block, 1)
        self.assertIsNone(deserialized.next_zone.next_zone)

if __name__ == "__main__":
    unittest.main()