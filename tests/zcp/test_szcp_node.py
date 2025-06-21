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
from unittest.mock import Mock
from typing import Dict, Any

# Import the modules under test
from src.workflow_forge.zcp.nodes import SZCPNode, LZCPNode, GraphError, GraphLoweringError
from src.workflow_forge.backend.tag_converter import TagConverter
from src.workflow_forge.tokenizer_interface import TokenizerInterface


class TestSZCPNodeConstruction(unittest.TestCase):
    """Test SZCPNode creation and validation."""

    def setUp(self):
        """Set up test fixtures."""
        # Basic valid node data
        self.valid_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'text': 'Test resolved text',
            'zone_advance_str': '[Answer]',
            'tags': ['Training', 'Correct'],
            'timeout': 1000,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_advance_str': None,
            'jump_zone': None,
            'tool_name': None
        }

    def test_valid_node_creation(self):
        """Test creating a valid SZCPNode with all required fields."""
        node = SZCPNode(**self.valid_node_data)

        self.assertEqual(node.sequence, 'test_sequence')
        self.assertEqual(node.block, 0)
        self.assertEqual(node.text, 'Test resolved text')
        self.assertEqual(node.zone_advance_str, '[Answer]')
        self.assertEqual(node.tags, ['Training', 'Correct'])
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
        input_data = self.valid_node_data.copy()
        input_data['input'] = True
        input_node = SZCPNode(**input_data)
        self.assertTrue(input_node.input)

        # Test output node
        output_data = self.valid_node_data.copy()
        output_data['output'] = True
        output_node = SZCPNode(**output_data)
        self.assertTrue(output_node.output)

    def test_tool_name_assignment(self):
        """Test node with tool name."""
        tool_data = self.valid_node_data.copy()
        tool_data['tool_name'] = 'calculator'
        tool_node = SZCPNode(**tool_data)
        self.assertEqual(tool_node.tool_name, 'calculator')

    def test_post_init_jump_consistency_both_present(self):
        """Test __post_init__ validation when both jump_advance_str and jump_zone are present."""
        target_node = SZCPNode(**self.valid_node_data)

        node_data = self.valid_node_data.copy()
        node_data['jump_advance_str'] = '[Jump]'
        node_data['jump_zone'] = target_node

        # Should not raise exception
        node = SZCPNode(**node_data)
        self.assertIsNotNone(node.jump_advance_str)
        self.assertIsNotNone(node.jump_zone)



class TestSZCPNodeStateQueries(unittest.TestCase):
    """Test state query methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'text': 'Test text',
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_advance_str': None,
            'jump_zone': None,
            'tool_name': None
        }

    def test_has_jump_false(self):
        """Test has_jump returns False when no jump capability."""
        node = SZCPNode(**self.base_node_data)
        self.assertFalse(node.has_jump())

    def test_has_jump_true(self):
        """Test has_jump returns True when jump capability exists."""
        target_node = SZCPNode(**self.base_node_data)

        node_data = self.base_node_data.copy()
        node_data['jump_advance_str'] = '[Jump]'
        node_data['jump_zone'] = target_node

        node = SZCPNode(**node_data)
        self.assertTrue(node.has_jump())

    def test_is_terminal_true(self):
        """Test is_terminal returns True for terminal nodes."""
        node = SZCPNode(**self.base_node_data)
        # No next_zone, no jump_zone
        self.assertTrue(node.is_terminal())

    def test_is_terminal_false_has_next(self):
        """Test is_terminal returns False when node has next_zone."""
        next_node = SZCPNode(**self.base_node_data)

        node_data = self.base_node_data.copy()
        node_data['next_zone'] = next_node

        node = SZCPNode(**node_data)
        self.assertFalse(node.is_terminal())

    def test_is_terminal_false_has_jump(self):
        """Test is_terminal returns False when node has jump capability."""
        target_node = SZCPNode(**self.base_node_data)

        node_data = self.base_node_data.copy()
        node_data['jump_advance_str'] = '[Jump]'
        node_data['jump_zone'] = target_node

        node = SZCPNode(**node_data)
        self.assertFalse(node.is_terminal())

    def test_is_input_zone(self):
        """Test is_input_zone reflects input flag."""
        # Test False
        node = SZCPNode(**self.base_node_data)
        self.assertFalse(node.is_input_zone())

        # Test True
        node_data = self.base_node_data.copy()
        node_data['input'] = True
        input_node = SZCPNode(**node_data)
        self.assertTrue(input_node.is_input_zone())

    def test_is_output_zone(self):
        """Test is_output_zone reflects output flag."""
        # Test False
        node = SZCPNode(**self.base_node_data)
        self.assertFalse(node.is_output_zone())

        # Test True
        node_data = self.base_node_data.copy()
        node_data['output'] = True
        output_node = SZCPNode(**node_data)
        self.assertTrue(output_node.is_output_zone())

    def test_has_tool_false(self):
        """Test has_tool returns False when no tool name."""
        node = SZCPNode(**self.base_node_data)
        self.assertFalse(node.has_tool())

    def test_has_tool_true(self):
        """Test has_tool returns True when tool name exists."""
        node_data = self.base_node_data.copy()
        node_data['tool_name'] = 'calculator'

        node = SZCPNode(**node_data)
        self.assertTrue(node.has_tool())


class TestSZCPNodeLinkedList(unittest.TestCase):
    """Test linked list operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'text': 'Test text',
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000,
            'input': False,
            'output': False
        }

    def test_get_last_node_single(self):
        """Test get_last_node with single node returns self."""
        node = SZCPNode(**self.base_node_data)

        last_node = node.get_last_node()
        self.assertEqual(last_node, node)

    def test_get_last_node_chain(self):
        """Test get_last_node traverses to end of chain."""
        # Create three nodes
        node1 = SZCPNode(**self.base_node_data)

        node2_data = self.base_node_data.copy()
        node2_data['block'] = 1
        node2 = SZCPNode(**node2_data)

        node3_data = self.base_node_data.copy()
        node3_data['block'] = 2
        node3 = SZCPNode(**node3_data)

        # Link them: node1 -> node2 -> node3
        node1.next_zone = node2
        node2.next_zone = node3

        # get_last_node should return node3 from any starting point
        self.assertEqual(node1.get_last_node(), node3)
        self.assertEqual(node2.get_last_node(), node3)
        self.assertEqual(node3.get_last_node(), node3)

    def test_chain_building(self):
        """Test building chains by setting next_zone."""
        node1 = SZCPNode(**self.base_node_data)

        node2_data = self.base_node_data.copy()
        node2_data['block'] = 1
        node2 = SZCPNode(**node2_data)

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
        nodes = []
        for i in range(4):
            node_data = self.base_node_data.copy()
            node_data['block'] = i
            nodes.append(SZCPNode(**node_data))

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


class TestSZCPNodeBasicLowering(unittest.TestCase):
    """Test basic lowering operations."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock tokenizer properly
        self.mock_tokenizer = Mock(spec=TokenizerInterface)
        self.mock_tokenizer.tokenize = Mock()
        self.mock_tokenizer.tokenize.side_effect = lambda text: np.array([hash(text) % 1000], dtype=np.int32)

        # Create mock tag converter properly
        self.mock_tag_converter = Mock(spec=TagConverter)
        self.mock_tag_converter.tensorize = Mock()
        self.mock_tag_converter.tensorize.side_effect = lambda tags: np.array([True] * len(tags), dtype=np.bool_)

        # Create mock tool registry
        self.mock_tool_callback = Mock()
        self.mock_tool_callback.return_value = np.array([99], dtype=np.int32)
        self.tool_registry = {'calculator': self.mock_tool_callback}

        # Base node data
        self.base_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'text': 'Test resolved text',
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_advance_str': None,
            'jump_zone': None,
            'tool_name': None
        }

    def test_lower_node_success(self):
        """Test lower creates valid LZCPNode."""
        node = SZCPNode(**self.base_node_data)

        result = node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify result type and basic properties
        self.assertIsInstance(result, LZCPNode)
        self.assertEqual(result.sequence, 'test_sequence')
        self.assertEqual(result.block, 0)
        self.assertEqual(result.timeout, 1000)
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
        self.mock_tag_converter.tensorize.assert_called_once_with(['Training'])

    def test_lower_node_field_preservation(self):
        """Test that all fields are properly preserved during lowering."""
        node_data = self.base_node_data.copy()
        node_data['input'] = True
        node_data['output'] = True
        node_data['tool_name'] = 'calculator'

        node = SZCPNode(**node_data)
        result = node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify flags preserved
        self.assertTrue(result.input)
        self.assertTrue(result.output)
        self.assertEqual(result.tool_callback, self.mock_tool_callback)

    def test_lower_single_node(self):
        """Test lower() method with single node."""
        node = SZCPNode(**self.base_node_data)

        result = node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Should return LZCPNode
        self.assertIsInstance(result, LZCPNode)
        self.assertEqual(result.sequence, 'test_sequence')
        self.assertIsNone(result.next_zone)

    def test_lower_chain_of_nodes(self):
        """Test lower() method with chain of nodes."""
        # Create two nodes
        node1 = SZCPNode(**self.base_node_data)

        node2_data = self.base_node_data.copy()
        node2_data['block'] = 1
        node2_data['text'] = 'Second node text'
        node2 = SZCPNode(**node2_data)

        # Link them
        node1.next_zone = node2

        # Lower the chain
        result_head = node1.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify chain structure is preserved
        self.assertIsInstance(result_head, LZCPNode)
        self.assertEqual(result_head.block, 0)

        self.assertIsNotNone(result_head.next_zone)
        self.assertIsInstance(result_head.next_zone, LZCPNode)
        self.assertEqual(result_head.next_zone.block, 1)

        self.assertIsNone(result_head.next_zone.next_zone)

    def test_lower_with_jump(self):
        """Test lower() preserves jump information."""
        target_node = SZCPNode(**self.base_node_data)

        node_data = self.base_node_data.copy()
        node_data['jump_advance_str'] = '[Jump]'
        node_data['jump_zone'] = target_node

        node = SZCPNode(**node_data)
        result = node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify jump tokens were tokenized and preserved
        self.assertIsNotNone(result.jump_tokens)
        self.assertIsNotNone(result.jump_zone)
        self.assertIsInstance(result.jump_zone, LZCPNode)

        # Verify jump string was tokenized
        self.mock_tokenizer.tokenize.assert_any_call('[Jump]')

    def test_lower_with_tool(self):
        """Test lower() preserves tool information."""
        node_data = self.base_node_data.copy()
        node_data['tool_name'] = 'calculator'

        node = SZCPNode(**node_data)
        result = node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Verify tool callback preserved
        self.assertEqual(result.tool_callback, self.mock_tool_callback)

    def test_lower_missing_tool_in_registry(self):
        """Test lower() fails when tool not found in registry."""
        node_data = self.base_node_data.copy()
        node_data['tool_name'] = 'missing_tool'

        node = SZCPNode(**node_data)

        with self.assertRaises(GraphLoweringError) as context:
            node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        self.assertEqual(context.exception.sequence, 'test_sequence')
        self.assertEqual(context.exception.block, 0)


class TestSZCPNodeGraphTopology(unittest.TestCase):
    """Test graph topology lowering - the mathematical DCG-IO focus."""

    def setUp(self):
        # Create mock tokenizer properly
        self.mock_tokenizer = Mock(spec=TokenizerInterface)
        self.mock_tokenizer.tokenize = Mock()
        self.mock_tokenizer.tokenize.side_effect = lambda text: np.array([hash(text) % 1000], dtype=np.int32)

        # Create mock tag converter properly
        self.mock_tag_converter = Mock(spec=TagConverter)
        self.mock_tag_converter.tensorize = Mock()
        self.mock_tag_converter.tensorize.side_effect = lambda tags: np.array([True] * len(tags), dtype=np.bool_)

        # Create mock tool registry
        self.mock_tool_callback = Mock()
        self.mock_tool_callback.return_value = np.array([99], dtype=np.int32)

        self.tool_registry = {}
        self.base_node_data = {
            'sequence': 'test_sequence',
            'text': 'Test text',
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_advance_str': None,
            'jump_zone': None
        }

    def _create_node(self, block: int, **overrides) -> SZCPNode:
        """Helper to create nodes with block number and optional overrides."""
        node_data = self.base_node_data.copy()
        node_data['block'] = block
        node_data.update(overrides)
        return SZCPNode(**node_data)

    def _test_graph_identity(self, original_szcp: SZCPNode, lowered_lzcp: LZCPNode):
        """
        Helper to verify that a lowered LZCP graph preserves the structure and
        essential properties of the original SZCP graph.

        Args:
            original_szcp: The original SZCP graph head
            lowered_lzcp: The lowered LZCP graph head
        """
        visited_szcp = {}
        visited_lzcp = {}

        def traverse_and_compare(szcp_node, lzcp_node, visited_s, visited_l):
            # Handle terminal cases
            if szcp_node is None and lzcp_node is None:
                return True
            if szcp_node is None or lzcp_node is None:
                return False

            # Cycle detection
            szcp_id = id(szcp_node)
            lzcp_id = id(lzcp_node)

            if szcp_id in visited_s:
                return lzcp_id in visited_l and visited_s[szcp_id] == visited_l[lzcp_id]

            # Mark as visited with same visit number
            visit_count = len(visited_s)
            visited_s[szcp_id] = visit_count
            visited_l[lzcp_id] = visit_count

            # Compare essential node properties
            if not self._nodes_essentially_equal(szcp_node, lzcp_node):
                return False

            # Recursively check graph structure
            return (traverse_and_compare(szcp_node.next_zone, lzcp_node.next_zone, visited_s, visited_l) and
                    traverse_and_compare(szcp_node.jump_zone, lzcp_node.jump_zone, visited_s, visited_l))

        self.assertTrue(traverse_and_compare(original_szcp, lowered_lzcp, visited_szcp, visited_lzcp))

    def _nodes_essentially_equal(self, szcp_node: SZCPNode, lzcp_node: LZCPNode) -> bool:
        """
        Compare essential properties between SZCP and LZCP nodes.

        Args:
            szcp_node: SZCP node with string data
            lzcp_node: LZCP node with tokenized data

        Returns:
            True if nodes represent the same logical zone
        """
        # Basic properties that should be preserved
        return (szcp_node.sequence == lzcp_node.sequence and
                szcp_node.block == lzcp_node.block and
                szcp_node.timeout == lzcp_node.timeout and
                szcp_node.input == lzcp_node.input and
                szcp_node.output == lzcp_node.output and
                # Jump capability should be preserved
                szcp_node.has_jump() == lzcp_node.has_jump() and
                szcp_node.has_tool() == (lzcp_node.tool_callback is not None))

    def test_linear_chain_topology(self):
        """Test: A → B → C → Terminal"""
        # Build the graph
        nodeA = self._create_node(0)
        nodeB = self._create_node(1)
        nodeC = self._create_node(2)
        terminal = self._create_node(3)

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
        nodeA = self._create_node(0)
        nodeB = self._create_node(1)
        nodeC = self._create_node(2)
        nodeD = self._create_node(3)
        terminal = self._create_node(4)

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
        nodeA = self._create_node(0)
        nodeB = self._create_node(1)
        nodeC = self._create_node(2)
        terminal = self._create_node(3)

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
        nodeA = self._create_node(0)
        nodeB = self._create_node(1)
        nodeC = self._create_node(2)
        nodeD = self._create_node(3)
        terminal = self._create_node(4)

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
        nodeA = self._create_node(0)
        nodeB = self._create_node(1)
        nodeC = self._create_node(2)

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


class TestSZCPNodeErrorHandling(unittest.TestCase):
    """Test error handling and exception propagation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tokenizer = Mock(spec=TokenizerInterface)
        self.mock_tokenizer.tokenize = Mock()

        self.mock_tag_converter = Mock(spec=TagConverter)
        self.mock_tag_converter.tensorize = Mock()

        self.tool_registry = {}

        self.base_node_data = {
            'sequence': 'error_sequence',
            'block': 5,
            'text': 'Test text',
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000,
            'input': False,
            'output': False
        }

    def test_lower_error_propagation(self):
        """Test error propagation when lowering fails."""
        # Make tokenizer fail
        self.mock_tokenizer.tokenize.side_effect = RuntimeError("Tokenization failed")

        node = SZCPNode(**self.base_node_data)

        with self.assertRaises(GraphLoweringError) as context:
            node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 5)

    def test_exception_chaining_preserved(self):
        """Test that original exceptions are preserved in the chain."""
        # Make tag converter fail
        original_error = ValueError("Tag conversion failed")
        self.mock_tag_converter.tensorize.side_effect = original_error

        node = SZCPNode(**self.base_node_data)

        with self.assertRaises(GraphLoweringError) as context:
            node.lower(self.mock_tokenizer, self.mock_tag_converter, self.tool_registry)

        # Check that original exception is chained
        self.assertIsInstance(context.exception.__cause__, ValueError)


class TestSZCPSerialization(unittest.TestCase):
    """Test serialization and deserialization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'text': 'Test text',
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000,
            'input': False,
            'output': False
        }

    def _create_node(self, block: int, **overrides) -> SZCPNode:
        """Helper to create nodes with block number and optional overrides."""
        node_data = self.base_node_data.copy()
        node_data['block'] = block
        node_data.update(overrides)
        return SZCPNode(**node_data)

    def test_single_node_round_trip(self):
        """Test serialize→deserialize round-trip for single node."""
        node = SZCPNode(**self.base_node_data)

        # Serialize
        serialized = node.serialize()

        # Deserialize
        deserialized = SZCPNode.deserialize(serialized)

        # Verify basic properties match
        self.assertEqual(deserialized.sequence, node.sequence)
        self.assertEqual(deserialized.block, node.block)
        self.assertEqual(deserialized.text, node.text)
        self.assertEqual(deserialized.zone_advance_str, node.zone_advance_str)
        self.assertEqual(deserialized.tags, node.tags)
        self.assertEqual(deserialized.timeout, node.timeout)
        self.assertEqual(deserialized.input, node.input)
        self.assertEqual(deserialized.output, node.output)
        self.assertIsNone(deserialized.next_zone)
        self.assertIsNone(deserialized.jump_advance_str)
        self.assertIsNone(deserialized.jump_zone)

    def test_node_with_all_fields(self):
        """Test serialization of node with all optional fields populated."""
        target_node = self._create_node(1)

        node_data = self.base_node_data.copy()
        node_data['input'] = True
        node_data['output'] = True
        node_data['tool_name'] = 'calculator'
        node_data['jump_advance_str'] = '[Jump]'
        node_data['jump_zone'] = target_node
        node_data['next_zone'] = target_node

        node = SZCPNode(**node_data)

        # Serialize and deserialize
        serialized = node.serialize()
        deserialized = SZCPNode.deserialize(serialized)

        # Verify all fields preserved
        self.assertTrue(deserialized.input)
        self.assertTrue(deserialized.output)
        self.assertEqual(deserialized.tool_name, 'calculator')
        self.assertEqual(deserialized.jump_advance_str, '[Jump]')
        self.assertIsNotNone(deserialized.jump_zone)
        self.assertIsNotNone(deserialized.next_zone)

    def test_simple_chain_serialization(self):
        """Test serialization of simple A→B→C chain."""
        # Create chain
        nodeA = self._create_node(0)
        nodeB = self._create_node(1)
        nodeC = self._create_node(2)

        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC

        # Serialize and deserialize
        serialized = nodeA.serialize()
        deserialized = SZCPNode.deserialize(serialized)

        # Verify chain structure preserved
        self.assertEqual(deserialized.block, 0)  # A
        self.assertEqual(deserialized.next_zone.block, 1)  # B
        self.assertEqual(deserialized.next_zone.next_zone.block, 2)  # C
        self.assertIsNone(deserialized.next_zone.next_zone.next_zone)

    def test_jump_reference_preservation(self):
        """Test that jump references are preserved correctly."""
        nodeA = self._create_node(0)
        nodeB = self._create_node(1)
        nodeC = self._create_node(2)

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
        nodeA = self._create_node(0)
        nodeB = self._create_node(1)
        nodeC = self._create_node(2)

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
        nodeA = self._create_node(0)
        nodeB = self._create_node(1)
        nodeC = self._create_node(2)
        nodeD = self._create_node(3)

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
        node = SZCPNode(**self.base_node_data)
        serialized = node.serialize()

        # Should be dict mapping int indices to node data
        self.assertIsInstance(serialized, dict)
        self.assertIn(0, serialized)  # Root node always index 0

        # Each entry should have data and links sections
        node_entry = serialized[0]
        self.assertIn('data', node_entry)
        self.assertIn('links', node_entry)

        # Data section should contain node fields
        data = node_entry['data']
        self.assertEqual(data['sequence'], 'test_sequence')
        self.assertEqual(data['block'], 0)
        self.assertEqual(data['text'], 'Test text')

        # Links section should contain reference indices
        links = node_entry['links']
        self.assertIn('next_zone_index', links)
        self.assertIn('jump_zone_index', links)

    def test_index_assignment_correctness(self):
        """Test that indices are assigned correctly."""
        # Create chain A → B → C
        nodeA = self._create_node(0)
        nodeB = self._create_node(1)
        nodeC = self._create_node(2)

        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC

        serialized = nodeA.serialize()

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
        nodeA = self._create_node(0, sequence='seq_A')
        nodeB = self._create_node(1, sequence='seq_B', input=True)
        nodeC = self._create_node(2, sequence='seq_C', output=True, tool_name='calc')

        # A → B → C, A can jump to C, C cycles back to B
        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeA.jump_advance_str = '[Jump]'
        nodeA.jump_zone = nodeC
        nodeC.jump_advance_str = '[Loop]'
        nodeC.jump_zone = nodeB

        # Serialize and deserialize
        serialized = nodeA.serialize()
        deserialized = SZCPNode.deserialize(serialized)

        # Verify graph structure identity
        self._verify_graph_identity(nodeA, deserialized)

    def test_reference_integrity(self):
        """Test that all object references are correctly preserved."""
        # Create diamond pattern: A → B → D, A → C → D
        nodeA = self._create_node(0)
        nodeB = self._create_node(1)
        nodeC = self._create_node(2)
        nodeD = self._create_node(3)

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

    def _verify_graph_identity(self, original: SZCPNode, deserialized: SZCPNode):
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

            # Compare node data
            if not self._nodes_data_equal(orig, deser):
                return False

            # Recursively compare linked nodes
            return (traverse_and_compare(orig.next_zone, deser.next_zone, visited_o, visited_d) and
                    traverse_and_compare(orig.jump_zone, deser.jump_zone, visited_o, visited_d))

        self.assertTrue(traverse_and_compare(original, deserialized, visited_orig, visited_deser))

    def _nodes_data_equal(self, node1: SZCPNode, node2: SZCPNode) -> bool:
        """Helper to compare data equality between two nodes."""
        return (node1.sequence == node2.sequence and
                node1.block == node2.block and
                node1.text == node2.text and
                node1.zone_advance_str == node2.zone_advance_str and
                node1.tags == node2.tags and
                node1.timeout == node2.timeout and
                node1.input == node2.input and
                node1.output == node2.output and
                node1.jump_advance_str == node2.jump_advance_str and
                node1.tool_name == node2.tool_name)


if __name__ == "__main__":
    unittest.main()