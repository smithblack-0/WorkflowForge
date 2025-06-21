"""
Unit tests for RZCPNode class.

Tests cover:
1. Constructor and validation (__post_init__)
2. State query methods (has_jump, is_terminal, etc.)
3. Linked list operations (get_last_node, attach)
4. Basic lowering operations (_lower_node, lower)
5. Graph topology lowering (the mathematical focus)
6. Error handling and exception propagation
"""

import unittest
from unittest.mock import Mock

# Import the modules under test
from src.workflow_forge.zcp.nodes import RZCPNode, SZCPNode, GraphLoweringError, GraphError


class TestRZCPNodeConstruction(unittest.TestCase):
    """Test RZCPNode creation and validation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock sampling callback
        self.mock_sampling_callback = Mock()
        self.mock_sampling_callback.return_value = "resolved text"

        # Basic valid node data
        self.valid_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'zone_advance_str': '[Answer]',
            'tags': ['Training', 'Correct'],
            'timeout': 1000,
            'sampling_callback': self.mock_sampling_callback,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_advance_str': None,
            'jump_zone': None,
            'tool_name': None
        }

    def test_valid_node_creation(self):
        """Test creating a valid RZCPNode with all required fields."""
        node = RZCPNode(**self.valid_node_data)

        self.assertEqual(node.sequence, 'test_sequence')
        self.assertEqual(node.block, 0)
        self.assertEqual(node.zone_advance_str, '[Answer]')
        self.assertEqual(node.tags, ['Training', 'Correct'])
        self.assertEqual(node.timeout, 1000)
        self.assertFalse(node.input)
        self.assertFalse(node.output)
        self.assertIsNone(node.next_zone)
        self.assertIsNone(node.jump_advance_str)
        self.assertIsNone(node.jump_zone)
        self.assertIsNone(node.tool_name)
        self.assertEqual(node.sampling_callback, self.mock_sampling_callback)

    def test_input_output_flags(self):
        """Test nodes with input/output flags set."""
        # Test input node
        input_data = self.valid_node_data.copy()
        input_data['input'] = True
        input_node = RZCPNode(**input_data)
        self.assertTrue(input_node.input)

        # Test output node
        output_data = self.valid_node_data.copy()
        output_data['output'] = True
        output_node = RZCPNode(**output_data)
        self.assertTrue(output_node.output)

    def test_tool_name_assignment(self):
        """Test node with tool name."""
        tool_data = self.valid_node_data.copy()
        tool_data['tool_name'] = 'calculator'
        tool_node = RZCPNode(**tool_data)
        self.assertEqual(tool_node.tool_name, 'calculator')

    def test_post_init_jump_consistency_both_present(self):
        """Test __post_init__ validation when both jump_advance_str and jump_zone are present."""
        target_node = RZCPNode(**self.valid_node_data)

        node_data = self.valid_node_data.copy()
        node_data['jump_advance_str'] = '[Jump]'
        node_data['jump_zone'] = target_node

        # Should not raise exception
        node = RZCPNode(**node_data)
        self.assertIsNotNone(node.jump_advance_str)
        self.assertIsNotNone(node.jump_zone)

    def test_post_init_jump_consistency_mismatch_str_only(self):
        """Test __post_init__ validation fails when only jump_advance_str is present."""
        node_data = self.valid_node_data.copy()
        node_data['jump_advance_str'] = '[Jump]'
        # jump_zone remains None

        with self.assertRaises(GraphError) as context:
            RZCPNode(**node_data)

        self.assertEqual(context.exception.sequence, 'test_sequence')
        self.assertEqual(context.exception.block, 0)

    def test_post_init_jump_consistency_mismatch_zone_only(self):
        """Test __post_init__ validation fails when only jump_zone is present."""
        target_node = RZCPNode(**self.valid_node_data)

        node_data = self.valid_node_data.copy()
        node_data['jump_zone'] = target_node
        # jump_advance_str remains None

        with self.assertRaises(GraphError) as context:
            RZCPNode(**node_data)

        self.assertEqual(context.exception.sequence, 'test_sequence')
        self.assertEqual(context.exception.block, 0)


class TestRZCPNodeStateQueries(unittest.TestCase):
    """Test state query methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_callback = Mock(return_value="resolved text")

        self.base_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000,
            'sampling_callback': self.mock_callback,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_advance_str': None,
            'jump_zone': None
        }

    def test_has_jump_false(self):
        """Test has_jump returns False when no jump capability."""
        node = RZCPNode(**self.base_node_data)
        self.assertFalse(node.has_jump())

    def test_has_jump_true(self):
        """Test has_jump returns True when jump capability exists."""
        target_node = RZCPNode(**self.base_node_data)

        node_data = self.base_node_data.copy()
        node_data['jump_advance_str'] = '[Jump]'
        node_data['jump_zone'] = target_node

        node = RZCPNode(**node_data)
        self.assertTrue(node.has_jump())

    def test_is_terminal_true(self):
        """Test is_terminal returns True for terminal nodes."""
        node = RZCPNode(**self.base_node_data)
        # No next_zone, no jump_zone
        self.assertTrue(node.is_terminal())

    def test_is_terminal_false_has_next(self):
        """Test is_terminal returns False when node has next_zone."""
        next_node = RZCPNode(**self.base_node_data)

        node_data = self.base_node_data.copy()
        node_data['next_zone'] = next_node

        node = RZCPNode(**node_data)
        self.assertFalse(node.is_terminal())

    def test_is_terminal_false_has_jump(self):
        """Test is_terminal returns False when node has jump capability."""
        target_node = RZCPNode(**self.base_node_data)

        node_data = self.base_node_data.copy()
        node_data['jump_advance_str'] = '[Jump]'
        node_data['jump_zone'] = target_node

        node = RZCPNode(**node_data)
        self.assertFalse(node.is_terminal())

    def test_is_input_zone(self):
        """Test is_input_zone reflects input flag."""
        # Test False
        node = RZCPNode(**self.base_node_data)
        self.assertFalse(node.is_input_zone())

        # Test True
        node_data = self.base_node_data.copy()
        node_data['input'] = True
        input_node = RZCPNode(**node_data)
        self.assertTrue(input_node.is_input_zone())

    def test_is_output_zone(self):
        """Test is_output_zone reflects output flag."""
        # Test False
        node = RZCPNode(**self.base_node_data)
        self.assertFalse(node.is_output_zone())

        # Test True
        node_data = self.base_node_data.copy()
        node_data['output'] = True
        output_node = RZCPNode(**node_data)
        self.assertTrue(output_node.is_output_zone())

    def test_has_tool_false(self):
        """Test has_tool returns False when no tool name."""
        node = RZCPNode(**self.base_node_data)
        self.assertFalse(node.has_tool())

    def test_has_tool_true(self):
        """Test has_tool returns True when tool name exists."""
        node_data = self.base_node_data.copy()
        node_data['tool_name'] = 'calculator'

        node = RZCPNode(**node_data)
        self.assertTrue(node.has_tool())


class TestRZCPNodeLinkedList(unittest.TestCase):
    """Test linked list operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_callback = Mock(return_value="resolved text")

        self.base_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000,
            'sampling_callback': self.mock_callback,
            'input': False,
            'output': False
        }

    def test_get_last_node_single(self):
        """Test get_last_node with single node returns self."""
        node = RZCPNode(**self.base_node_data)

        last_node = node.get_last_node()
        self.assertEqual(last_node, node)

    def test_get_last_node_chain(self):
        """Test get_last_node traverses to end of chain."""
        # Create three nodes
        node1 = RZCPNode(**self.base_node_data)

        node2_data = self.base_node_data.copy()
        node2_data['block'] = 1
        node2 = RZCPNode(**node2_data)

        node3_data = self.base_node_data.copy()
        node3_data['block'] = 2
        node3 = RZCPNode(**node3_data)

        # Link them: node1 -> node2 -> node3
        node1.next_zone = node2
        node2.next_zone = node3

        # get_last_node should return node3 from any starting point
        self.assertEqual(node1.get_last_node(), node3)
        self.assertEqual(node2.get_last_node(), node3)
        self.assertEqual(node3.get_last_node(), node3)

    def test_attach_single_source(self):
        """Test attach method connects source node to this node."""
        target = RZCPNode(**self.base_node_data)

        source_data = self.base_node_data.copy()
        source_data['block'] = 1
        source = RZCPNode(**source_data)

        # Attach source to target
        result = target.attach([source])

        # Should return self
        self.assertEqual(result, target)

        # Source should now point to target
        self.assertEqual(source.next_zone, target)

    def test_attach_multiple_sources(self):
        """Test attach method connects multiple source nodes."""
        target = RZCPNode(**self.base_node_data)

        sources = []
        for i in range(3):
            source_data = self.base_node_data.copy()
            source_data['block'] = i + 1
            sources.append(RZCPNode(**source_data))

        # Attach all sources to target
        result = target.attach(sources)

        # Should return self
        self.assertEqual(result, target)

        # All sources should point to target
        for source in sources:
            self.assertEqual(source.next_zone, target)


class TestRZCPNodeBasicLowering(unittest.TestCase):
    """Test basic lowering operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_callback = Mock(return_value="resolved prompt text")

        self.base_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'zone_advance_str': '[Answer]',
            'tags': ['Training', 'Correct'],
            'timeout': 1000,
            'sampling_callback': self.mock_callback,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_advance_str': None,
            'jump_zone': None,
            'tool_name': None
        }

    def test_lower_node_success(self):
        """Test _lower_node creates valid SZCPNode."""
        node = RZCPNode(**self.base_node_data)

        result = node.lower()

        # Verify result type and basic properties
        self.assertIsInstance(result, SZCPNode)
        self.assertEqual(result.sequence, 'test_sequence')
        self.assertEqual(result.block, 0)
        self.assertEqual(result.zone_advance_str, '[Answer]')
        self.assertEqual(result.tags, ['Training', 'Correct'])
        self.assertEqual(result.timeout, 1000)
        self.assertFalse(result.input)
        self.assertFalse(result.output)
        self.assertIsNone(result.next_zone)
        self.assertIsNone(result.jump_advance_str)
        self.assertIsNone(result.jump_zone)
        self.assertIsNone(result.tool_name)

        # Verify resolved text from sampling callback
        self.assertEqual(result.text, "resolved prompt text")

        # Verify sampling callback was called
        self.mock_callback.assert_called_once()

    def test_lower_node_with_jump(self):
        """Test _lower_node preserves jump information."""
        target_node = RZCPNode(**self.base_node_data)

        node_data = self.base_node_data.copy()
        node_data['jump_advance_str'] = '[Jump]'
        node_data['jump_zone'] = target_node

        node = RZCPNode(**node_data)
        result = node.lower()

        # Verify jump string preserved
        print(type(result.jump_advance_str))
        self.assertEqual(result.jump_advance_str, '[Jump]')
        # Verify jump_zone was lowered and connected
        self.assertIsNotNone(result.jump_zone)
        self.assertIsInstance(result.jump_zone, SZCPNode)

    def test_lower_node_with_tool(self):
        """Test _lower_node preserves tool information."""
        node_data = self.base_node_data.copy()
        node_data['tool_name'] = 'calculator'

        node = RZCPNode(**node_data)
        result = node.lower()

        # Verify tool name preserved
        self.assertEqual(result.tool_name, 'calculator')

    def test_lower_single_node(self):
        """Test lower() method with single node."""
        node = RZCPNode(**self.base_node_data)

        result = node.lower()

        # Should return SZCPNode
        self.assertIsInstance(result, SZCPNode)
        self.assertEqual(result.sequence, 'test_sequence')
        self.assertIsNone(result.next_zone)

    def test_lower_chain_of_nodes(self):
        """Test lower() method with chain of nodes."""
        # Create two nodes with different mock callbacks
        mock_callback1 = Mock(return_value="first node text")
        mock_callback2 = Mock(return_value="second node text")

        node1 = RZCPNode(**self.base_node_data)
        node1.sampling_callback = mock_callback1

        node2_data = self.base_node_data.copy()
        node2_data['block'] = 1
        node2_data['sampling_callback'] = mock_callback2
        node2 = RZCPNode(**node2_data)

        # Link them
        node1.next_zone = node2

        # Lower the chain
        result_head = node1.lower()

        # Verify chain structure is preserved
        self.assertIsInstance(result_head, SZCPNode)
        self.assertEqual(result_head.block, 0)
        self.assertEqual(result_head.text, "first node text")

        self.assertIsNotNone(result_head.next_zone)
        self.assertIsInstance(result_head.next_zone, SZCPNode)
        self.assertEqual(result_head.next_zone.block, 1)
        self.assertEqual(result_head.next_zone.text, "second node text")

        self.assertIsNone(result_head.next_zone.next_zone)

        # Verify both callbacks were called
        mock_callback1.assert_called_once()
        mock_callback2.assert_called_once()


class TestRZCPNodeGraphTopology(unittest.TestCase):
    """Test graph topology lowering - the mathematical focus."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_callback = Mock(return_value="resolved text")

        self.base_node_data = {
            'sequence': 'test_sequence',
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000,
            'sampling_callback': self.mock_callback,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_advance_str': None,
            'jump_zone': None
        }

    def _create_node(self, block: int, **overrides) -> RZCPNode:
        """Helper to create nodes with block number and optional overrides."""
        node_data = self.base_node_data.copy()
        node_data['block'] = block
        node_data.update(overrides)
        return RZCPNode(**node_data)

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
        result = nodeA.lower()

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
        result = nodeA.lower()

        # Verify linear structure preserved
        self.assertEqual(result.block, 0)  # A
        self.assertEqual(result.next_zone.block, 1)  # B
        self.assertEqual(result.next_zone.next_zone.block, 2)  # C
        self.assertEqual(result.next_zone.next_zone.next_zone.block, 3)  # D

        # Verify jump preserved
        nodeB_lowered = result.next_zone
        self.assertIsNotNone(nodeB_lowered.jump_advance_str)
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
        result = nodeA.lower()

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
        result = nodeA.lower()

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
        result = nodeA.lower()

        # Verify the cycle is preserved in lowered graph
        self.assertEqual(result.block, 0)  # A
        self.assertEqual(result.next_zone.block, 1)  # B
        self.assertEqual(result.next_zone.next_zone.block, 2)  # C
        self.assertEqual(result.next_zone.next_zone.next_zone.block, 1)  # Back to B

        # Verify they're the same instance (proper cycle detection)
        nodeB_lowered = result.next_zone
        nodeB_from_cycle = result.next_zone.next_zone.next_zone
        self.assertEqual(nodeB_lowered, nodeB_from_cycle)


class TestRZCPNodeErrorHandling(unittest.TestCase):
    """Test error handling and exception propagation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_callback = Mock(return_value="resolved text")

        self.base_node_data = {
            'sequence': 'error_sequence',
            'block': 5,
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000,
            'sampling_callback': self.mock_callback,
            'input': False,
            'output': False
        }

    def test_post_init_validation_error_context(self):
        """Test that __post_init__ validation includes proper context."""
        node_data = self.base_node_data.copy()
        node_data['jump_advance_str'] = '[Jump]'
        # Missing jump_zone

        with self.assertRaises(GraphError) as context:
            RZCPNode(**node_data)

        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 5)

    def test_sampling_callback_failure(self):
        """Test error handling when sampling callback fails."""
        failing_callback = Mock(side_effect=RuntimeError("Sampling failed"))

        node_data = self.base_node_data.copy()
        node_data['sampling_callback'] = failing_callback

        node = RZCPNode(**node_data)

        with self.assertRaises(GraphLoweringError) as context:
            node._lower_node()

        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 5)
        self.assertIsInstance(context.exception.__cause__, RuntimeError)

    def test_lower_chain_error_propagation(self):
        """Test error propagation when lowering chains."""
        # Create chain where second node will fail
        node1 = RZCPNode(**self.base_node_data)

        failing_callback = Mock(side_effect=RuntimeError("Node2 failed"))
        node2_data = self.base_node_data.copy()
        node2_data['block'] = 6
        node2_data['sampling_callback'] = failing_callback
        node2 = RZCPNode(**node2_data)

        node1.next_zone = node2

        with self.assertRaises(GraphLoweringError) as context:
            node1.lower()

        # Error should reference the failing node's context
        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 6)  # Should be node2's block


if __name__ == "__main__":
    unittest.main()