"""
Unit tests for RZCPNode class.

Tests cover:
1. Constructor and validation (__post_init__)
2. State query methods (has_jump, is_terminal, etc.)
3. Linked list operations (get_last_node, attach)
4. Basic lowering operations (_lower_node, lower)
5. Graph topology lowering (the mathematical focus)
6. Error handling and exception propagation
7. Three-tier resource system integration
"""

import unittest
from unittest.mock import Mock
from typing import Dict, Any, Optional

# Import the modules under test
from src.workflow_forge.zcp.nodes import RZCPNode, SZCPNode, GraphLoweringError, GraphError
from src.workflow_forge.resources import AbstractResource


class BaseRZCPNodeTest(unittest.TestCase):
    """Base test class with common setup and helper methods."""

    def setUp(self):
        """Set up common test fixtures."""
        # Create mock sampling callback that accepts dynamic resources
        self.mock_sampling_callback = Mock()
        self.mock_sampling_callback.return_value = "resolved text"

    def get_valid_node_data(self, **overrides) -> Dict[str, Any]:
        """
        Return valid node data for testing, with optional field overrides.

        Args:
            **overrides: Fields to override in the base node data

        Returns:
            Dictionary of valid RZCPNode constructor arguments
        """
        base_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'zone_advance_str': '[Answer]',
            'escape_strs': ('[Escape]', '[EndEscape]'),
            'tags': ['Training'],
            'timeout': 1000,
            'sampling_callback': self.mock_sampling_callback,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_advance_str': None,
            'jump_zone': None,
            'tool_name': None
        }
        base_data.update(overrides)
        return base_data

    def create_node(self, **overrides) -> RZCPNode:
        """
        Create an RZCPNode with valid data and optional overrides.

        Args:
            **overrides: Fields to override in the base node data

        Returns:
            Configured RZCPNode instance
        """
        return RZCPNode(**self.get_valid_node_data(**overrides))

    def create_node_chain(self, length: int, **base_overrides) -> RZCPNode:
        """
        Create a chain of linked RZCPNodes.

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
            # Create separate mock for each node to avoid shared state
            mock_callback = Mock(return_value=f"resolved text {i}")
            node_overrides = base_overrides.copy()
            node_overrides.update({'block': i, 'sampling_callback': mock_callback})
            node = self.create_node(**node_overrides)
            nodes.append(node)

        # Link them
        for i in range(length - 1):
            nodes[i].next_zone = nodes[i + 1]

        return nodes[0]

    def create_jump_node(self, target_node: RZCPNode, jump_str: str = '[Jump]', **overrides) -> RZCPNode:
        """
        Create an RZCPNode with jump capability.

        Args:
            target_node: The node to jump to
            jump_str: The jump advance string
            **overrides: Additional field overrides

        Returns:
            RZCPNode with jump capability configured
        """
        jump_overrides = {
            'jump_advance_str': jump_str,
            'jump_zone': target_node
        }
        jump_overrides.update(overrides)
        return self.create_node(**jump_overrides)

    def assert_szcp_node_properties(self, szcp_node: SZCPNode, expected_sequence: str,
                                  expected_block: int, expected_timeout: int = 1000):
        """
        Assert common properties of an SZCPNode.

        Args:
            szcp_node: The SZCPNode to validate
            expected_sequence: Expected sequence name
            expected_block: Expected block number
            expected_timeout: Expected timeout value
        """
        self.assertIsInstance(szcp_node, SZCPNode)
        self.assertEqual(szcp_node.sequence, expected_sequence)
        self.assertEqual(szcp_node.block, expected_block)
        self.assertEqual(szcp_node.timeout, expected_timeout)
        self.assertEqual(szcp_node.escape_strs, ('[Escape]', '[EndEscape]'))

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


class TestRZCPNodeConstruction(BaseRZCPNodeTest):
    """Test RZCPNode creation and validation."""

    def test_valid_node_creation(self):
        """Test creating a valid RZCPNode with all required fields."""
        node_data = self.get_valid_node_data()
        node = RZCPNode(**node_data)

        self.assertEqual(node.sequence, 'test_sequence')
        self.assertEqual(node.block, 0)
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
        self.assertEqual(node.sampling_callback, self.mock_sampling_callback)

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

    def test_post_init_jump_consistency_both_present(self):
        """Test __post_init__ validation when both jump_advance_str and jump_zone are present."""
        target_node = self.create_node()
        jump_node = self.create_jump_node(target_node)

        # Should not raise exception
        self.assertIsNotNone(jump_node.jump_advance_str)
        self.assertIsNotNone(jump_node.jump_zone)

    def test_post_init_jump_consistency_mismatch_str_only(self):
        """Test __post_init__ validation fails when only jump_advance_str is present."""
        with self.assertRaises(GraphError) as context:
            self.create_node(jump_advance_str='[Jump]')  # jump_zone remains None

        self.assert_graph_error_context(context, 'test_sequence', 0)

    def test_post_init_jump_consistency_mismatch_zone_only(self):
        """Test __post_init__ validation fails when only jump_zone is present."""
        target_node = self.create_node()

        with self.assertRaises(GraphError) as context:
            self.create_node(jump_zone=target_node)  # jump_advance_str remains None

        self.assert_graph_error_context(context, 'test_sequence', 0)


class TestRZCPNodeStateQueries(BaseRZCPNodeTest):
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


class TestRZCPNodeLinkedList(BaseRZCPNodeTest):
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

    def test_attach_single_source(self):
        """Test attach method connects source node to this node."""
        target = self.create_node()
        source = self.create_node(block=1)

        # Attach source to target
        result = target.attach([source])

        # Should return self
        self.assertEqual(result, target)

        # Source should now point to target
        self.assertEqual(source.next_zone, target)

    def test_attach_multiple_sources(self):
        """Test attach method connects multiple source nodes."""
        target = self.create_node()

        sources = []
        for i in range(3):
            sources.append(self.create_node(block=i + 1))

        # Attach all sources to target
        result = target.attach(sources)

        # Should return self
        self.assertEqual(result, target)

        # All sources should point to target
        for source in sources:
            self.assertEqual(source.next_zone, target)


class TestRZCPNodeBasicLowering(BaseRZCPNodeTest):
    """Test basic lowering operations."""

    def test_lower_node_success(self):
        """Test lower() creates valid SZCPNode."""
        node = self.create_node(tags=['Training', 'Correct'])

        result = node.lower(resources={})

        # Verify result using helper assertion
        self.assert_szcp_node_properties(result, 'test_sequence', 0)

        # Verify specific properties
        self.assertEqual(result.zone_advance_str, '[Answer]')
        self.assertEqual(result.tags, ['Training', 'Correct'])
        self.assertFalse(result.input)
        self.assertFalse(result.output)
        self.assertIsNone(result.next_zone)
        self.assertIsNone(result.jump_advance_str)
        self.assertIsNone(result.jump_zone)
        self.assertIsNone(result.tool_name)

        # Verify resolved text from sampling callback
        self.assertEqual(result.text, "resolved text")

        # Verify sampling callback was called with empty resources
        self.mock_sampling_callback.assert_called_once_with({})

    def test_lower_node_with_dynamic_resources(self):
        """Test lower() passes resources to sampling callback."""
        node = self.create_node()

        # Create resources
        dynamic_mock = Mock(spec=AbstractResource)
        resources = {'dynamic_res': dynamic_mock}

        result = node.lower(resources=resources)

        # Verify sampling callback was called with resources
        self.mock_sampling_callback.assert_called_once_with(resources)

        # Verify result
        self.assertEqual(result.text, "resolved text")

    def test_lower_node_with_jump(self):
        """Test lower() preserves jump information."""
        target_node = self.create_node()
        jump_node = self.create_jump_node(target_node)

        result = jump_node.lower(resources={})

        # Verify jump string preserved
        self.assertEqual(result.jump_advance_str, '[Jump]')
        # Verify jump_zone was lowered and connected
        self.assertIsNotNone(result.jump_zone)
        self.assertIsInstance(result.jump_zone, SZCPNode)

    def test_lower_node_with_tool(self):
        """Test lower() preserves tool information."""
        tool_node = self.create_node(tool_name='calculator')
        result = tool_node.lower(resources={})

        # Verify tool name preserved
        self.assertEqual(result.tool_name, 'calculator')

    def test_lower_single_node(self):
        """Test lower() method with single node."""
        node = self.create_node()
        result = node.lower(resources={})

        # Should return SZCPNode
        self.assertIsInstance(result, SZCPNode)
        self.assertEqual(result.sequence, 'test_sequence')
        self.assertIsNone(result.next_zone)

    def test_lower_chain_of_nodes(self):
        """Test lower() method with chain of nodes."""
        # Create two-node chain with different callbacks
        head_node = self.create_node_chain(2)

        # Lower the chain
        result_head = head_node.lower(resources={})

        # Verify chain structure is preserved
        self.assert_szcp_node_properties(result_head, 'test_sequence', 0)

        self.assertIsNotNone(result_head.next_zone)
        self.assert_szcp_node_properties(result_head.next_zone, 'test_sequence', 1)

        self.assertIsNone(result_head.next_zone.next_zone)

        # Verify resolved text from different callbacks
        self.assertEqual(result_head.text, "resolved text 0")
        self.assertEqual(result_head.next_zone.text, "resolved text 1")



class TestRZCPNodeResourceSystem(BaseRZCPNodeTest):
    """Test three-tier resource system integration."""

    def test_lower_with_multiple_resources(self):
        """Test lower() with multiple resources."""
        node = self.create_node()

        # Create multiple resources
        resources = {
            'resource1': Mock(spec=AbstractResource),
            'resource2': Mock(spec=AbstractResource),
            'resource3': Mock(spec=AbstractResource)
        }

        result = node.lower(resources=resources)

        # Verify all resources were passed
        self.mock_sampling_callback.assert_called_once_with(resources)

    def test_lower_chain_with_different_resources(self):
        """Test lowering chain where each node gets same resources."""
        # Create two-node chain
        head_node = self.create_node_chain(2)

        resources = {'shared_resource': Mock(spec=AbstractResource)}

        # Lower the chain - all nodes should get same resources
        result = head_node.lower(resources=resources)

        # Both callbacks should have been called with same resources
        head_callback = head_node.sampling_callback
        second_callback = head_node.next_zone.sampling_callback

        head_callback.assert_called_once_with(resources)
        second_callback.assert_called_once_with(resources)

    def test_sampling_callback_signature_compatibility(self):
        """Test that sampling callback signature is correct for three-tier system."""
        node = self.create_node()

        # Create a more realistic callback mock
        realistic_callback = Mock()
        realistic_callback.return_value = "realistic result"
        node.sampling_callback = realistic_callback

        # Test with empty resources
        result1 = node.lower(resources={})
        realistic_callback.assert_called_with({})

        # Reset mock and test with actual resources
        realistic_callback.reset_mock()
        resources = {'test': Mock(spec=AbstractResource)}
        result2 = node.lower(resources=resources)
        realistic_callback.assert_called_with(resources)


class TestRZCPNodeGraphTopology(BaseRZCPNodeTest):
    """Test graph topology lowering - the mathematical focus."""

    def _create_topology_node(self, block: int, **overrides) -> RZCPNode:
        """Helper to create nodes for topology tests with unique callbacks."""
        mock_callback = Mock(return_value=f"text_{block}")
        overrides.update({'block': block, 'sampling_callback': mock_callback})
        return self.create_node(**overrides)

    def test_linear_chain_topology(self):
        """Test: A → B → C → Terminal"""
        # Build the graph
        nodeA = self._create_topology_node(0)
        nodeB = self._create_topology_node(1)
        nodeC = self._create_topology_node(2)
        terminal = self._create_topology_node(3)

        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeC.next_zone = terminal

        # Lower from head
        result = nodeA.lower(resources={})

        # Verify structure preservation
        self.assertEqual(result.block, 0)  # A
        self.assertEqual(result.next_zone.block, 1)  # B
        self.assertEqual(result.next_zone.next_zone.block, 2)  # C
        self.assertEqual(result.next_zone.next_zone.next_zone.block, 3)  # Terminal
        self.assertIsNone(result.next_zone.next_zone.next_zone.next_zone)

    def test_simple_branch_topology(self):
        """Test: A → B (jump to D), A → B → C → D → Terminal"""
        # Build the graph
        nodeA = self._create_topology_node(0)
        nodeB = self._create_topology_node(1)
        nodeC = self._create_topology_node(2)
        nodeD = self._create_topology_node(3)
        terminal = self._create_topology_node(4)

        # Linear path: A → B → C → D → Terminal
        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeC.next_zone = nodeD
        nodeD.next_zone = terminal

        # Jump path: B can jump to D
        nodeB.jump_advance_str = '[Jump]'
        nodeB.jump_zone = nodeD

        # Lower from head
        result = nodeA.lower(resources={})

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
        nodeA = self._create_topology_node(0)
        nodeB = self._create_topology_node(1)
        nodeC = self._create_topology_node(2)
        terminal = self._create_topology_node(3)

        # Linear path: A → B → C → Terminal
        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeC.next_zone = terminal

        # Loop: C can jump back to B
        nodeC.jump_advance_str = '[Jump]'
        nodeC.jump_zone = nodeB

        # Lower from head (tests cycle handling)
        result = nodeA.lower(resources={})

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
        nodeA = self._create_topology_node(0)
        nodeB = self._create_topology_node(1)
        nodeC = self._create_topology_node(2)
        nodeD = self._create_topology_node(3)
        terminal = self._create_topology_node(4)

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
        result = nodeA.lower(resources={})

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
        nodeA = self._create_topology_node(0)
        nodeB = self._create_topology_node(1)
        nodeC = self._create_topology_node(2)

        # Create the cycle
        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeC.next_zone = nodeB  # Cycle back to B

        # This should complete without infinite recursion
        result = nodeA.lower(resources={})

        # Verify the cycle is preserved in lowered graph
        self.assertEqual(result.block, 0)  # A
        self.assertEqual(result.next_zone.block, 1)  # B
        self.assertEqual(result.next_zone.next_zone.block, 2)  # C
        self.assertEqual(result.next_zone.next_zone.next_zone.block, 1)  # Back to B

        # Verify they're the same instance (proper cycle detection)
        nodeB_lowered = result.next_zone
        nodeB_from_cycle = result.next_zone.next_zone.next_zone
        self.assertEqual(nodeB_lowered, nodeB_from_cycle)

    def test_topology_with_dynamic_resources(self):
        """Test complex topology with resources passed through."""
        # Create a complex graph: A → B (jump to D), C → D → Terminal
        nodeA = self._create_topology_node(0)
        nodeB = self._create_topology_node(1)
        nodeC = self._create_topology_node(2)
        nodeD = self._create_topology_node(3)

        # Build topology
        nodeA.next_zone = nodeB
        nodeB.jump_advance_str = '[Jump]'
        nodeB.jump_zone = nodeD
        nodeC.next_zone = nodeD

        # Create resources
        resources = {
            'shared_resource': Mock(spec=AbstractResource),
            'topology_resource': Mock(spec=AbstractResource)
        }

        # Lower from head
        result = nodeA.lower(resources=resources)

        # Verify all callbacks were called with same resources
        nodeA.sampling_callback.assert_called_with(resources)
        nodeB.sampling_callback.assert_called_with(resources)
        nodeD.sampling_callback.assert_called_with(resources)


class TestRZCPNodeErrorHandling(BaseRZCPNodeTest):
    """Test error handling and exception propagation."""

    def test_post_init_validation_error_context(self):
        """Test that __post_init__ validation includes proper context."""
        with self.assertRaises(GraphError) as context:
            self.create_node(
                sequence='error_sequence',
                block=5,
                jump_advance_str='[Jump]'  # Missing jump_zone
            )

        self.assert_graph_error_context(context, "error_sequence", 5)

    def test_sampling_callback_failure(self):
        """Test error handling when sampling callback fails."""
        failing_callback = Mock(side_effect=RuntimeError("Sampling failed"))
        node = self.create_node(
            sequence='error_sequence',
            block=5,
            sampling_callback=failing_callback
        )

        with self.assertRaises(GraphLoweringError) as context:
            node.lower(resources={})

        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 5)
        self.assertIsInstance(context.exception.__cause__, RuntimeError)

    def test_lower_chain_error_propagation(self):
        """Test error propagation when lowering chains."""
        # Create chain where second node will fail
        failing_callback = Mock(side_effect=RuntimeError("Node2 failed"))

        node1 = self.create_node(sequence='error_sequence', block=5)
        node2 = self.create_node(
            sequence='error_sequence',
            block=6,
            sampling_callback=failing_callback
        )

        node1.next_zone = node2

        with self.assertRaises(GraphLoweringError) as context:
            node1.lower(resources={})

        # Error should reference the failing node's context
        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 6)  # Should be node2's block

    def test_sampling_callback_error_with_resources(self):
        """Test error handling when sampling callback fails with resources."""
        failing_callback = Mock(side_effect=ValueError("Resource error"))
        node = self.create_node(
            sequence='resource_error_sequence',
            block=7,
            sampling_callback=failing_callback
        )

        resources = {'failing_resource': Mock(spec=AbstractResource)}

        with self.assertRaises(GraphLoweringError) as context:
            node.lower(resources=resources)

        # Verify error context
        self.assertEqual(context.exception.sequence, "resource_error_sequence")
        self.assertEqual(context.exception.block, 7)
        self.assertIsInstance(context.exception.__cause__, ValueError)

        # Verify callback was called with resources before failing
        failing_callback.assert_called_once_with(resources)


if __name__ == "__main__":
    unittest.main()