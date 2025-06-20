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
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Optional

# Import the modules under test
from src.workflow_forge.zcp.nodes import RZCPNode, LZCPNode, GraphLoweringError, GraphError
from src.workflow_forge.resources import AbstractResource
from src.workflow_forge.tokenizer_interface import TokenizerInterface
from src.workflow_forge.flow_control.tag_converter import TagConverter


class TestRZCPNodeConstruction(unittest.TestCase):
    """Test RZCPNode creation and validation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock sampling callback
        self.mock_sampling_callback = Mock()
        self.mock_sampling_callback.return_value = np.array([1, 2, 3])

        # Basic valid node data
        self.valid_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'zone_advance_tokens': np.array([10], dtype=np.int32),
            'tags': np.array([True, False], dtype=np.bool_),
            'timeout': 1000,
            'sampling_callback': self.mock_sampling_callback,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_tokens': None,
            'jump_zone': None,
            'tool_callback': None
        }

    def test_valid_node_creation(self):
        """Test creating a valid RZCPNode with all required fields."""
        node = RZCPNode(**self.valid_node_data)

        self.assertEqual(node.sequence, 'test_sequence')
        self.assertEqual(node.block, 0)
        self.assertEqual(node.timeout, 1000)
        self.assertFalse(node.input)
        self.assertFalse(node.output)
        self.assertIsNone(node.next_zone)
        self.assertIsNone(node.jump_tokens)
        self.assertIsNone(node.jump_zone)
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

    def test_post_init_jump_consistency_both_present(self):
        """Test __post_init__ validation when both jump_tokens and jump_zone are present."""
        target_node = RZCPNode(**self.valid_node_data)

        node_data = self.valid_node_data.copy()
        node_data['jump_tokens'] = np.array([20], dtype=np.int32)
        node_data['jump_zone'] = target_node

        # Should not raise exception
        node = RZCPNode(**node_data)
        self.assertIsNotNone(node.jump_tokens)
        self.assertIsNotNone(node.jump_zone)

    def test_post_init_jump_consistency_mismatch_tokens_only(self):
        """Test __post_init__ validation fails when only jump_tokens is present."""
        node_data = self.valid_node_data.copy()
        node_data['jump_tokens'] = np.array([20], dtype=np.int32)
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
        # jump_tokens remains None

        with self.assertRaises(GraphError) as context:
            RZCPNode(**node_data)

        self.assertEqual(context.exception.sequence, 'test_sequence')
        self.assertEqual(context.exception.block, 0)


class TestRZCPNodeStateQueries(unittest.TestCase):
    """Test state query methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_callback = Mock(return_value=np.array([1, 2, 3]))

        self.base_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'zone_advance_tokens': np.array([10], dtype=np.int32),
            'tags': np.array([True], dtype=np.bool_),
            'timeout': 1000,
            'sampling_callback': self.mock_callback,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_tokens': None,
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
        node_data['jump_tokens'] = np.array([20], dtype=np.int32)
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
        node_data['jump_tokens'] = np.array([20], dtype=np.int32)
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


class TestRZCPNodeLinkedList(unittest.TestCase):
    """Test linked list operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_callback = Mock(return_value=np.array([1, 2, 3]))

        self.base_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'zone_advance_tokens': np.array([10], dtype=np.int32),
            'tags': np.array([True], dtype=np.bool_),
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
        self.mock_callback = Mock(return_value=np.array([1, 2, 3]))

        self.base_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'zone_advance_tokens': np.array([10], dtype=np.int32),
            'tags': np.array([True, False], dtype=np.bool_),
            'timeout': 1000,
            'sampling_callback': self.mock_callback,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_tokens': None,
            'jump_zone': None,
            'tool_callback': None
        }

    def test_lower_node_success(self):
        """Test _lower_node creates valid LZCPNode."""
        node = RZCPNode(**self.base_node_data)

        result = node._lower_node()

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

        # Verify tokens from sampling callback
        np.testing.assert_array_equal(result.tokens, np.array([1, 2, 3]))

        # Verify other arrays copied correctly
        np.testing.assert_array_equal(result.zone_advance_tokens, np.array([10]))
        np.testing.assert_array_equal(result.tags, np.array([True, False]))

        # Verify sampling callback was called
        self.mock_callback.assert_called_once()

    def test_lower_node_with_jump(self):
        """Test lower() preserves jump information."""
        target_node = RZCPNode(**self.base_node_data)

        node_data = self.base_node_data.copy()
        node_data['jump_tokens'] = np.array([20], dtype=np.int32)
        node_data['jump_zone'] = target_node

        node = RZCPNode(**node_data)
        result = node.lower()

        # Verify jump tokens preserved
        np.testing.assert_array_equal(result.jump_tokens, np.array([20]))
        # Verify jump_zone was lowered and connected
        self.assertIsNotNone(result.jump_zone)
        self.assertIsInstance(result.jump_zone, LZCPNode)

    def test_lower_single_node(self):
        """Test lower() method with single node."""
        node = RZCPNode(**self.base_node_data)

        result = node.lower()

        # Should return LZCPNode
        self.assertIsInstance(result, LZCPNode)
        self.assertEqual(result.sequence, 'test_sequence')
        self.assertIsNone(result.next_zone)

    def test_lower_chain_of_nodes(self):
        """Test lower() method with chain of nodes."""
        # Create two nodes
        node1 = RZCPNode(**self.base_node_data)

        node2_data = self.base_node_data.copy()
        node2_data['block'] = 1
        node2 = RZCPNode(**node2_data)

        # Link them
        node1.next_zone = node2

        # Lower the chain
        result_head = node1.lower()

        # Verify chain structure is preserved
        self.assertIsInstance(result_head, LZCPNode)
        self.assertEqual(result_head.block, 0)

        self.assertIsNotNone(result_head.next_zone)
        self.assertIsInstance(result_head.next_zone, LZCPNode)
        self.assertEqual(result_head.next_zone.block, 1)

        self.assertIsNone(result_head.next_zone.next_zone)


class TestRZCPNodeGraphTopology(unittest.TestCase):
    """Test graph topology lowering - the mathematical focus."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_callback = Mock(return_value=np.array([1, 2, 3]))

        self.base_node_data = {
            'sequence': 'test_sequence',
            'zone_advance_tokens': np.array([10], dtype=np.int32),
            'tags': np.array([True], dtype=np.bool_),
            'timeout': 1000,
            'sampling_callback': self.mock_callback,
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_tokens': None,
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
        nodeB.jump_tokens = np.array([20], dtype=np.int32)
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
        nodeC.jump_tokens = np.array([30], dtype=np.int32)
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
        nodeA.jump_tokens = np.array([15], dtype=np.int32)
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

    def test_complex_branch_with_convergence(self):
        """Test: A → B (jump to E), A → B → C → D → E → Terminal"""
        # Build the graph
        nodeA = self._create_node(0)
        nodeB = self._create_node(1)
        nodeC = self._create_node(2)
        nodeD = self._create_node(3)
        nodeE = self._create_node(4)
        terminal = self._create_node(5)

        # Linear path: A → B → C → D → E → Terminal
        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeC.next_zone = nodeD
        nodeD.next_zone = nodeE
        nodeE.next_zone = terminal

        # Jump path: B can jump directly to E (skipping C, D)
        nodeB.jump_tokens = np.array([25], dtype=np.int32)
        nodeB.jump_zone = nodeE

        # Lower from head
        result = nodeA.lower()

        # Verify linear path
        current = result
        blocks = []
        while current is not None:
            blocks.append(current.block)
            current = current.next_zone
        self.assertEqual(blocks, [0, 1, 2, 3, 4, 5])

        # Verify jump bypasses middle nodes
        nodeB_lowered = result.next_zone
        self.assertEqual(nodeB_lowered.jump_zone.block, 4)  # Points to E

    def test_nested_loop_structure(self):
        """Test: A → B → C (jump to B) → D (jump to B) → Terminal"""
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

        # Multiple jumps back to B
        nodeC.jump_tokens = np.array([30], dtype=np.int32)
        nodeC.jump_zone = nodeB

        nodeD.jump_tokens = np.array([40], dtype=np.int32)
        nodeD.jump_zone = nodeB

        # Lower from head
        result = nodeA.lower()

        # Verify both jumps point to same B
        nodeC_lowered = result.next_zone.next_zone
        nodeD_lowered = result.next_zone.next_zone.next_zone

        self.assertEqual(nodeC_lowered.jump_zone.block, 1)  # C → B
        self.assertEqual(nodeD_lowered.jump_zone.block, 1)  # D → B

        # Verify they point to the same instance (convergence)
        nodeB_lowered = result.next_zone
        self.assertEqual(nodeC_lowered.jump_zone, nodeB_lowered)
        self.assertEqual(nodeD_lowered.jump_zone, nodeB_lowered)

    def test_multiple_jump_targets(self):
        """Test: A → B (jump to D), A → B → C (jump to E), A → B → C → D → E → Terminal"""
        # Build the graph
        nodeA = self._create_node(0)
        nodeB = self._create_node(1)
        nodeC = self._create_node(2)
        nodeD = self._create_node(3)
        nodeE = self._create_node(4)
        terminal = self._create_node(5)

        # Linear path: A → B → C → D → E → Terminal
        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeC.next_zone = nodeD
        nodeD.next_zone = nodeE
        nodeE.next_zone = terminal

        # Multiple jump targets
        nodeB.jump_tokens = np.array([25], dtype=np.int32)
        nodeB.jump_zone = nodeD

        nodeC.jump_tokens = np.array([35], dtype=np.int32)
        nodeC.jump_zone = nodeE

        # Lower from head
        result = nodeA.lower()

        # Verify jump structure
        nodeB_lowered = result.next_zone
        nodeC_lowered = result.next_zone.next_zone

        self.assertEqual(nodeB_lowered.jump_zone.block, 3)  # B → D
        self.assertEqual(nodeC_lowered.jump_zone.block, 4)  # C → E

        # Verify linear path still intact
        self.assertEqual(nodeB_lowered.next_zone.block, 2)  # B → C
        self.assertEqual(nodeC_lowered.next_zone.block, 3)  # C → D


class TestRZCPNodeErrorHandling(unittest.TestCase):
    """Test error handling and exception propagation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create proper mock callback that returns numpy array
        self.mock_callback = Mock(return_value=np.array([1, 2, 3]))

        self.base_node_data = {
            'sequence': 'error_sequence',
            'block': 5,
            'zone_advance_tokens': np.array([10], dtype=np.int32),
            'tags': np.array([True], dtype=np.bool_),
            'timeout': 1000,
            'sampling_callback': self.mock_callback,  # Use proper mock
            'input': False,
            'output': False
        }

    def test_post_init_validation_error_context(self):
        """Test that __post_init__ validation includes proper context."""
        node_data = self.base_node_data.copy()
        node_data['jump_tokens'] = np.array([20], dtype=np.int32)
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

    def test_cycle_detection_prevents_infinite_recursion(self):
        """Test that lowering with cycles doesn't cause infinite recursion."""
        # Create a cycle: A → B → C → B
        nodeA = RZCPNode(**self.base_node_data)

        nodeB_data = self.base_node_data.copy()
        nodeB_data['block'] = 6
        nodeB = RZCPNode(**nodeB_data)

        nodeC_data = self.base_node_data.copy()
        nodeC_data['block'] = 7
        nodeC = RZCPNode(**nodeC_data)

        # Create the cycle
        nodeA.next_zone = nodeB
        nodeB.next_zone = nodeC
        nodeC.next_zone = nodeB  # Cycle back to B

        # This should complete without infinite recursion
        result = nodeA.lower()

        # Verify the cycle is preserved in lowered graph
        self.assertEqual(result.block, 5)  # A
        self.assertEqual(result.next_zone.block, 6)  # B
        self.assertEqual(result.next_zone.next_zone.block, 7)  # C
        self.assertEqual(result.next_zone.next_zone.next_zone.block, 6)  # Back to B

        # Verify they're the same instance (proper cycle detection)
        nodeB_lowered = result.next_zone
        nodeB_from_cycle = result.next_zone.next_zone.next_zone
        self.assertEqual(nodeB_lowered, nodeB_from_cycle)


if __name__ == "__main__":
    unittest.main()