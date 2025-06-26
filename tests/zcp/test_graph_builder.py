"""
Unit tests for GraphBuilderNode

Tests the forward reference resolution system for building complex
RZCP graphs with loops, branches, and merges.
"""

import unittest
from unittest.mock import Mock
from typing import List

# Import the modules under test
from src.workflow_forge.zcp.builder import GraphBuilderNode, GraphBuilderException
from src.workflow_forge.zcp.nodes import RZCPNode


class TestGraphBuilderNode(unittest.TestCase):
    """Test GraphBuilderNode forward reference resolution."""

    def setUp(self):
        """Set up test fixtures."""
        self.jump_pattern = "[Jump]"

    def create_mock_rzcp_node(self, sequence: str, block: int) -> RZCPNode:
        """Create a mock RZCP node for testing."""
        mock_node = Mock(spec=RZCPNode)
        mock_node.sequence = sequence
        mock_node.block = block
        mock_node.next_zone = None
        mock_node.jump_zone = None
        mock_node.jump_advance_str = None
        mock_node.get_last_node.return_value = mock_node  # Single node chain
        return mock_node

    def create_mock_rzcp_chain(self, sequence: str, length: int) -> RZCPNode:
        """Create a mock chain of RZCP nodes."""
        nodes = []
        for i in range(length):
            node = self.create_mock_rzcp_node(sequence, i)
            nodes.append(node)

        # Link the chain
        for i in range(length - 1):
            nodes[i].next_zone = nodes[i + 1]

        # Make get_last_node return the actual last node
        for node in nodes:
            node.get_last_node.return_value = nodes[-1]

        return nodes[0]  # Return head

    def test_initialization(self):
        """Test GraphBuilderNode initialization."""
        # Test with no forward references
        builder = GraphBuilderNode(self.jump_pattern)
        self.assertEqual(builder.jump_pattern, "[Jump]")
        self.assertEqual(builder.nominal_refs, [])
        self.assertEqual(builder.flow_control_refs, [])
        self.assertIsNone(builder.head)

        # Test with forward references
        nominal_node = self.create_mock_rzcp_node("test", 0)
        flow_node = self.create_mock_rzcp_node("test", 1)

        builder = GraphBuilderNode(
            self.jump_pattern,
            nominal_refs=[nominal_node],
            flow_control_refs=[flow_node]
        )

        self.assertEqual(builder.nominal_refs, [nominal_node])
        self.assertEqual(builder.flow_control_refs, [flow_node])

    def test_extend_basic(self):
        """Test basic extend operation."""
        # Create initial builder with forward reference
        source_node = self.create_mock_rzcp_node("source", 0)
        builder = GraphBuilderNode(self.jump_pattern, nominal_refs=[source_node])

        # Create target sequence
        target_node = self.create_mock_rzcp_node("target", 0)

        # Extend
        new_builder = builder.extend(target_node)

        # Verify forward reference was resolved
        self.assertEqual(source_node.next_zone, target_node)
        self.assertEqual(builder.head, target_node)

        # Verify new builder has target as forward reference
        self.assertEqual(new_builder.nominal_refs, [target_node])
        self.assertEqual(new_builder.flow_control_refs, [])

    def test_extend_with_chain(self):
        """Test extend with a chain of nodes."""
        # Create initial builder
        source_node = self.create_mock_rzcp_node("source", 0)
        builder = GraphBuilderNode(self.jump_pattern, nominal_refs=[source_node])

        # Create target chain
        target_chain = self.create_mock_rzcp_chain("target", 3)
        target_tail = target_chain.get_last_node()

        # Extend
        new_builder = builder.extend(target_chain)

        # Verify forward reference points to chain head
        self.assertEqual(source_node.next_zone, target_chain)

        # Verify new builder has chain tail as forward reference
        self.assertEqual(new_builder.nominal_refs, [target_tail])

    def test_fork_basic(self):
        """Test basic fork operation for flow control."""
        # Create initial builder with forward reference
        source_node = self.create_mock_rzcp_node("source", 0)
        builder = GraphBuilderNode(self.jump_pattern, nominal_refs=[source_node])

        # Create control node
        control_node = self.create_mock_rzcp_node("control", 0)

        # Fork
        nominal_builder, jump_builder = builder.fork(control_node)

        # Verify forward reference was resolved
        self.assertEqual(source_node.next_zone, control_node)
        self.assertEqual(builder.head, control_node)

        # Verify nominal path builder (first return value)
        self.assertEqual(nominal_builder.nominal_refs, [control_node])
        self.assertEqual(nominal_builder.flow_control_refs, [])

        # Verify jump path builder (second return value)
        self.assertEqual(jump_builder.nominal_refs, [])
        self.assertEqual(jump_builder.flow_control_refs, [control_node])

    def test_fork_wires_both_paths(self):
        """Test that fork creates proper forward references for both paths."""
        source_node = self.create_mock_rzcp_node("source", 0)
        builder = GraphBuilderNode(self.jump_pattern, nominal_refs=[source_node])

        control_node = self.create_mock_rzcp_node("control", 0)
        nominal_builder, jump_builder = builder.fork(control_node)

        # Extend nominal path
        nominal_target = self.create_mock_rzcp_node("nominal", 0)
        nominal_builder.extend(nominal_target)

        # Extend jump path
        jump_target = self.create_mock_rzcp_node("jump", 0)
        jump_builder.extend(jump_target)

        # Verify both paths are wired correctly
        self.assertEqual(control_node.next_zone, nominal_target)
        self.assertEqual(control_node.jump_zone, jump_target)
        self.assertEqual(control_node.jump_advance_str, self.jump_pattern)

    def test_merge_basic(self):
        """Test basic merge operation."""
        # Create two builders with different forward references
        node1 = self.create_mock_rzcp_node("branch1", 0)
        node2 = self.create_mock_rzcp_node("branch2", 0)

        builder1 = GraphBuilderNode(self.jump_pattern, nominal_refs=[node1])
        builder2 = GraphBuilderNode(self.jump_pattern, nominal_refs=[node2])

        # Merge
        merged_builder = GraphBuilderNode.merge(builder1, builder2)

        # Verify merged builder has both forward references
        self.assertEqual(set(merged_builder.nominal_refs), {node1, node2})
        self.assertEqual(merged_builder.flow_control_refs, [])
        self.assertEqual(merged_builder.jump_pattern, self.jump_pattern)

    def test_merge_resolves_all_references(self):
        """Test that merge properly resolves all forward references."""
        # Create multiple builders
        node1 = self.create_mock_rzcp_node("branch1", 0)
        node2 = self.create_mock_rzcp_node("branch2", 0)
        node3 = self.create_mock_rzcp_node("branch3", 0)

        builder1 = GraphBuilderNode(self.jump_pattern, nominal_refs=[node1])
        builder2 = GraphBuilderNode(self.jump_pattern, nominal_refs=[node2])
        builder3 = GraphBuilderNode(self.jump_pattern, nominal_refs=[node3])

        # Merge all
        merged_builder = GraphBuilderNode.merge(builder1, builder2, builder3)

        # Extend merged builder
        target_node = self.create_mock_rzcp_node("target", 0)
        merged_builder.extend(target_node)

        # Verify all forward references were resolved
        self.assertEqual(node1.next_zone, target_node)
        self.assertEqual(node2.next_zone, target_node)
        self.assertEqual(node3.next_zone, target_node)

    def test_correct_loop_pattern(self):
        """Test the correct loop construction pattern."""
        # Setup
        setup_node = self.create_mock_rzcp_node("setup", 0)
        builder = GraphBuilderNode(self.jump_pattern)
        builder = builder.extend(setup_node)

        # Create loop control
        loop_control = self.create_mock_rzcp_node("loop_control", 0)
        loop_back, main = builder.fork(loop_control)

        # Add loop body to loop_back path
        loop_body = self.create_mock_rzcp_chain("loop_body", 3)
        loop_body_builder = loop_back.extend(loop_body)

        # Create the loop: attach loop body back to original builder
        builder.attach(loop_body_builder)

        # Verify loop structure
        # Setup connects to loop control
        self.assertEqual(setup_node.next_zone, loop_control)

        # Loop control's next_zone goes to loop body (loop_back path)
        self.assertEqual(loop_control.next_zone, loop_body)

        # Loop body's tail connects back to loop control (the loop!)
        loop_tail = loop_body.get_last_node()
        self.assertEqual(loop_tail.next_zone, loop_control)


    def test_error_conditions(self):
        """Test error conditions and exception handling."""
        # Test merge with no builders
        with self.assertRaises(GraphBuilderException):
            GraphBuilderNode.merge()

        # Test attach to unresolved builder
        source_builder = GraphBuilderNode(self.jump_pattern)
        target_builder = GraphBuilderNode(self.jump_pattern)  # No head set

        with self.assertRaises(GraphBuilderException):
            source_builder.attach(target_builder)

        # Test double-wiring nominal reference
        node = self.create_mock_rzcp_node("test", 0)
        node.next_zone = Mock()  # Already has a connection

        builder = GraphBuilderNode(self.jump_pattern, nominal_refs=[node])
        target = self.create_mock_rzcp_node("target", 0)

        with self.assertRaises(GraphBuilderException):
            builder.extend(target)

        # Test double-wiring flow control reference
        node = self.create_mock_rzcp_node("test", 0)
        node.jump_zone = Mock()  # Already has a connection

        builder = GraphBuilderNode(self.jump_pattern, flow_control_refs=[node])
        target = self.create_mock_rzcp_node("target", 0)

        with self.assertRaises(GraphBuilderException):
            builder.extend(target)

    def test_mixed_reference_types(self):
        """Test handling mixed nominal and flow control references."""
        nominal_node = self.create_mock_rzcp_node("nominal", 0)
        flow_node = self.create_mock_rzcp_node("flow", 0)

        builder = GraphBuilderNode(
            self.jump_pattern,
            nominal_refs=[nominal_node],
            flow_control_refs=[flow_node]
        )

        target = self.create_mock_rzcp_node("target", 0)
        new_builder = builder.extend(target)

        # Verify both types of references were resolved
        self.assertEqual(nominal_node.next_zone, target)
        self.assertEqual(flow_node.jump_zone, target)
        self.assertEqual(flow_node.jump_advance_str, self.jump_pattern)

        # Verify new builder only has target as nominal reference
        self.assertEqual(new_builder.nominal_refs, [target])
        self.assertEqual(new_builder.flow_control_refs, [])


if __name__ == "__main__":
    unittest.main()