"""
Unit tests for ZCPNode class.

Tests cover:
1. Constructor and basic properties
2. Linked list operations and chain traversal
3. Resource resolution via construction callbacks
4. Graph lowering operations
5. Error handling and exception propagation
"""

import unittest
from unittest.mock import Mock
from typing import Dict, Any

# Import the modules under test
from src.workflow_forge.zcp.nodes import ZCPNode, RZCPNode, GraphLoweringError
from src.workflow_forge.resources import AbstractResource
from src.workflow_forge.parsing.config_parsing import Config


class BaseZCPNodeTest(unittest.TestCase):
    """Base test class with common setup and helper methods."""

    def setUp(self):
        """Set up common test fixtures."""
        # Create mock construction callback
        self.mock_construction_callback = Mock()
        self.mock_construction_callback.return_value = "resolved text"

        # Create mock resource
        self.mock_resource = Mock(spec=AbstractResource)
        self.mock_resource.return_value = "resource value"

        # Create mock config
        self.mock_config = Mock(spec=Config)
        self.mock_config.escape_patterns = ("[Escape]", "[EndEscape]")

        # Valid resources dictionary
        self.resources = {'test_resource': self.mock_resource}

    def get_valid_node_data(self, **overrides) -> Dict[str, Any]:
        """
        Return valid node data for testing, with optional field overrides.

        Args:
            **overrides: Fields to override in the base node data

        Returns:
            Dictionary of valid ZCPNode constructor arguments
        """
        base_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'construction_callback': self.mock_construction_callback,
            'resource_specs': {'placeholder': {'name': 'test_resource', 'arguments': None, 'type': 'default'}},
            'raw_text': 'Test {placeholder} text',
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000
        }
        base_data.update(overrides)
        return base_data

    def create_node(self, **overrides) -> ZCPNode:
        """
        Create a ZCPNode with valid data and optional overrides.

        Args:
            **overrides: Fields to override in the base node data

        Returns:
            Configured ZCPNode instance
        """
        return ZCPNode(**self.get_valid_node_data(**overrides))

    def create_node_chain(self, length: int) -> ZCPNode:
        """
        Create a chain of linked ZCPNodes.

        Args:
            length: Number of nodes in the chain

        Returns:
            Head node of the chain
        """
        if length < 1:
            raise ValueError("Chain length must be at least 1")

        # Create nodes
        nodes = []
        for i in range(length):
            node = self.create_node(block=i)
            nodes.append(node)

        # Link them
        for i in range(length - 1):
            nodes[i].next_zone = nodes[i + 1]

        return nodes[0]

    def assert_rzcp_node_properties(self, rzcp_node: RZCPNode, expected_sequence: str,
                                  expected_block: int, expected_timeout: int = 1000):
        """
        Assert common properties of an RZCPNode.

        Args:
            rzcp_node: The RZCPNode to validate
            expected_sequence: Expected sequence name
            expected_block: Expected block number
            expected_timeout: Expected timeout value
        """
        self.assertIsInstance(rzcp_node, RZCPNode)
        self.assertEqual(rzcp_node.sequence, expected_sequence)
        self.assertEqual(rzcp_node.block, expected_block)
        self.assertEqual(rzcp_node.timeout, expected_timeout)
        self.assertFalse(rzcp_node.input)
        self.assertFalse(rzcp_node.output)
        self.assertIsNone(rzcp_node.jump_advance_str)
        self.assertIsNone(rzcp_node.jump_zone)
        self.assertTrue(callable(rzcp_node.sampling_callback))


class TestZCPNodeConstruction(BaseZCPNodeTest):
    """Test ZCPNode creation and basic properties."""

    def test_valid_node_creation(self):
        """Test creating a valid ZCPNode with all required fields."""
        node_data = self.get_valid_node_data()
        node = ZCPNode(**node_data)

        self.assertEqual(node.sequence, 'test_sequence')
        self.assertEqual(node.block, 0)
        self.assertEqual(node.construction_callback, self.mock_construction_callback)
        self.assertEqual(node.resource_specs, {'placeholder': {'name': 'test_resource', 'arguments': None, 'type': 'default'}})
        self.assertEqual(node.raw_text, 'Test {placeholder} text')
        self.assertEqual(node.zone_advance_str, '[Answer]')
        self.assertEqual(node.tags, ['Training'])
        self.assertEqual(node.timeout, 1000)
        self.assertIsNone(node.next_zone)

    def test_default_next_zone(self):
        """Test that next_zone defaults to None."""
        node = self.create_node()
        self.assertIsNone(node.next_zone)

    def test_construction_callback_assignment(self):
        """Test that construction callback is properly assigned."""
        node = self.create_node()
        self.assertEqual(node.construction_callback, self.mock_construction_callback)
        self.assertTrue(callable(node.construction_callback))

    def test_empty_resource_specs(self):
        """Test node creation with empty resource specs."""
        node = self.create_node(
            resource_specs={},
            raw_text='Text with no placeholders'
        )
        self.assertEqual(node.resource_specs, {})

    def test_empty_tags(self):
        """Test node creation with empty tags list."""
        node = self.create_node(tags=[])
        self.assertEqual(node.tags, [])

    def test_complex_resource_specs(self):
        """Test node creation with complex resource specifications."""
        complex_specs = {
            'principle': {'name': 'constitution', 'arguments': None, 'type': 'default'},
            'count': {'name': 'counter', 'arguments': {'start': 5}, 'type': 'flow_control'},
            'feedback': {'name': 'feedback_sampler', 'arguments': {'num_samples': 3}, 'type': 'default'}
        }

        node = self.create_node(
            resource_specs=complex_specs,
            raw_text='Follow {principle}, repeat {count} times, consider {feedback}'
        )
        self.assertEqual(node.resource_specs, complex_specs)


class TestZCPNodeLinkedList(BaseZCPNodeTest):
    """Test linked list operations and chain traversal."""

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

    def test_empty_chain(self):
        """Test behavior with single unconnected node."""
        node = self.create_node()

        # Should be its own last node
        self.assertEqual(node.get_last_node(), node)

        # Should have no next
        self.assertIsNone(node.next_zone)


class TestZCPNodeResourceResolution(BaseZCPNodeTest):
    """Test resource resolution via construction callbacks."""

    def test_make_sampling_factory_success(self):
        """Test _make_sampling_factory creates working sampling function."""
        node = self.create_node()

        # Create sampling factory
        sampling_fn = node._make_sampling_factory(self.resources)

        # Call the sampling function
        result = sampling_fn()

        # Verify construction callback was called with resources
        self.mock_construction_callback.assert_called_once_with(self.resources)

        # Verify final result
        self.assertEqual(result, "resolved text")

    def test_make_sampling_factory_construction_callback_failure(self):
        """Test _make_sampling_factory when construction callback fails."""
        self.mock_construction_callback.side_effect = ValueError("Callback failed")

        node = self.create_node()
        sampling_fn = node._make_sampling_factory(self.resources)

        # Should raise GraphLoweringError with chained exception
        with self.assertRaises(GraphLoweringError) as context:
            sampling_fn()

        self.assertEqual(context.exception.sequence, "test_sequence")
        self.assertEqual(context.exception.block, 0)
        self.assertIsInstance(context.exception.__cause__, ValueError)

    def test_sampling_factory_multiple_calls(self):
        """Test that sampling factory can be called multiple times."""
        node = self.create_node()
        sampling_fn = node._make_sampling_factory(self.resources)

        # Call multiple times
        result1 = sampling_fn()
        result2 = sampling_fn()

        # Should work both times
        self.assertEqual(result1, "resolved text")
        self.assertEqual(result2, "resolved text")

        # Verify callbacks were called each time
        self.assertEqual(self.mock_construction_callback.call_count, 2)


class TestZCPNodeLowering(BaseZCPNodeTest):
    """Test graph lowering operations."""

    def test_lower_single_node(self):
        """Test lower() method with single node."""
        node = self.create_node()
        result = node.lower(self.resources, self.mock_config)

        # Should return RZCPNode
        self.assertIsInstance(result, RZCPNode)
        self.assertEqual(result.sequence, 'test_sequence')
        self.assertIsNone(result.next_zone)

    def test_lower_single_node_success(self):
        """Test lower() method with single node creates valid RZCPNode."""
        node = self.create_node()
        result = node.lower(self.resources, self.mock_config)

        # Verify result using helper assertion
        self.assert_rzcp_node_properties(result, 'test_sequence', 0)

        # Verify specific properties
        self.assertEqual(result.zone_advance_str, '[Answer]')
        self.assertEqual(result.tags, ['Training'])
        self.assertIsNone(result.next_zone)

        # Verify sampling callback works
        text = result.sampling_callback()
        self.assertEqual(text, "resolved text")

    def test_lower_missing_resource(self):
        """Test lower() fails when required resource is missing."""
        node = self.create_node()
        empty_resources = {}

        with self.assertRaises(GraphLoweringError) as context:
            node.lower(empty_resources, self.mock_config)

        self.assertEqual(context.exception.sequence, "test_sequence")
        self.assertEqual(context.exception.block, 0)

    def test_lower_chain_of_nodes(self):
        """Test lower() method with chain of nodes."""
        # Create two-node chain
        head_node = self.create_node_chain(2)

        # Lower the chain
        result_head = head_node.lower(self.resources, self.mock_config)

        # Verify chain structure is preserved
        self.assert_rzcp_node_properties(result_head, 'test_sequence', 0)

        self.assertIsNotNone(result_head.next_zone)
        self.assert_rzcp_node_properties(result_head.next_zone, 'test_sequence', 1)

        self.assertIsNone(result_head.next_zone.next_zone)

    def test_lower_preserves_node_independence(self):
        """Test that lowering creates independent RZCPNode instances."""
        # Create two-node chain
        head_node = self.create_node_chain(2)
        second_node = head_node.next_zone

        # Lower the chain
        result_head = head_node.lower(self.resources, self.mock_config)

        # Verify original ZCP nodes are unchanged
        self.assertIsInstance(head_node, ZCPNode)
        self.assertIsInstance(second_node, ZCPNode)
        self.assertEqual(head_node.next_zone, second_node)

        # Verify lowered nodes are different instances
        self.assertNotEqual(result_head, head_node)
        self.assertNotEqual(result_head.next_zone, second_node)


class TestZCPNodeErrorHandling(BaseZCPNodeTest):
    """Test error handling and exception propagation."""

    def test_sampling_factory_error_context(self):
        """Test error context in sampling factory."""
        self.mock_construction_callback.side_effect = RuntimeError("Construction failed")

        node = self.create_node(sequence='error_sequence', block=5)
        sampling_fn = node._make_sampling_factory({})

        with self.assertRaises(GraphLoweringError) as context:
            sampling_fn()

        # Verify error context and cause
        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 5)
        self.assertIsInstance(context.exception.__cause__, RuntimeError)

    def test_lower_chain_error_propagation(self):
        """Test error propagation when lowering chains."""
        # Create chain where second node will fail
        node1 = self.create_node(sequence='error_sequence', block=5)
        node2 = self.create_node(
            sequence='error_sequence',
            block=6,
            resource_specs={'missing': {'name': 'missing_resource', 'arguments': None, 'type': 'default'}}
        )

        node1.next_zone = node2

        with self.assertRaises(GraphLoweringError) as context:
            node1.lower({}, self.mock_config)

        # Error should reference the failing node's context
        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 6)  # Should be node2's block

    def test_lower_chain_error_propagation(self):
        """Test error propagation when lowering chains."""
        # Create chain where second node will fail due to missing resource
        node1 = self.create_node(sequence='error_sequence', block=5, resource_specs={})
        node2 = self.create_node(
            sequence='error_sequence',
            block=6,
            resource_specs={'missing': {'name': 'missing_resource', 'arguments': None, 'type': 'default'}}
        )

        node1.next_zone = node2

        # When lowering the chain, node2 should fail because 'missing_resource' is not in resources
        with self.assertRaises(GraphLoweringError) as context:
            node1.lower({}, self.mock_config)

        # Error should reference the failing node's context
        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 6)  # Should be node2's block


if __name__ == "__main__":
    unittest.main()