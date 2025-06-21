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

# Import the modules under test
from src.workflow_forge.zcp.nodes import ZCPNode, RZCPNode, GraphLoweringError
from src.workflow_forge.resources import AbstractResource


class TestZCPNodeConstruction(unittest.TestCase):
    """Test ZCPNode creation and basic properties."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock construction callback
        self.mock_construction_callback = Mock()
        self.mock_construction_callback.return_value = "resolved text"

        # Basic valid node data
        self.valid_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'construction_callback': self.mock_construction_callback,
            'resource_specs': {'placeholder': {'name': 'test_resource', 'arguments': None, 'type': 'default'}},
            'raw_text': 'Test {placeholder} text',
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000
        }

    def test_valid_node_creation(self):
        """Test creating a valid ZCPNode with all required fields."""
        node = ZCPNode(**self.valid_node_data)

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
        node = ZCPNode(**self.valid_node_data)
        self.assertIsNone(node.next_zone)

    def test_construction_callback_assignment(self):
        """Test that construction callback is properly assigned."""
        node = ZCPNode(**self.valid_node_data)
        self.assertEqual(node.construction_callback, self.mock_construction_callback)

        # Verify it's callable
        self.assertTrue(callable(node.construction_callback))

    def test_empty_resource_specs(self):
        """Test node creation with empty resource specs."""
        node_data = self.valid_node_data.copy()
        node_data['resource_specs'] = {}
        node_data['raw_text'] = 'Text with no placeholders'

        node = ZCPNode(**node_data)
        self.assertEqual(node.resource_specs, {})

    def test_empty_tags(self):
        """Test node creation with empty tags list."""
        node_data = self.valid_node_data.copy()
        node_data['tags'] = []

        node = ZCPNode(**node_data)
        self.assertEqual(node.tags, [])

    def test_complex_resource_specs(self):
        """Test node creation with complex resource specifications."""
        complex_specs = {
            'principle': {'name': 'constitution', 'arguments': None, 'type': 'default'},
            'count': {'name': 'counter', 'arguments': {'start': 5}, 'type': 'flow_control'},
            'feedback': {'name': 'feedback_sampler', 'arguments': {'num_samples': 3}, 'type': 'default'}
        }

        node_data = self.valid_node_data.copy()
        node_data['resource_specs'] = complex_specs
        node_data['raw_text'] = 'Follow {principle}, repeat {count} times, consider {feedback}'

        node = ZCPNode(**node_data)
        self.assertEqual(node.resource_specs, complex_specs)


class TestZCPNodeLinkedList(unittest.TestCase):
    """Test linked list operations and chain traversal."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_callback = Mock(return_value="resolved")

        self.base_node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'construction_callback': self.mock_callback,
            'resource_specs': {},
            'raw_text': 'Test text',
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000
        }

    def test_get_last_node_single(self):
        """Test get_last_node with single node returns self."""
        node = ZCPNode(**self.base_node_data)

        last_node = node.get_last_node()
        self.assertEqual(last_node, node)

    def test_get_last_node_chain(self):
        """Test get_last_node traverses to end of chain."""
        # Create three nodes
        node1 = ZCPNode(**self.base_node_data)

        node2_data = self.base_node_data.copy()
        node2_data['block'] = 1
        node2 = ZCPNode(**node2_data)

        node3_data = self.base_node_data.copy()
        node3_data['block'] = 2
        node3 = ZCPNode(**node3_data)

        # Link them: node1 -> node2 -> node3
        node1.next_zone = node2
        node2.next_zone = node3

        # get_last_node should return node3 from any starting point
        self.assertEqual(node1.get_last_node(), node3)
        self.assertEqual(node2.get_last_node(), node3)
        self.assertEqual(node3.get_last_node(), node3)

    def test_chain_building(self):
        """Test building chains by setting next_zone."""
        node1 = ZCPNode(**self.base_node_data)

        node2_data = self.base_node_data.copy()
        node2_data['block'] = 1
        node2 = ZCPNode(**node2_data)

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
            nodes.append(ZCPNode(**node_data))

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

    def test_empty_chain(self):
        """Test behavior with single unconnected node."""
        node = ZCPNode(**self.base_node_data)

        # Should be its own last node
        self.assertEqual(node.get_last_node(), node)

        # Should have no next
        self.assertIsNone(node.next_zone)


class TestZCPNodeResourceResolution(unittest.TestCase):
    """Test resource resolution via construction callbacks."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_construction_callback = Mock()
        self.mock_construction_callback.return_value = "resolved text"

        self.mock_resource = Mock(spec=AbstractResource)
        self.mock_resource.return_value = "resource value"

        # Node with resource dependencies
        self.node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'construction_callback': self.mock_construction_callback,
            'resource_specs': {
                'placeholder': {'name': 'test_resource', 'arguments': None, 'type': 'default'}
            },
            'raw_text': 'Test {placeholder} text',
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000
        }

    def test_make_sampling_factory_success(self):
        """Test _make_sampling_factory creates working sampling function."""
        node = ZCPNode(**self.node_data)
        resources = {'test_resource': self.mock_resource}

        # Create sampling factory
        sampling_fn = node._make_sampling_factory(resources)

        # Call the sampling function
        result = sampling_fn()

        # Verify construction callback was called with resources
        self.mock_construction_callback.assert_called_once_with(resources)

        # Verify final result
        self.assertEqual(result, "resolved text")

    def test_make_sampling_factory_construction_callback_failure(self):
        """Test _make_sampling_factory when construction callback fails."""
        self.mock_construction_callback.side_effect = ValueError("Callback failed")

        node = ZCPNode(**self.node_data)
        resources = {'test_resource': self.mock_resource}

        sampling_fn = node._make_sampling_factory(resources)

        # Should raise GraphLoweringError with chained exception
        with self.assertRaises(GraphLoweringError) as context:
            sampling_fn()

        self.assertEqual(context.exception.sequence, "test_sequence")
        self.assertEqual(context.exception.block, 0)
        self.assertIsInstance(context.exception.__cause__, ValueError)

    def test_sampling_factory_multiple_calls(self):
        """Test that sampling factory can be called multiple times."""
        node = ZCPNode(**self.node_data)
        resources = {'test_resource': self.mock_resource}

        sampling_fn = node._make_sampling_factory(resources)

        # Call multiple times
        result1 = sampling_fn()
        result2 = sampling_fn()

        # Should work both times
        self.assertEqual(result1, "resolved text")
        self.assertEqual(result2, "resolved text")

        # Verify callbacks were called each time
        self.assertEqual(self.mock_construction_callback.call_count, 2)


class TestZCPNodeLowering(unittest.TestCase):
    """Test graph lowering operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_construction_callback = Mock()
        self.mock_construction_callback.return_value = "resolved text"

        self.mock_resource = Mock(spec=AbstractResource)
        self.mock_resource.return_value = "resource value"

        self.resources = {'test_resource': self.mock_resource}

        # Base node data
        self.node_data = {
            'sequence': 'test_sequence',
            'block': 0,
            'construction_callback': self.mock_construction_callback,
            'resource_specs': {
                'placeholder': {'name': 'test_resource', 'arguments': None, 'type': 'default'}
            },
            'raw_text': 'Test {placeholder} text',
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000
        }

    def test_lower_node_success(self):
        """Test _lower_node creates valid RZCPNode."""
        node = ZCPNode(**self.node_data)

        result = node._lower_node(self.resources)

        # Verify result type and basic properties
        self.assertIsInstance(result, RZCPNode)
        self.assertEqual(result.sequence, 'test_sequence')
        self.assertEqual(result.block, 0)
        self.assertEqual(result.timeout, 1000)
        self.assertFalse(result.input)
        self.assertFalse(result.output)
        self.assertIsNone(result.next_zone)
        self.assertIsNone(result.jump_advance_str)
        self.assertIsNone(result.jump_zone)

        # Verify zone advance string
        self.assertEqual(result.zone_advance_str, '[Answer]')

        # Verify tags as string list
        self.assertEqual(result.tags, ['Training'])

        # Verify sampling callback exists and works
        self.assertTrue(callable(result.sampling_callback))
        text = result.sampling_callback()
        self.assertEqual(text, "resolved text")

    def test_lower_node_missing_resource(self):
        """Test _lower_node fails when required resource is missing."""
        node = ZCPNode(**self.node_data)
        empty_resources = {}

        with self.assertRaises(GraphLoweringError) as context:
            node._lower_node(empty_resources)

        self.assertEqual(context.exception.sequence, "test_sequence")
        self.assertEqual(context.exception.block, 0)

    def test_lower_single_node(self):
        """Test lower() method with single node."""
        node = ZCPNode(**self.node_data)

        result = node.lower(self.resources)

        # Should return RZCPNode
        self.assertIsInstance(result, RZCPNode)
        self.assertEqual(result.sequence, 'test_sequence')
        self.assertIsNone(result.next_zone)

    def test_lower_chain_of_nodes(self):
        """Test lower() method with chain of nodes."""
        # Create two nodes
        node1 = ZCPNode(**self.node_data)

        node2_data = self.node_data.copy()
        node2_data['block'] = 1
        node2_data['sequence'] = 'test_sequence'
        node2 = ZCPNode(**node2_data)

        # Link them
        node1.next_zone = node2

        # Lower the chain
        result_head = node1.lower(self.resources)

        # Verify chain structure is preserved
        self.assertIsInstance(result_head, RZCPNode)
        self.assertEqual(result_head.block, 0)

        self.assertIsNotNone(result_head.next_zone)
        self.assertIsInstance(result_head.next_zone, RZCPNode)
        self.assertEqual(result_head.next_zone.block, 1)

        self.assertIsNone(result_head.next_zone.next_zone)

    def test_lower_preserves_node_independence(self):
        """Test that lowering creates independent RZCPNode instances."""
        node1 = ZCPNode(**self.node_data)

        node2_data = self.node_data.copy()
        node2_data['block'] = 1
        node2 = ZCPNode(**node2_data)

        node1.next_zone = node2

        # Lower the chain
        result_head = node1.lower(self.resources)

        # Verify original ZCP nodes are unchanged
        self.assertIsInstance(node1, ZCPNode)
        self.assertIsInstance(node2, ZCPNode)
        self.assertEqual(node1.next_zone, node2)

        # Verify lowered nodes are different instances
        self.assertNotEqual(result_head, node1)
        self.assertNotEqual(result_head.next_zone, node2)


class TestZCPNodeErrorHandling(unittest.TestCase):
    """Test error handling and exception propagation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_construction_callback = Mock()
        self.mock_construction_callback.return_value = "resolved text"

        self.node_data = {
            'sequence': 'error_sequence',
            'block': 5,
            'construction_callback': self.mock_construction_callback,
            'resource_specs': {},
            'raw_text': 'Test text',
            'zone_advance_str': '[Answer]',
            'tags': ['Training'],
            'timeout': 1000
        }

    def test_sampling_factory_error_context(self):
        """Test error context in sampling factory."""
        self.mock_construction_callback.side_effect = RuntimeError("Construction failed")

        node = ZCPNode(**self.node_data)
        sampling_fn = node._make_sampling_factory({})

        with self.assertRaises(GraphLoweringError) as context:
            sampling_fn()

        # Verify error context and cause
        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 5)
        self.assertIsInstance(context.exception.__cause__, RuntimeError)

    def test_lower_propagates_node_errors(self):
        """Test that lower() propagates errors from _lower_node."""
        # Create a node with a resource that doesn't exist
        node_data = self.node_data.copy()
        node_data['resource_specs'] = {'missing': {'name': 'missing_resource', 'arguments': None, 'type': 'default'}}

        node = ZCPNode(**node_data)

        with self.assertRaises(GraphLoweringError) as context:
            node.lower({})

        # Should be the same error that _lower_node would raise
        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 5)

    def test_lower_chain_error_propagation(self):
        """Test error propagation when lowering chains."""
        # Create chain where second node will fail
        node1 = ZCPNode(**self.node_data)

        node2_data = self.node_data.copy()
        node2_data['block'] = 6
        node2_data['resource_specs'] = {'missing': {'name': 'missing_resource', 'arguments': None, 'type': 'default'}}
        node2 = ZCPNode(**node2_data)

        node1.next_zone = node2

        with self.assertRaises(GraphLoweringError) as context:
            node1.lower({})

        # Error should reference the failing node's context
        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 6)  # Should be node2's block


if __name__ == "__main__":
    unittest.main()