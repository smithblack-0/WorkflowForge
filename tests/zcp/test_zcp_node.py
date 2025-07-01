"""
Unit tests for ZCPNode class.

Tests cover:
1. Constructor and basic properties
2. Linked list operations and chain traversal
3. Resource resolution via construction callbacks
4. Graph lowering operations
5. Error handling and exception propagation
6. Three-tier resource system (standard, custom, argument)
"""

import unittest
from unittest.mock import Mock
from typing import Dict, Any

# Import the modules under test
from workflow_forge.zcp.nodes import ZCPNode, RZCPNode, GraphLoweringError
from workflow_forge.resources import AbstractResource
from workflow_forge.frontend.parsing.config_parsing import Config


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
            'resource_specs': {'placeholder': {'name': 'test_resource', 'arguments': None, 'type': 'standard'}},
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
        self.assertEqual(node.resource_specs, {'placeholder': {'name': 'test_resource', 'arguments': None, 'type': 'standard'}})
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
            'principle': {'name': 'constitution', 'arguments': None, 'type': 'standard'},
            'count': {'name': 'counter', 'arguments': {'start': 5}, 'type': 'custom'},
            'feedback': {'name': 'feedback_sampler', 'arguments': {'num_samples': 3}, 'type': 'argument'}
        }

        node = self.create_node(
            resource_specs=complex_specs,
            raw_text='Follow {principle}, repeat {count} times, consider {feedback}'
        )
        self.assertEqual(node.resource_specs, complex_specs)

    def test_standard_resource_type(self):
        """Test node creation with standard resource type."""
        node = self.create_node(
            resource_specs={'test': {'name': 'test_resource', 'type': 'standard'}}
        )
        self.assertEqual(node.resource_specs['test']['type'], 'standard')

    def test_custom_resource_type(self):
        """Test node creation with custom resource type."""
        node = self.create_node(
            resource_specs={'test': {'name': 'test_resource', 'type': 'custom'}}
        )
        self.assertEqual(node.resource_specs['test']['type'], 'custom')

    def test_argument_resource_type(self):
        """Test node creation with argument resource type."""
        node = self.create_node(
            resource_specs={'test': {'name': 'test_resource', 'type': 'argument'}}
        )
        self.assertEqual(node.resource_specs['test']['type'], 'argument')


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

        # Call the sampling function with empty dynamic resources
        result = sampling_fn({})

        # Verify construction callback was called with resources
        self.mock_construction_callback.assert_called_once_with(self.resources)

        # Verify final result
        self.assertEqual(result, "resolved text")

    def test_make_sampling_factory_with_dynamic_resources(self):
        """Test _make_sampling_factory handles dynamic resources correctly."""
        node = self.create_node()

        # Create additional dynamic resource
        dynamic_mock = Mock(spec=AbstractResource)
        dynamic_mock.return_value = "dynamic value"
        dynamic_resources = {'dynamic_resource': dynamic_mock}

        # Create sampling factory
        sampling_fn = node._make_sampling_factory(self.resources)

        # Call with dynamic resources
        result = sampling_fn(dynamic_resources)

        # Verify construction callback was called with merged resources
        expected_resources = {**dynamic_resources, **self.resources}
        self.mock_construction_callback.assert_called_once_with(expected_resources)

        # Verify result
        self.assertEqual(result, "resolved text")

    def test_make_sampling_factory_resource_precedence(self):
        """Test that compile-time resources override dynamic resources."""
        node = self.create_node()

        # Create dynamic resource with same name as compile-time resource
        dynamic_mock = Mock(spec=AbstractResource)
        dynamic_mock.return_value = "dynamic value"
        dynamic_resources = {'test_resource': dynamic_mock}

        # Create sampling factory
        sampling_fn = node._make_sampling_factory(self.resources)

        # Call with conflicting dynamic resources
        result = sampling_fn(dynamic_resources)

        # Verify compile-time resource was used (not dynamic)
        expected_resources = {**dynamic_resources, **self.resources}
        self.assertEqual(expected_resources['test_resource'], self.mock_resource)
        self.mock_construction_callback.assert_called_once_with(expected_resources)

    def test_make_sampling_factory_missing_argument_resource(self):
        """Test _make_sampling_factory when argument resource is missing."""
        # Create node with argument resource
        node = self.create_node(
            resource_specs={
                'arg_placeholder': {'name': 'arg_resource', 'type': 'argument'}
            }
        )

        # Create sampling factory with no compile-time resources for argument type
        sampling_fn = node._make_sampling_factory({})

        # Should raise GraphError when argument resource not provided
        with self.assertRaises(Exception) as context:  # Could be GraphError or GraphLoweringError
            sampling_fn({})

        # Error should mention the missing argument resource
        error_msg = str(context.exception).lower()
        self.assertIn("arg_resource", error_msg)
        self.assertIn("argument", error_msg)

    def test_make_sampling_factory_construction_callback_failure(self):
        """Test _make_sampling_factory when construction callback fails."""
        self.mock_construction_callback.side_effect = ValueError("Callback failed")

        node = self.create_node()
        sampling_fn = node._make_sampling_factory(self.resources)

        # Should raise GraphLoweringError with chained exception
        with self.assertRaises(Exception) as context:  # Could be GraphError or GraphLoweringError
            sampling_fn({})

        self.assertEqual(context.exception.sequence, "test_sequence")
        self.assertEqual(context.exception.block, 0)
        self.assertIsInstance(context.exception.__cause__, ValueError)

    def test_sampling_factory_multiple_calls(self):
        """Test that sampling factory can be called multiple times."""
        node = self.create_node()
        sampling_fn = node._make_sampling_factory(self.resources)

        # Call multiple times
        result1 = sampling_fn({})
        result2 = sampling_fn({})

        # Should work both times
        self.assertEqual(result1, "resolved text")
        self.assertEqual(result2, "resolved text")

        # Verify callbacks were called each time
        self.assertEqual(self.mock_construction_callback.call_count, 2)

    def test_sampling_factory_with_different_dynamic_resources(self):
        """Test sampling factory with different dynamic resources per call."""
        node = self.create_node()
        sampling_fn = node._make_sampling_factory(self.resources)

        # Create different dynamic resources
        dynamic1 = {'dyn1': Mock(spec=AbstractResource)}
        dynamic2 = {'dyn2': Mock(spec=AbstractResource)}

        # Call with different dynamics each time
        result1 = sampling_fn(dynamic1)
        result2 = sampling_fn(dynamic2)

        # Should work both times
        self.assertEqual(result1, "resolved text")
        self.assertEqual(result2, "resolved text")

        # Verify different resource sets were passed
        self.assertEqual(self.mock_construction_callback.call_count, 2)
        call1_resources = self.mock_construction_callback.call_args_list[0][0][0]
        call2_resources = self.mock_construction_callback.call_args_list[1][0][0]

        self.assertIn('dyn1', call1_resources)
        self.assertNotIn('dyn2', call1_resources)
        self.assertIn('dyn2', call2_resources)
        self.assertNotIn('dyn1', call2_resources)


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
        text = result.sampling_callback({})
        self.assertEqual(text, "resolved text")

    def test_lower_missing_standard_resource(self):
        """Test lower() fails when required standard resource is missing."""
        node = self.create_node(
            resource_specs={
                'placeholder': {'name': 'missing_resource', 'type': 'standard'}
            }
        )
        empty_resources = {}

        with self.assertRaises(GraphLoweringError) as context:
            node.lower(empty_resources, self.mock_config)

        self.assertEqual(context.exception.sequence, "test_sequence")
        self.assertEqual(context.exception.block, 0)

    def test_lower_missing_custom_resource(self):
        """Test lower() fails when required custom resource is missing."""
        node = self.create_node(
            resource_specs={
                'placeholder': {'name': 'missing_resource', 'type': 'custom'}
            }
        )
        empty_resources = {}

        with self.assertRaises(GraphLoweringError) as context:
            node.lower(empty_resources, self.mock_config)

        self.assertEqual(context.exception.sequence, "test_sequence")
        self.assertEqual(context.exception.block, 0)

    def test_lower_with_argument_resources_skips_validation(self):
        """Test that argument-type resources are not validated during lowering."""
        node = self.create_node(
            resource_specs={
                'arg_placeholder': {'name': 'arg_resource', 'type': 'argument'}
            }
        )

        # Should succeed even though arg_resource is not in resources
        result = node.lower({}, self.mock_config)
        self.assertIsInstance(result, RZCPNode)
        self.assert_rzcp_node_properties(result, 'test_sequence', 0)

    def test_lower_with_mixed_resource_types(self):
        """Test lowering with standard, custom, and argument resources."""
        # Create additional mock resources
        standard_mock = Mock(spec=AbstractResource)
        custom_mock = Mock(spec=AbstractResource)

        node = self.create_node(
            resource_specs={
                'standard_res': {'name': 'std_resource', 'type': 'standard'},
                'custom_res': {'name': 'custom_resource', 'type': 'custom'},
                'arg_res': {'name': 'arg_resource', 'type': 'argument'}
            }
        )

        resources = {
            'std_resource': standard_mock,
            'custom_resource': custom_mock
            # Note: arg_resource intentionally missing
        }

        result = node.lower(resources, self.mock_config)
        self.assertIsInstance(result, RZCPNode)
        self.assert_rzcp_node_properties(result, 'test_sequence', 0)

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

    def test_lower_with_all_resource_types_in_chain(self):
        """Test lowering chain where different nodes have different resource types."""
        # Node 1: standard resource
        node1 = self.create_node(
            block=0,
            resource_specs={'std': {'name': 'std_resource', 'type': 'standard'}}
        )

        # Node 2: custom resource
        node2 = self.create_node(
            block=1,
            resource_specs={'custom': {'name': 'custom_resource', 'type': 'custom'}}
        )

        # Node 3: argument resource
        node3 = self.create_node(
            block=2,
            resource_specs={'arg': {'name': 'arg_resource', 'type': 'argument'}}
        )

        # Link chain
        node1.next_zone = node2
        node2.next_zone = node3

        # Provide only standard and custom resources
        resources = {
            'std_resource': Mock(spec=AbstractResource),
            'custom_resource': Mock(spec=AbstractResource)
            # arg_resource intentionally missing
        }

        # Should succeed - argument resource validation skipped
        result = node1.lower(resources, self.mock_config)

        # Verify all three nodes were lowered
        self.assert_rzcp_node_properties(result, 'test_sequence', 0)
        self.assert_rzcp_node_properties(result.next_zone, 'test_sequence', 1)
        self.assert_rzcp_node_properties(result.next_zone.next_zone, 'test_sequence', 2)


class TestZCPNodeErrorHandling(BaseZCPNodeTest):
    """Test error handling and exception propagation."""

    def test_sampling_factory_error_context(self):
        """Test error context in sampling factory."""
        self.mock_construction_callback.side_effect = RuntimeError("Construction failed")

        node = self.create_node(sequence='error_sequence', block=5)
        sampling_fn = node._make_sampling_factory({})

        with self.assertRaises(Exception) as context:  # Could be GraphError or GraphLoweringError
            sampling_fn({})

        # Verify error context and cause
        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 5)

    def test_lower_chain_error_propagation(self):
        """Test error propagation when lowering chains."""
        # Create chain where second node will fail due to missing standard resource
        node1 = self.create_node(sequence='error_sequence', block=5, resource_specs={})
        node2 = self.create_node(
            sequence='error_sequence',
            block=6,
            resource_specs={'missing': {'name': 'missing_resource', 'arguments': None, 'type': 'standard'}}
        )

        node1.next_zone = node2

        # When lowering the chain, node2 should fail because 'missing_resource' is not in resources
        with self.assertRaises(GraphLoweringError) as context:
            node1.lower({}, self.mock_config)

        # Error should reference the failing node's context
        self.assertEqual(context.exception.sequence, "error_sequence")
        self.assertEqual(context.exception.block, 6)  # Should be node2's block

    def test_lower_error_for_mixed_missing_resources(self):
        """Test proper error reporting when some but not all required resources are missing."""
        node = self.create_node(
            resource_specs={
                'present': {'name': 'present_resource', 'type': 'standard'},
                'missing': {'name': 'missing_resource', 'type': 'custom'}
            }
        )

        # Provide only one of the two required resources
        partial_resources = {'present_resource': Mock(spec=AbstractResource)}

        with self.assertRaises(GraphLoweringError) as context:
            node.lower(partial_resources, self.mock_config)

        # Should mention the missing resource
        error_msg = str(context.exception).lower()
        self.assertIn("missing_resource", error_msg)

    def test_lower_succeeds_with_mixed_resources_when_all_required_present(self):
        """Test that lowering succeeds when all standard/custom resources are present."""
        node = self.create_node(
            resource_specs={
                'standard': {'name': 'std_resource', 'type': 'standard'},
                'custom': {'name': 'custom_resource', 'type': 'custom'},
                'argument': {'name': 'arg_resource', 'type': 'argument'}
            }
        )

        # Provide standard and custom resources (argument will be provided later)
        resources = {
            'std_resource': Mock(spec=AbstractResource),
            'custom_resource': Mock(spec=AbstractResource)
        }

        # Should succeed
        result = node.lower(resources, self.mock_config)
        self.assertIsInstance(result, RZCPNode)

    def test_argument_resource_error_context_in_sampling(self):
        """Test that missing argument resources produce good error messages during sampling."""
        node = self.create_node(
            resource_specs={'arg': {'name': 'arg_resource', 'type': 'argument'}}
        )

        # Lower successfully (no argument validation)
        result = node.lower({}, self.mock_config)

        # Sampling should fail with descriptive error
        with self.assertRaises(Exception) as context:
            result.sampling_callback({})  # No argument resources provided

        error_msg = str(context.exception).lower()
        self.assertIn("arg_resource", error_msg)


if __name__ == "__main__":
    unittest.main()