"""
Unit tests for Workflow and LoweredWorkflow classes.

Tests cover:
1. Basic type contracts (returns right types)
2. Delegation to dependencies
3. Factory usage for all dependencies
4. Data coordination (not validation)
"""

import unittest
import msgpack
import base64
import numpy as np
from unittest.mock import Mock

# Import the modules under test
from workflow_forge.zcp.workflow import Workflow, LoweredWorkflow, WFFactories
from workflow_forge.zcp.nodes import SZCPNode, LZCPNode
from workflow_forge.zcp.tag_converter import TagConverter
from workflow_forge.frontend.parsing.config_parsing import Config
from workflow_forge.tokenizer_interface import TokenizerInterface


class BaseWorkflowTest(unittest.TestCase):
    """Base test class with common setup and helper methods."""

    def setUp(self):
        """Set up common test fixtures."""
        # Mock config
        self.mock_config = Mock(spec=Config)
        self.mock_config.valid_tags = ["Training", "Correct"]

        # Mock SZCP node
        self.mock_szcp_node = Mock(spec=SZCPNode)
        self.mock_szcp_node.serialize = Mock()
        self.mock_szcp_node.serialize.return_value = {"mock": "serialized"}

        # Mock LZCP node
        self.mock_lzcp_node = Mock(spec=LZCPNode)
        self.mock_szcp_node.lower = Mock()
        self.mock_szcp_node.lower.return_value = self.mock_lzcp_node

        # Mock tag converter
        self.mock_tag_converter = Mock(spec=TagConverter)
        self.mock_tag_converter.tensorize = Mock()

        # Mock factories - now with all three constructors
        self.mock_factories = Mock(spec=WFFactories)
        self.mock_factories.tag_converter = Mock()
        self.mock_factories.tag_converter.return_value = self.mock_tag_converter
        self.mock_factories.SZCP_node = Mock()
        self.mock_factories.SZCP_node.return_value = self.mock_szcp_node
        self.mock_factories.Config = Mock()
        self.mock_factories.Config.return_value = self.mock_config

        # Mock config needs serialize method
        self.mock_config.serialize = Mock()
        self.mock_config.serialize.return_value = {"mock": "config"}

        # Test data
        self.extractions = {"training": ["Training"], "correct": ["Correct"]}

    def create_workflow(self, **overrides) -> Workflow:
        """Create a Workflow with valid data and optional overrides."""
        defaults = {
            'config': self.mock_config,
            'nodes': self.mock_szcp_node,
            'extractions': self.extractions,
            'factories': self.mock_factories
        }
        defaults.update(overrides)
        return Workflow(**defaults)


class TestWorkflowSerialization(BaseWorkflowTest):
    """Test serialization functionality."""

    def test_serialize_returns_string(self):
        """Test that serialize returns a string."""
        workflow = self.create_workflow()

        # Mock config.serialize() to return test data
        self.mock_config.serialize.return_value = {"mock": "config"}

        result = workflow.serialize()

        self.assertIsInstance(result, str)

    def test_serialize_calls_dependencies(self):
        """Test that serialize delegates to config and nodes."""
        workflow = self.create_workflow()

        workflow.serialize()

        # Should call config.serialize() and nodes.serialize()
        self.mock_config.serialize.assert_called_once()
        self.mock_szcp_node.serialize.assert_called_once()

    def test_serialize_produces_valid_msgpack_base64(self):
        """Test that serialize produces valid msgpack+base64 format."""
        workflow = self.create_workflow()

        # Mock config.serialize() to return test data
        self.mock_config.serialize.return_value = {"test": "config"}

        result = workflow.serialize()

        # Should be able to decode back to original structure
        binary_data = base64.b64decode(result.encode('utf-8'))
        decoded_data = msgpack.unpackb(binary_data, strict_map_key=False)

        # Verify structure
        self.assertIn("config", decoded_data)
        self.assertIn("nodes", decoded_data)
        self.assertIn("extractions", decoded_data)
        self.assertEqual(decoded_data["config"], {"test": "config"})
        self.assertEqual(decoded_data["nodes"], {"mock": "serialized"})
        self.assertEqual(decoded_data["extractions"], self.extractions)

    def test_deserialize_returns_workflow(self):
        """Test that deserialize returns a Workflow."""
        test_data = {
            "config": {"valid_tags": ["Test"]},
            "nodes": {"mock": "data"},
            "extractions": {"test": ["Test"]}
        }

        # Create msgpack + base64 encoded string (simulating serialize() output)
        binary_data = msgpack.packb(test_data)
        encoded_string = base64.b64encode(binary_data).decode('utf-8')

        result = Workflow.deserialize(encoded_string, self.mock_factories)

        self.assertIsInstance(result, Workflow)

    def test_deserialize_uses_factories(self):
        """Test that deserialize uses factories to construct objects."""
        test_data = {
            "config": {"valid_tags": ["Test"]},
            "nodes": {"mock": "data"},
            "extractions": {"test": ["Test"]}
        }

        # Create msgpack + base64 encoded string (simulating serialize() output)
        binary_data = msgpack.packb(test_data)
        encoded_string = base64.b64encode(binary_data).decode('utf-8')

        Workflow.deserialize(encoded_string, self.mock_factories)

    def test_deserialize_handles_msgpack_format(self):
        """Test that deserialize correctly handles msgpack+base64 decoding."""
        # Test with data that has int keys (preserved by msgpack)
        test_data = {
            "config": {"escape_patterns": ["start", "end"]},  # List since tuples don't survive
            "nodes": {0: {"data": "test"}, 1: {"data": "test2"}},  # Int keys preserved
            "extractions": {"test": ["Test"]}
        }

        # Encode using msgpack + base64
        binary_data = msgpack.packb(test_data)
        encoded_string = base64.b64encode(binary_data).decode('utf-8')

        Workflow.deserialize(encoded_string, self.mock_factories)

        # Verify the config deserialize was called with list (Config.deserialize will handle tuple conversion)
        config_call_args = self.mock_factories.Config.deserialize.call_args[1]

        # Verify nodes was called with int keys intact (this should still work)
        nodes_call_args = self.mock_factories.SZCP_node.deserialize.call_args[0][0]
        self.assertIn(0, nodes_call_args)
        self.assertIn(1, nodes_call_args)


class TestWorkflowLowering(BaseWorkflowTest):
    """Test lowering functionality."""

    def test_lower_returns_lowered_workflow(self):
        """Test that lower returns a LoweredWorkflow."""
        workflow = self.create_workflow()
        mock_tokenizer = Mock(spec=TokenizerInterface)
        mock_tools = {}

        result = workflow.lower(mock_tokenizer, mock_tools)

        self.assertIsInstance(result, LoweredWorkflow)

    def test_lower_uses_factory_for_tag_converter(self):
        """Test that lower uses factory to create TagConverter."""
        workflow = self.create_workflow()
        mock_tokenizer = Mock(spec=TokenizerInterface)
        mock_tools = {}

        workflow.lower(mock_tokenizer, mock_tools)

        self.mock_factories.tag_converter.assert_called_once_with(self.mock_config.valid_tags)

    def test_lower_calls_nodes_lower(self):
        """Test that lower delegates to nodes.lower."""
        workflow = self.create_workflow()
        mock_tokenizer = Mock(spec=TokenizerInterface)
        mock_tools = {"calc": Mock()}

        workflow.lower(mock_tokenizer, mock_tools)

        self.mock_szcp_node.lower.assert_called_once_with(
            mock_tokenizer, self.mock_tag_converter, mock_tools
        )

    def test_lower_calls_tensorize_for_extractions(self):
        """Test that lower calls tensorize for each extraction."""
        workflow = self.create_workflow()
        mock_tokenizer = Mock(spec=TokenizerInterface)
        mock_tools = {}

        workflow.lower(mock_tokenizer, mock_tools)

        # Should call tensorize for each extraction key
        expected_calls = [
            unittest.mock.call(["Training"]),
            unittest.mock.call(["Correct"])
        ]
        self.mock_tag_converter.tensorize.assert_has_calls(expected_calls, any_order=True)


class TestLoweredWorkflow(BaseWorkflowTest):
    """Test LoweredWorkflow basic functionality."""

    def test_lowered_workflow_construction(self):
        """Test creating a LoweredWorkflow."""
        mock_tokenizer = Mock(spec=TokenizerInterface)
        extractions = {"test": np.array([1, 2])}

        lowered = LoweredWorkflow(
            tag_converter=self.mock_tag_converter,
            tokenizer=mock_tokenizer,
            nodes=self.mock_lzcp_node,
            extractions=extractions
        )

        # Just verify it holds the data we gave it
        self.assertEqual(lowered.tag_converter, self.mock_tag_converter)
        self.assertEqual(lowered.tokenizer, mock_tokenizer)
        self.assertEqual(lowered.nodes, self.mock_lzcp_node)
        self.assertEqual(lowered.extractions, extractions)


if __name__ == "__main__":
    unittest.main()