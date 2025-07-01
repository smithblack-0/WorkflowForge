"""
Unit tests for Scope class in the flow control system.

Tests cover:
1. Scope construction and dependency injection
2. Internal utility methods for resource and sequence handling
3. User commands for workflow building
"""

import unittest
import warnings
from unittest.mock import Mock
from typing import Dict

# Import the modules under test
from workflow_forge.frontend.flow_control.program import (
    Scope, FCFactories, ScopeException
)
from workflow_forge.zcp.builder import GraphBuilderNode
from workflow_forge.zcp.nodes import ZCPNode, RZCPNode
from workflow_forge.frontend.parsing.config_parsing import Config
from workflow_forge.resources import AbstractResource, StaticStringResource
from workflow_forge.frontend.flow_control.program import ConditionalContext, WhileContext


class ScopeTestResources(unittest.TestCase):
    """Base test class with common setup and helper methods."""

    def setUp(self):
        """Set up common test fixtures."""
        # Mock resources (need to be defined first)
        self.mock_string_resource = Mock(spec=StaticStringResource)
        self.mock_abstract_resource = Mock(spec=AbstractResource)

        # Mock factories
        self.mock_factories = Mock(spec=FCFactories)
        self.mock_factories.condition_context = Mock(return_value=Mock(spec=ConditionalContext))
        self.mock_factories.while_context = Mock(return_value=Mock(spec=WhileContext))
        self.mock_factories.scope = Mock()
        self.mock_factories.str_resource = Mock(return_value=self.mock_string_resource)

        # Mock config
        self.mock_config = Mock(spec=Config)
        self.mock_config.tools = ["search", "calculator"]
        self.mock_config.control_pattern = "[Jump]"

        # Mock builder and head
        self.mock_builder = Mock(spec=GraphBuilderNode)
        self.mock_head = Mock(spec=RZCPNode)

        # Mock program
        self.mock_program = Mock()
        self.mock_program.merge = Mock()

        # Mock sequences
        self.mock_zcp_node = Mock(spec=ZCPNode)
        self.mock_rzcp_node = Mock(spec=RZCPNode)
        self.mock_zcp_node.lower = Mock(return_value=self.mock_rzcp_node)

        # Test data
        self.test_resources = {
            "test_resource": self.mock_abstract_resource
        }
        self.test_sequences = {
            "test_sequence": self.mock_zcp_node
        }

        # Mock builder methods
        self.mock_builder.extend = Mock(return_value=Mock(spec=GraphBuilderNode))

    def create_scope(self, **overrides) -> Scope:
        """Create a Scope with valid data and optional overrides."""
        defaults = {
            'factories': self.mock_factories,
            'config': self.mock_config,
            'builder': self.mock_builder,
            'head': self.mock_head,
            'program': self.mock_program,
            'resources': self.test_resources.copy(),
            'sequences': self.test_sequences.copy()
        }
        defaults.update(overrides)
        return Scope(**defaults)

    def create_mock_resource_dict(self, **resources) -> Dict[str, AbstractResource]:
        """Create a dictionary of mock resources."""
        result = {}
        for name, value in resources.items():
            if isinstance(value, AbstractResource):
                result[name] = value
            else:
                mock_resource = Mock(spec=AbstractResource)
                result[name] = mock_resource
        return result


class TestScopeConstruction(ScopeTestResources):
    """Test Scope construction and dependency injection."""

    def test_can_be_created_with_all_dependencies(self):
        """Test that Scope can be created with all required dependencies."""
        scope = self.create_scope()

        self.assertIsInstance(scope, Scope)

    def test_factories_dependency_injection_works(self):
        """Test that factories are properly injected and accessible."""
        scope = self.create_scope()

        # Should be able to access factory methods through scope
        scope.when("test_sequence")
        self.mock_factories.condition_context.assert_called_once()

    def test_creation_with_empty_resources(self):
        """Test scope creation with empty resources dictionary."""
        scope = self.create_scope(resources={})

        self.assertIsInstance(scope, Scope)

    def test_creation_with_empty_sequences(self):
        """Test scope creation with empty sequences dictionary."""
        scope = self.create_scope(sequences={})

        self.assertIsInstance(scope, Scope)


class TestInternalUtils(ScopeTestResources):
    """Test internal utility methods for resource and sequence handling."""

    def test_fetch_resources_returns_copy_of_existing_resources(self):
        """Test that _fetch_resources returns copy of existing resources."""
        scope = self.create_scope()

        result = scope._fetch_resources({})

        self.assertEqual(result, self.test_resources)
        # Should be a copy, not the same object
        self.assertIsNot(result, self.test_resources)

    def test_fetch_resources_adds_extra_resources(self):
        """Test that _fetch_resources adds extra resources to the copy."""
        scope = self.create_scope()
        extra_resource = Mock(spec=AbstractResource)

        result = scope._fetch_resources({"extra": extra_resource})

        self.assertIn("extra", result)
        self.assertEqual(result["extra"], extra_resource)
        self.assertIn("test_resource", result)

    def test_fetch_resources_converts_python_objects_to_static_resources(self):
        """Test that _fetch_resources converts Python objects to StaticStringResource."""
        scope = self.create_scope()

        result = scope._fetch_resources({"string_val": "test_string"})

        self.mock_factories.str_resource.assert_called_once_with("test_string")
        self.assertEqual(result["string_val"], self.mock_string_resource)

    def test_fetch_resources_warns_on_overwrite(self):
        """Test that _fetch_resources warns when overwriting existing resources."""
        scope = self.create_scope()
        new_resource = Mock(spec=AbstractResource)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scope._fetch_resources({"test_resource": new_resource})

            self.assertEqual(len(w), 1)
            self.assertIn("overwritten", str(w[0].message))

    def test_fetch_resources_handles_conversion_failure(self):
        """Test that _fetch_resources handles resource conversion failures."""
        scope = self.create_scope()

        # Create an object that raises TypeError when str() is called on it
        class UnconvertibleObject:
            def __repr__(self):
                raise TypeError("Cannot convert to string")

        unconvertible = UnconvertibleObject()

        with self.assertRaises(ScopeException) as cm:
            scope._fetch_resources({"bad": unconvertible})

        self.assertIn("convert python resource to text resource", str(cm.exception))

    def test_load_sequence_finds_existing_sequence(self):
        """Test that _load_sequence finds and returns existing sequence."""
        scope = self.create_scope()
        resources = self.create_mock_resource_dict()

        result = scope._load_sequence("test_sequence", resources)

        self.assertEqual(result, self.mock_rzcp_node)

    def test_load_sequence_calls_lower_with_resources_and_config(self):
        """Test that _load_sequence calls lower() with resources and config."""
        scope = self.create_scope()
        resources = self.create_mock_resource_dict()

        scope._load_sequence("test_sequence", resources)

        self.mock_zcp_node.lower.assert_called_once_with(resources, self.mock_config)

    def test_load_sequence_raises_exception_for_missing_sequence(self):
        """Test that _load_sequence raises ScopeException for missing sequence."""
        scope = self.create_scope()
        resources = self.create_mock_resource_dict()

        with self.assertRaises(ScopeException) as cm:
            scope._load_sequence("nonexistent", resources)

        self.assertIn("not found in available sequences", str(cm.exception))

    def test_fork_creates_new_scope_with_new_builder(self):
        """Test that fork creates new scope with provided builder."""
        scope = self.create_scope()
        new_builder = Mock(spec=GraphBuilderNode)
        new_scope_mock = Mock(spec=Scope)
        self.mock_factories.scope.return_value = new_scope_mock

        result = scope.fork(new_builder)

        self.mock_factories.scope.assert_called_once_with(
            self.mock_factories,
            self.mock_config,
            new_builder,
            self.mock_head,
            self.mock_program,
            self.test_resources,
            self.test_sequences
        )
        self.assertEqual(result, new_scope_mock)

    def test_replace_builder_updates_builder(self):
        """Test that replace_builder updates the scope's builder."""
        scope = self.create_scope()
        new_builder = Mock(spec=GraphBuilderNode)

        scope.replace_builder(new_builder)

        self.assertEqual(scope.builder, new_builder)


class TestCommands(ScopeTestResources):
    """Test user command methods for workflow building."""

    def test_run_calls_sequence_loading_and_extends_builder(self):
        """Test that run() loads sequence and extends builder."""
        scope = self.create_scope()
        new_builder = Mock(spec=GraphBuilderNode)
        self.mock_builder.extend.return_value = new_builder

        scope.run("test_sequence")

        # Should call lower on the ZCP node with resources and config
        self.mock_zcp_node.lower.assert_called_once()
        # Should extend builder with the result
        self.mock_builder.extend.assert_called_once_with(self.mock_rzcp_node)
        self.assertEqual(scope.builder, new_builder)

    def test_run_with_extra_resources_includes_them(self):
        """Test that run() includes extra resources in sequence loading."""
        scope = self.create_scope()
        extra_resource = Mock(spec=AbstractResource)

        scope.run("test_sequence", extra=extra_resource)

        # Should call lower with resources that include the extra resource
        call_args = self.mock_zcp_node.lower.call_args
        passed_resources = call_args[0][0]  # First positional argument
        self.assertIn("extra", passed_resources)
        self.assertEqual(passed_resources["extra"], extra_resource)

    def test_when_creates_conditional_context_with_correct_parameters(self):
        """Test that when() creates ConditionalContext with correct parameters."""
        scope = self.create_scope()

        result = scope.when("test_sequence")

        self.mock_factories.condition_context.assert_called_once_with(
            scope, self.mock_builder, self.mock_rzcp_node
        )

    def test_loop_creates_while_context_with_correct_parameters(self):
        """Test that loop() creates WhileContext with correct parameters."""
        scope = self.create_scope()

        result = scope.loop("test_sequence")

        self.mock_factories.while_context.assert_called_once_with(
            scope, self.mock_builder, self.mock_rzcp_node
        )

    def test_capture_validates_tool_exists_in_config(self):
        """Test that capture() validates tool exists in config."""
        scope = self.create_scope()

        with self.assertRaises(ScopeException) as cm:
            scope.capture("test_sequence", "nonexistent_tool")

        self.assertIn("not found in available tools", str(cm.exception))

    def test_capture_sets_output_flag_and_tool_name_on_last_node(self):
        """Test that capture() sets output flag and tool name on sequence tail."""
        scope = self.create_scope()
        tail_node = Mock(spec=RZCPNode)
        self.mock_rzcp_node.get_last_node = Mock(return_value=tail_node)

        scope.capture("test_sequence", "search")

        self.assertTrue(tail_node.output)
        self.assertEqual(tail_node.tool_name, "search")

    def test_capture_extends_builder_with_sequence(self):
        """Test that capture() extends builder with the sequence."""
        scope = self.create_scope()
        tail_node = Mock(spec=RZCPNode)
        self.mock_rzcp_node.get_last_node = Mock(return_value=tail_node)
        new_builder = Mock(spec=GraphBuilderNode)
        self.mock_builder.extend.return_value = new_builder

        scope.capture("test_sequence", "search")

        self.mock_builder.extend.assert_called_once_with(self.mock_rzcp_node)
        self.assertEqual(scope.builder, new_builder)

    def test_feed_sets_input_flag_on_last_node(self):
        """Test that feed() sets input flag on sequence tail."""
        scope = self.create_scope()
        tail_node = Mock(spec=RZCPNode)
        self.mock_rzcp_node.get_last_node = Mock(return_value=tail_node)

        scope.feed("test_sequence")

        self.assertTrue(tail_node.input)

    def test_feed_extends_builder_with_sequence(self):
        """Test that feed() extends builder with the sequence."""
        scope = self.create_scope()
        tail_node = Mock(spec=RZCPNode)
        self.mock_rzcp_node.get_last_node = Mock(return_value=tail_node)
        new_builder = Mock(spec=GraphBuilderNode)
        self.mock_builder.extend.return_value = new_builder

        scope.feed("test_sequence")

        self.mock_builder.extend.assert_called_once_with(self.mock_rzcp_node)
        self.assertEqual(scope.builder, new_builder)

    def test_subroutine_extends_builder_with_subroutine_head(self):
        """Test that subroutine() extends builder with subroutine head."""
        scope = self.create_scope()
        subroutine_program = Mock()
        subroutine_head = Mock(spec=RZCPNode)
        subroutine_program.head = subroutine_head
        new_builder = Mock(spec=GraphBuilderNode)
        self.mock_builder.extend.return_value = new_builder

        scope.subroutine(subroutine_program)

        # Should extend with some sequence (the copied head)
        self.mock_builder.extend.assert_called_once()
        extended_sequence = self.mock_builder.extend.call_args[0][0]
        # Should be a copy, not the original
        self.assertIsNot(extended_sequence, subroutine_head)
        self.assertEqual(scope.builder, new_builder)

    def test_subroutine_merges_program_state(self):
        """Test that subroutine() merges the subroutine program into main program."""
        scope = self.create_scope()
        subroutine_program = Mock()
        subroutine_head = Mock(spec=RZCPNode)
        subroutine_program.head = subroutine_head

        scope.subroutine(subroutine_program)

        self.mock_program.merge.assert_called_once_with(subroutine_program)


if __name__ == "__main__":
    unittest.main()