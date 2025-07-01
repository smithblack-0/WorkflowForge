"""
Unit tests for Program class in the flow control system.

Tests cover:
1. Program construction and initialization
2. Extraction management and tag validation
3. Compilation to workflow factories
4. Scope passthrough methods
"""

import unittest
from unittest.mock import Mock
from typing import Dict, Any, Callable

# Import the modules under test
from workflow_forge.frontend.flow_control.program import (
    Program, FCFactories, ProgramException
)
from workflow_forge.zcp.builder import GraphBuilderNode
from workflow_forge.zcp.nodes import ZCPNode, RZCPNode
from workflow_forge.zcp.workflow import Workflow
from workflow_forge.frontend.parsing.config_parsing import Config
from workflow_forge.resources import AbstractResource, StaticStringResource
from workflow_forge.frontend.flow_control.program import Scope, ConditionalContext, WhileContext


class ProgramTestResources(unittest.TestCase):
    """Base test class with common setup and helper methods."""

    def setUp(self):
        """Set up common test fixtures."""
        # Mock resources and workflow (need to be defined first)
        self.mock_resource = Mock(spec=AbstractResource)
        self.mock_string_resource = Mock(spec=StaticStringResource)
        self.mock_workflow = Mock(spec=Workflow)

        # Mock factories
        self.mock_factories = Mock(spec=FCFactories)
        self.mock_factories.graph_builder = Mock()
        self.mock_factories.scope = Mock()
        self.mock_factories.str_resource = Mock(return_value=self.mock_string_resource)
        self.mock_factories.workflow = Mock(return_value=self.mock_workflow)

        # Mock config
        self.mock_config = Mock(spec=Config)
        self.mock_config.valid_tags = ["Training", "Correct", "Incorrect"]
        self.mock_config.control_pattern = "[Jump]"

        # Mock sequences
        self.mock_zcp_node = Mock(spec=ZCPNode)
        self.test_sequences = {
            "test_sequence": self.mock_zcp_node,
            "another_sequence": Mock(spec=ZCPNode)
        }

        # Test resources
        self.test_resources = {
            "test_resource": self.mock_resource
        }

        # Mock scope and builder
        self.mock_scope = Mock(spec=Scope)
        self.mock_builder = Mock(spec=GraphBuilderNode)
        self.mock_head = Mock(spec=RZCPNode)

        # Configure factories
        self.mock_factories.graph_builder.return_value = self.mock_builder
        self.mock_factories.scope.return_value = self.mock_scope

        # Mock scope structure - Program expects scope to have .head
        self.mock_scope.head = self.mock_head
        self.mock_scope.builder = self.mock_builder
        self.mock_head.next_zone = Mock(spec=RZCPNode)

        # Mock workflow and SZCP
        self.mock_szcp = Mock()
        self.mock_head.next_zone.lower = Mock(return_value=self.mock_szcp)

    def create_program(self, **overrides) -> Program:
        """Create a Program with valid data and optional overrides."""
        defaults = {
            'factories': self.mock_factories,
            'sequences': self.test_sequences.copy(),
            'resources': self.test_resources.copy(),
            'config': self.mock_config
        }
        defaults.update(overrides)
        return Program(**defaults)

    def create_mock_workflow_factory(self) -> Callable:
        """Create a mock workflow factory function."""

        def mock_factory(resources: Dict[str, Any]) -> Workflow:
            return self.mock_workflow

        return Mock(side_effect=mock_factory)


class TestProgramConstruction(ProgramTestResources):
    """Test Program construction and initialization."""

    def test_can_be_created_with_all_dependencies(self):
        """Test that Program can be created with all required dependencies."""
        program = self.create_program()

        self.assertIsInstance(program, Program)

    def test_creates_initial_graph_builder(self):
        """Test that Program creates initial GraphBuilder with control pattern."""
        program = self.create_program()

        self.mock_factories.graph_builder.assert_called_once_with(
            self.mock_config.control_pattern,
            unittest.mock.ANY  # The head list
        )

    def test_creates_scope_with_correct_parameters(self):
        """Test that Program creates Scope with all required parameters."""
        program = self.create_program()

        self.mock_factories.scope.assert_called_once_with(
            factories=self.mock_factories,
            config=self.mock_config,
            builder=self.mock_builder,
            head=unittest.mock.ANY,  # The placeholder head
            program=program,
            resources=self.test_resources,
            sequences=self.test_sequences
        )

    def test_initializes_empty_extractions_dict(self):
        """Test that Program initializes with empty extractions dictionary."""
        program = self.create_program()

        self.assertEqual(program.extractions, {})


class TestExtractionManagement(ProgramTestResources):
    """Test extraction management and tag validation."""

    def test_extract_adds_extraction_with_valid_tags(self):
        """Test that extract() adds extraction configuration with valid tags."""
        program = self.create_program()

        program.extract("training_data", ["Training", "Correct"])

        self.assertEqual(program.extractions["training_data"], ["Training", "Correct"])

    def test_extract_validates_tags_exist_in_config(self):
        """Test that extract() validates all tags exist in config."""
        program = self.create_program()

        with self.assertRaises(ProgramException) as cm:
            program.extract("invalid_data", ["InvalidTag"])

        self.assertIn("Invalid tag 'InvalidTag' not in config.valid_tags", str(cm.exception))

    def test_extract_prevents_duplicate_extraction_names(self):
        """Test that extract() prevents duplicate extraction names."""
        program = self.create_program()
        program.extract("data", ["Training"])

        with self.assertRaises(ProgramException) as cm:
            program.extract("data", ["Correct"])

        self.assertIn("Already specified an extract to that name", str(cm.exception))

    def test_extract_allows_multiple_different_extractions(self):
        """Test that extract() allows multiple extractions with different names."""
        program = self.create_program()

        program.extract("training", ["Training"])
        program.extract("correct", ["Correct"])

        self.assertEqual(program.extractions["training"], ["Training"])
        self.assertEqual(program.extractions["correct"], ["Correct"])

    def test_merge_combines_extractions_from_other_program(self):
        """Test that merge() combines extractions from another program."""
        program = self.create_program()
        other_program = Mock()
        other_program.extractions = {"other_data": ["Training", "Incorrect"]}

        program.merge(other_program)

        self.assertEqual(program.extractions["other_data"], ["Training", "Incorrect"])

    def test_merge_updates_existing_extractions(self):
        """Test that merge() updates existing extractions."""
        program = self.create_program()
        program.extract("data", ["Training"])

        other_program = Mock()
        other_program.extractions = {"data": ["Correct"], "new_data": ["Incorrect"]}

        program.merge(other_program)

        self.assertEqual(program.extractions["data"], ["Correct"])
        self.assertEqual(program.extractions["new_data"], ["Incorrect"])


class TestCompilation(ProgramTestResources):
    """Test compilation to workflow factories."""

    def test_compile_raises_exception_for_empty_program(self):
        """Test that compile() raises exception when no sequences were added."""
        program = self.create_program()
        # Mock empty program (no next_zone)
        self.mock_head.next_zone = None

        with self.assertRaises(ProgramException) as cm:
            program.compile()

        self.assertIn("Cannot compile empty program", str(cm.exception))

    def test_compile_returns_workflow_factory_function(self):
        """Test that compile() returns a callable workflow factory."""
        program = self.create_program()

        factory = program.compile()

        self.assertTrue(callable(factory))

    def test_workflow_factory_converts_argument_resources(self):
        """Test that workflow factory converts argument resources to AbstractResource."""
        program = self.create_program()
        factory = program.compile()

        factory({"string_arg": "test_value"})

        self.mock_factories.str_resource.assert_called_with("test_value")

    def test_workflow_factory_calls_lower_with_converted_resources(self):
        """Test that workflow factory calls lower() with converted resources."""
        program = self.create_program()
        factory = program.compile()

        factory({"test_arg": "value"})

        # Should call lower with the converted resources
        expected_resources = {"test_arg": self.mock_string_resource}
        self.mock_head.next_zone.lower.assert_called_once_with(expected_resources)

    def test_workflow_factory_creates_workflow_with_correct_parameters(self):
        """Test that workflow factory creates Workflow with correct parameters."""
        program = self.create_program()
        program.extract("test_data", ["Training"])
        factory = program.compile()

        result = factory({})

        self.mock_factories.workflow.assert_called_once_with(
            config=self.mock_config,
            nodes=self.mock_szcp,
            extractions={"test_data": ["Training"]}
        )
        self.assertEqual(result, self.mock_workflow)

    def test_convert_resources_handles_conversion_failure(self):
        """Test that _convert_resources handles conversion failures."""
        program = self.create_program()

        # Create an object that raises TypeError when str() is called on it
        class UnconvertibleObject:
            def __str__(self):
                raise TypeError("Cannot convert to string")

        unconvertible = UnconvertibleObject()

        with self.assertRaises(ProgramException) as cm:
            program._convert_resources({"bad": unconvertible})

        self.assertIn("Conversion of python into resources failed", str(cm.exception))

    def test_convert_resources_converts_all_values_to_static_resources(self):
        """Test that _convert_resources converts all values to StaticStringResource."""
        program = self.create_program()

        result = program._convert_resources({"key1": "value1", "key2": 123})

        # Should call str_resource factory for each conversion
        self.assertEqual(self.mock_factories.str_resource.call_count, 2)
        self.mock_factories.str_resource.assert_any_call("value1")
        self.mock_factories.str_resource.assert_any_call("123")

        # Should return dict with converted resources
        expected_result = {"key1": self.mock_string_resource, "key2": self.mock_string_resource}
        self.assertEqual(result, expected_result)


class TestScopePassthroughs(ProgramTestResources):
    """Test scope passthrough methods."""

    def test_run_delegates_to_scope(self):
        """Test that run() delegates to internal scope."""
        program = self.create_program()
        extra_resource = Mock(spec=AbstractResource)

        program.run("test_sequence", extra=extra_resource)

        self.mock_scope.run.assert_called_once_with("test_sequence", extra=extra_resource)

    def test_when_delegates_to_scope_and_returns_result(self):
        """Test that when() delegates to scope and returns the result."""
        program = self.create_program()
        mock_context = Mock(spec=ConditionalContext)
        self.mock_scope.when.return_value = mock_context

        result = program.when("test_sequence")

        self.mock_scope.when.assert_called_once_with("test_sequence")
        self.assertEqual(result, mock_context)

    def test_loop_delegates_to_scope_and_returns_result(self):
        """Test that loop() delegates to scope and returns the result."""
        program = self.create_program()
        mock_context = Mock(spec=WhileContext)
        self.mock_scope.loop.return_value = mock_context

        result = program.loop("test_sequence")

        self.mock_scope.loop.assert_called_once_with("test_sequence")
        self.assertEqual(result, mock_context)

    def test_subroutine_delegates_to_scope(self):
        """Test that subroutine() delegates to internal scope."""
        program = self.create_program()
        other_program = Mock(spec=Program)

        program.subroutine(other_program)

        self.mock_scope.subroutine.assert_called_once_with(other_program)

    def test_capture_delegates_to_scope(self):
        """Test that capture() delegates to internal scope."""
        program = self.create_program()
        mock_tool = Mock()

        program.capture("test_sequence", mock_tool)

        self.mock_scope.capture.assert_called_once_with("test_sequence", mock_tool)

    def test_feed_delegates_to_scope(self):
        """Test that feed() delegates to internal scope."""
        program = self.create_program()

        program.feed("test_sequence")

        self.mock_scope.feed.assert_called_once_with("test_sequence")

    def test_passthrough_methods_forward_extra_resources(self):
        """Test that passthrough methods forward extra resource arguments."""
        program = self.create_program()
        extra_resource = Mock(spec=AbstractResource)
        mock_tool = Mock()

        program.when("test_sequence", extra=extra_resource)
        program.loop("test_sequence", extra=extra_resource)
        program.capture("test_sequence", mock_tool, extra=extra_resource)
        program.feed("test_sequence", extra=extra_resource)

        self.mock_scope.when.assert_called_with("test_sequence", extra=extra_resource)
        self.mock_scope.loop.assert_called_with("test_sequence", extra=extra_resource)
        self.mock_scope.capture.assert_called_with("test_sequence", mock_tool, extra=extra_resource)
        self.mock_scope.feed.assert_called_with("test_sequence", extra=extra_resource)


if __name__ == "__main__":
    unittest.main()