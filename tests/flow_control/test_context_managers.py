"""
Unit tests for Context Manager classes in the flow control system.

Tests cover:
1. ConditionalContext - if/else flow control context management
2. WhileContext - loop flow control context management
3. Basic functionality and method calls only - no integration testing
"""

import unittest
from unittest.mock import Mock

# Import the modules under test
from workflow_forge.frontend.flow_control.program import (
    ConditionalContext, WhileContext
)
from workflow_forge.zcp.builder import GraphBuilderNode
from workflow_forge.zcp.nodes import RZCPNode
from workflow_forge.frontend.flow_control.program import Scope


class BaseContextManagerTest(unittest.TestCase):
    """Base test class with common setup and helper methods."""

    def setUp(self):
        """Set up common test fixtures."""
        # Mock RZCP sequence node
        self.mock_sequence = Mock(spec=RZCPNode)
        self.mock_sequence.sequence = "test_sequence"

        # Mock GraphBuilderNode instances
        self.mock_intake_node = Mock(spec=GraphBuilderNode)
        self.mock_if_branch = Mock(spec=GraphBuilderNode)
        self.mock_else_branch = Mock(spec=GraphBuilderNode)
        self.mock_loop_branch = Mock(spec=GraphBuilderNode)
        self.mock_escape_branch = Mock(spec=GraphBuilderNode)
        self.mock_merged_builder = Mock(spec=GraphBuilderNode)

        # Mock parent scope and child scopes
        self.mock_parent_scope = Mock(spec=Scope)
        self.mock_if_scope = Mock(spec=Scope)
        self.mock_else_scope = Mock(spec=Scope)
        self.mock_loop_scope = Mock(spec=Scope)

        # Configure mock behaviors
        self.mock_intake_node.fork = Mock(return_value=(self.mock_if_branch, self.mock_else_branch))
        self.mock_parent_scope.fork = Mock(side_effect=[self.mock_if_scope, self.mock_else_scope])
        self.mock_parent_scope.replace_builder = Mock()

        # Mock GraphBuilderNode class methods
        GraphBuilderNode.merge = Mock(return_value=self.mock_merged_builder)

        # Configure scope builders
        self.mock_if_scope.builder = self.mock_if_branch
        self.mock_else_scope.builder = self.mock_else_branch
        self.mock_loop_scope.builder = self.mock_loop_branch

    def create_conditional_context(self, **overrides) -> ConditionalContext:
        """Create a ConditionalContext with valid data and optional overrides."""
        defaults = {
            'parent_scope': self.mock_parent_scope,
            'intake_node': self.mock_intake_node,
            'sequence': self.mock_sequence
        }
        defaults.update(overrides)
        return ConditionalContext(**defaults)

    def create_while_context(self, **overrides) -> WhileContext:
        """Create a WhileContext with valid data and optional overrides."""
        defaults = {
            'parent_scope': self.mock_parent_scope,
            'intake_node': self.mock_intake_node,
            'sequence': self.mock_sequence
        }
        defaults.update(overrides)
        return WhileContext(**defaults)


class TestConditionalContext(BaseContextManagerTest):
    """Test ConditionalContext for if/else flow control."""

    def test_can_be_created(self):
        """Test that ConditionalContext can be created without errors."""
        context = self.create_conditional_context()
        self.assertIsInstance(context, ConditionalContext)

    def test_enter_calls_fork_with_sequence(self):
        """Test that __enter__ calls fork on intake node with the sequence."""
        context = self.create_conditional_context()

        context.__enter__()

        self.mock_intake_node.fork.assert_called_once_with(self.mock_sequence)

    def test_enter_creates_scopes_from_branches(self):
        """Test that __enter__ creates if and else scopes from forked branches."""
        context = self.create_conditional_context()

        context.__enter__()

        # Should call parent_scope.fork twice, once for each branch
        self.assertEqual(self.mock_parent_scope.fork.call_count, 2)
        self.mock_parent_scope.fork.assert_any_call(self.mock_if_branch)
        self.mock_parent_scope.fork.assert_any_call(self.mock_else_branch)

    def test_enter_returns_if_and_else_scopes(self):
        """Test that __enter__ returns tuple of (if_scope, else_scope)."""
        context = self.create_conditional_context()

        result = context.__enter__()

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        if_scope, else_scope = result
        self.assertEqual(if_scope, self.mock_if_scope)
        self.assertEqual(else_scope, self.mock_else_scope)

    def test_exit_calls_merge_with_scope_builders(self):
        """Test that __exit__ calls merge with the builders from both scopes."""
        context = self.create_conditional_context()
        context.__enter__()

        context.__exit__(None, None, None)

        GraphBuilderNode.merge.assert_called_once_with(self.mock_if_branch, self.mock_else_branch)

    def test_exit_calls_replace_builder_with_merged_result(self):
        """Test that __exit__ calls replace_builder with merged builder."""
        context = self.create_conditional_context()
        context.__enter__()

        context.__exit__(None, None, None)

        self.mock_parent_scope.replace_builder.assert_called_once_with(self.mock_merged_builder)

    def test_exit_returns_none(self):
        """Test that __exit__ returns None (doesn't suppress exceptions)."""
        context = self.create_conditional_context()
        context.__enter__()

        result = context.__exit__(None, None, None)

        self.assertIsNone(result)

    def test_exit_with_exception_returns_none(self):
        """Test that __exit__ returns None even when exception info is provided."""
        context = self.create_conditional_context()
        context.__enter__()

        result = context.__exit__(ValueError, ValueError("test"), None)

        self.assertIsNone(result)

    def test_fork_exception_propagates(self):
        """Test that exceptions from fork are propagated."""
        self.mock_intake_node.fork = Mock(side_effect=Exception("Fork failed"))
        context = self.create_conditional_context()

        with self.assertRaises(Exception) as cm:
            context.__enter__()

        self.assertEqual(str(cm.exception), "Fork failed")

    def test_parent_scope_fork_exception_propagates(self):
        """Test that exceptions from parent scope fork are propagated."""
        self.mock_parent_scope.fork = Mock(side_effect=Exception("Scope creation failed"))
        context = self.create_conditional_context()

        with self.assertRaises(Exception) as cm:
            context.__enter__()

        self.assertEqual(str(cm.exception), "Scope creation failed")


class TestWhileContext(BaseContextManagerTest):
    """Test WhileContext for loop flow control."""

    def setUp(self):
        """Set up additional mocks for while context testing."""
        super().setUp()

        # Configure fork to return loop and escape branches for while context
        self.mock_intake_node.fork = Mock(return_value=(self.mock_loop_branch, self.mock_escape_branch))
        self.mock_parent_scope.fork = Mock(return_value=self.mock_loop_scope)

        # Mock attach method
        self.mock_intake_node.attach = Mock()

    def test_can_be_created(self):
        """Test that WhileContext can be created without errors."""
        context = self.create_while_context()
        self.assertIsInstance(context, WhileContext)

    def test_enter_calls_fork_with_sequence(self):
        """Test that __enter__ calls fork on intake node with the sequence."""
        context = self.create_while_context()

        context.__enter__()

        self.mock_intake_node.fork.assert_called_once_with(self.mock_sequence)

    def test_enter_creates_loop_scope_from_loop_branch(self):
        """Test that __enter__ creates loop scope from the loop branch."""
        context = self.create_while_context()

        context.__enter__()

        self.mock_parent_scope.fork.assert_called_once_with(self.mock_loop_branch)

    def test_enter_returns_loop_scope(self):
        """Test that __enter__ returns the loop scope."""
        context = self.create_while_context()

        result = context.__enter__()

        self.assertEqual(result, self.mock_loop_scope)

    def test_exit_calls_attach_with_loop_builder(self):
        """Test that __exit__ calls attach to create loopback."""
        context = self.create_while_context()
        context.__enter__()

        context.__exit__(None, None, None)

        self.mock_intake_node.attach.assert_called_once_with(self.mock_loop_branch)

    def test_exit_calls_replace_builder_with_escape_branch(self):
        """Test that __exit__ calls replace_builder with escape branch."""
        context = self.create_while_context()
        context.__enter__()

        context.__exit__(None, None, None)

        self.mock_parent_scope.replace_builder.assert_called_once_with(self.mock_escape_branch)

    def test_exit_returns_none(self):
        """Test that __exit__ returns None (doesn't suppress exceptions)."""
        context = self.create_while_context()
        context.__enter__()

        result = context.__exit__(None, None, None)

        self.assertIsNone(result)

    def test_exit_with_exception_returns_none(self):
        """Test that __exit__ returns None even when exception info is provided."""
        context = self.create_while_context()
        context.__enter__()

        result = context.__exit__(ValueError, ValueError("test"), None)

        self.assertIsNone(result)

    def test_fork_exception_propagates(self):
        """Test that exceptions from fork are propagated."""
        self.mock_intake_node.fork = Mock(side_effect=Exception("Fork failed"))
        context = self.create_while_context()

        with self.assertRaises(Exception) as cm:
            context.__enter__()

        self.assertEqual(str(cm.exception), "Fork failed")

    def test_attach_exception_propagates(self):
        """Test that exceptions from attach are propagated."""
        self.mock_intake_node.attach = Mock(side_effect=Exception("Attach failed"))
        context = self.create_while_context()
        context.__enter__()

        with self.assertRaises(Exception) as cm:
            context.__exit__(None, None, None)

        self.assertEqual(str(cm.exception), "Attach failed")


if __name__ == "__main__":
    unittest.main()