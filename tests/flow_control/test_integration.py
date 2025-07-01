"""
Integration tests for flow control system.

Tests cover:
1. Real workflow assembly using new_program()
2. Control structure integration (loops, conditionals, sequences)
3. End-to-end workflow building and compilation
4. Complex nested workflow scenarios
"""

import unittest
from typing import Dict, List

# Import the modules under test
from workflow_forge.frontend.flow_control.program import new_program, Program
from workflow_forge.zcp.nodes import ZCPNode
from workflow_forge.resources import AbstractResource
from workflow_forge.frontend.parsing.config_parsing import Config

# Set to True to enable visualization of workflow graphs during testing
SHOW_VISUALIZATIONS = True


class IntegrationTestResources(unittest.TestCase):
    """Base test class with common setup and helper methods for integration testing."""

    def setUp(self):
        """Set up common test fixtures for integration testing."""
        # Create real config (as suggested)
        self.config = Config(
            zone_patterns=["[Prompt]", "[Answer]", "[EOS]"],
            required_patterns=["[Prompt]", "[Answer]"],
            valid_tags=["Training", "Correct", "Incorrect", "Feedback"],
            default_max_token_length=1000,
            sequences=["setup", "thinking", "loop_control", "conclusion", "decision"],
            control_pattern="[Jump]",
            escape_patterns=("[Escape]", "[EndEscape]"),
            tools=["search", "calculator"],
            misc={}
        )

    def create_zcp_chain(self, sequence_name: str, num_zones: int = 3) -> ZCPNode:
        """
        Create a chain of linked ZCPNodes representing zones in a sequence.

        Args:
            sequence_name: Name of the sequence
            num_zones: Number of zones in the chain

        Returns:
            Head node of the ZCP chain
        """
        nodes = []

        zone_patterns = ["[Prompt]", "[Answer]", "[EOS]", "[Extra]"]

        for i in range(num_zones):
            # Create real construction callback
            def make_callback(zone_text):
                return lambda resources: zone_text

            construction_callback = make_callback(f"Resolved text for {sequence_name} zone {i}")

            # Create real ZCPNode
            node = ZCPNode(
                sequence=sequence_name,
                block=0,  # Same block, different zones
                construction_callback=construction_callback,
                resource_specs={},  # No placeholders for simplicity
                raw_text=f"Zone {i} text for {sequence_name}",
                zone_advance_str=zone_patterns[i] if i < len(zone_patterns) else f"[Zone{i}]",
                tags=["Training"] if i % 2 == 0 else ["Correct"],
                timeout=1000
            )

            nodes.append(node)

        # Link them into a chain
        for i in range(len(nodes) - 1):
            nodes[i].next_zone = nodes[i + 1]

        return nodes[0]  # Return head of chain

    def create_sequences(self, sequence_configs: Dict[str, int] = None) -> Dict[str, ZCPNode]:
        """
        Create a dictionary of ZCP sequences for testing.

        Args:
            sequence_configs: Dict mapping sequence names to number of zones

        Returns:
            Dictionary of sequence name to ZCP chain head
        """
        if sequence_configs is None:
            sequence_configs = {
                "setup": 3,          # [Prompt] → [Answer] → [EOS]
                "thinking": 2,       # [Prompt] → [Answer]
                "loop_control": 2,   # [Prompt] → [Answer]
                "conclusion": 3,     # [Prompt] → [Answer] → [EOS]
                "decision": 2        # [Prompt] → [Answer]
            }

        sequences = {}
        for name, num_zones in sequence_configs.items():
            sequences[name] = self.create_zcp_chain(name, num_zones)

        return sequences

    def create_resources(self, resource_names: List[str] = None) -> Dict[str, AbstractResource]:
        """
        Create a dictionary of real resources for testing.

        Args:
            resource_names: List of resource names to create

        Returns:
            Dictionary of resource name to real AbstractResource
        """
        from workflow_forge.resources import StaticStringResource

        if resource_names is None:
            resource_names = ["constitution", "feedback", "examples", "context"]

        resources = {}
        for name in resource_names:
            resources[name] = StaticStringResource(f"Resource value for {name}")

        return resources

    def create_program(self,
                      sequence_configs: Dict[str, int] = None,
                      resource_names: List[str] = None) -> Program:
        """
        Create a Program with mock sequences and resources for testing.

        Args:
            sequence_configs: Dict mapping sequence names to number of zones
            resource_names: List of resource names to create

        Returns:
            Configured Program instance ready for testing
        """
        sequences = self.create_sequences(sequence_configs)
        resources = self.create_resources(resource_names)

        return new_program(sequences, resources, self.config)

    def assert_program_can_compile(self, program: Program):
        """Assert that a program can compile without errors."""
        try:
            factory = program.compile()
            self.assertTrue(callable(factory))
        except Exception as e:
            self.fail(f"Program compilation failed: {e}")

    def assert_workflow_factory_works(self, program: Program):
        """Assert that a compiled workflow factory can be executed."""
        factory = program.compile()

        try:
            workflow = factory({})  # Execute with no additional resources
            self.assertIsNotNone(workflow)
        except Exception as e:
            self.fail(f"Workflow factory execution failed: {e}")


class TestBasicWorkflowAssembly(IntegrationTestResources):
    """Test basic workflow assembly and compilation."""

    def test_simple_linear_workflow(self):
        """Test creating and compiling a simple linear workflow."""
        program = self.create_program()

        # Build simple linear workflow
        program.run("setup")
        program.run("thinking")
        program.run("conclusion")

        # Should be able to compile
        self.assert_program_can_compile(program)

    def test_workflow_with_extraction(self):
        """Test workflow with extraction configuration."""
        program = self.create_program()

        # Build workflow
        program.run("setup")
        program.run("thinking")
        program.run("conclusion")

        # Configure extraction
        program.extract("training_data", ["Training", "Correct"])
        program.extract("feedback_data", ["Feedback"])

        # Should compile and execute
        self.assert_program_can_compile(program)
        self.assert_workflow_factory_works(program)

    def test_empty_workflow_compilation_fails(self):
        """Test that empty workflows cannot be compiled."""
        program = self.create_program()

        # Don't add any sequences

        # Should fail to compile
        with self.assertRaises(Exception):
            program.compile()


class TestControlStructures(IntegrationTestResources):
    """Test control structure integration."""

    def test_simple_loop_workflow(self):
        """Test workflow with a simple loop structure."""
        program = self.create_program()

        # Build workflow with loop
        program.run("setup")
        with program.loop("loop_control") as loop:
            loop.run("thinking")
        program.run("conclusion")

        # Should compile successfully
        self.assert_program_can_compile(program)
        self.assert_workflow_factory_works(program)

    def test_simple_conditional_workflow(self):
        """Test workflow with a simple conditional structure."""
        program = self.create_program()

        # Build workflow with conditional
        program.run("setup")
        with program.when("decision") as (if_branch, else_branch):
            if_branch.run("thinking")
            else_branch.run("conclusion")

        # Should compile successfully
        self.assert_program_can_compile(program)
        self.assert_workflow_factory_works(program)

    def test_nested_control_structures(self):
        """Test workflow with nested control structures."""
        program = self.create_program()

        # Build complex nested workflow
        program.run("setup")
        with program.loop("loop_control") as loop:
            with loop.when("decision") as (if_branch, else_branch):
                if_branch.run("thinking")
                else_branch.run("thinking")
        program.run("conclusion")

        # Should compile successfully
        self.assert_program_can_compile(program)
        self.assert_workflow_factory_works(program)


class TestWorkflowComposition(IntegrationTestResources):
    """Test workflow composition and subroutines."""

    def test_subroutine_integration(self):
        """Test integrating subroutines into workflows."""
        # Create main program
        main_program = self.create_program()

        # Create subroutine
        subroutine = self.create_program(
            sequence_configs={"sub_thinking": 2, "sub_conclusion": 1}
        )
        subroutine.run("sub_thinking")
        subroutine.run("sub_conclusion")

        # Build main workflow with subroutine
        main_program.run("setup")
        main_program.subroutine(subroutine)
        main_program.run("conclusion")

        # Should compile successfully
        self.assert_program_can_compile(main_program)
        self.assert_workflow_factory_works(main_program)

    def test_multiple_subroutines(self):
        """Test workflow with multiple subroutines."""
        main_program = self.create_program()

        # Create multiple subroutines
        sub1 = self.create_program(sequence_configs={"sub1": 1})
        sub1.run("sub1")

        sub2 = self.create_program(sequence_configs={"sub2": 2})
        sub2.run("sub2")

        # Build main workflow
        main_program.run("setup")
        main_program.subroutine(sub1)
        main_program.run("thinking")
        main_program.subroutine(sub2)
        main_program.run("conclusion")

        # Should compile successfully
        self.assert_program_can_compile(main_program)


class TestComplexWorkflows(IntegrationTestResources):
    """Test complex, realistic workflow scenarios."""

    def test_constitutional_ai_workflow(self):
        """Test a complex Constitutional AI-style workflow."""
        program = self.create_program(
            sequence_configs={
                "setup": 3,
                "generate": 2,
                "critique": 3,
                "revise": 2,
                "loop_control": 2,
                "finalize": 2
            }
        )

        # Build Constitutional AI workflow
        program.run("setup")
        program.run("generate")

        # Revision loop
        with program.loop("loop_control") as revision_loop:
            revision_loop.run("critique")
            revision_loop.run("revise")

        program.run("finalize")

        # Configure extractions
        program.extract("initial_response", ["Training"])
        program.extract("critiques", ["Feedback"])
        program.extract("final_response", ["Correct"])

        # Should compile and execute
        self.assert_program_can_compile(program)
        self.assert_workflow_factory_works(program)

        # Visualization
        if SHOW_VISUALIZATIONS:
            factory = program.compile()
            workflow = factory()
            workflow.visualize()

    def test_multi_stage_reasoning_workflow(self):
        """Test a multi-stage reasoning workflow with branching."""
        program = self.create_program(
            sequence_configs={
                "problem_analysis": 3,
                "approach_decision": 2,
                "analytical_approach": 3,
                "creative_approach": 3,
                "synthesis": 2,
                "verification": 2
            }
        )

        # Build multi-stage reasoning workflow
        program.run("problem_analysis")

        # Branch based on approach
        with program.when("approach_decision") as (analytical, creative):
            analytical.run("analytical_approach")
            creative.run("creative_approach")

        program.run("synthesis")
        program.run("verification")

        # Configure comprehensive extraction
        program.extract("analysis", ["Training"])
        program.extract("reasoning", ["Correct"])
        program.extract("verification", ["Feedback"])

        # Should compile and execute
        self.assert_program_can_compile(program)
        self.assert_workflow_factory_works(program)

        # Visualization
        if SHOW_VISUALIZATIONS:
            factory = program.compile()
            workflow = factory()
            workflow.visualize()


if __name__ == "__main__":
    unittest.main()