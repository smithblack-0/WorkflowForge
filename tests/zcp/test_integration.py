"""
Integration tests for ZCP → RZCP → SZCP → LZCP pipeline.

These tests manually create ZCP chains (simulating parser output) and verify
the full lowering pipeline works correctly with resources, tools, and flow control.
"""

import unittest
import numpy as np
from unittest.mock import Mock
from typing import Dict, Any, Callable

# Import the modules under test
from src.workflow_forge.zcp.nodes import ZCPNode, RZCPNode, SZCPNode, LZCPNode
from src.workflow_forge.parsing.config_parsing import Config
from src.workflow_forge.backend.tag_converter import TagConverter
from src.workflow_forge.tokenizer_interface import TokenizerInterface
from src.workflow_forge.resources import AbstractResource, StaticStringResource, ListSamplerResource


class BaseIntegrationTest(unittest.TestCase):
    """Base class for integration tests with common setup."""

    def setUp(self):
        """Create realistic test fixtures manually (no parser)."""
        self.config = self._create_test_config()
        self.tokenizer = self._create_test_tokenizer()
        self.tag_converter = self._create_test_tag_converter()
        self.resources = self._create_test_resources()
        self.tool_registry = self._create_test_tool_registry()

    def _create_test_config(self) -> Config:
        """Create test config with current API."""
        return Config(
            zone_patterns=["[Prompt]", "[Answer]", "[EOS]"],
            required_patterns=["[Prompt]", "[Answer]"],
            valid_tags=["Training", "Correct", "Feedback"],
            default_max_token_length=1000,
            sequences=["setup", "control", "body"],
            control_pattern="[Jump]",
            escape_patterns=("[Escape]", "[EndEscape]"),
            tools=["calculator", "analyzer"],
            misc={}
        )

    def _create_test_tokenizer(self) -> TokenizerInterface:
        """Create REAL tokenizer interface implementation."""
        def tokenize_fn(text: str) -> np.ndarray:
            # Simple hash-based tokenization for testing
            return np.array([hash(text) % 1000 + 1], dtype=np.int32)

        def detokenize_fn(tokens: np.ndarray) -> str:
            # Simple reverse mapping for testing
            return f"detokenized_{tokens[0]}"

        def get_special_tokens_fn() -> Dict[str, int]:
            # Return mapping of special token strings to token IDs
            return {
                "[Prompt]": 1001,
                "[Answer]": 1002,
                "[EOS]": 1003,
                "[Jump]": 1004,
                "[Escape]": 1005,
                "[EndEscape]": 1006
            }

        return TokenizerInterface(tokenize_fn, detokenize_fn, get_special_tokens_fn)

    def _create_test_tag_converter(self) -> TagConverter:
        """Create REAL tag converter."""
        return TagConverter(self.config.valid_tags)

    def _create_test_resources(self) -> Dict[str, AbstractResource]:
        """Create real resource implementations for testing."""
        resources = {}

        # Simple string resource
        resources["constitution"] = StaticStringResource("Be kind and just")

        # List sampler resource
        feedback_options = ["Consider multiple perspectives", "Think about consequences", "Review your assumptions"]
        resources["feedback_sampler"] = ListSamplerResource(feedback_options)

        # Control resource
        resources["repeat_counter"] = StaticStringResource("3")

        return resources

    def _create_test_tool_registry(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        """Create real tool registry for testing."""
        def calculator_tool(tokens: np.ndarray) -> np.ndarray:
            return np.array([42], dtype=np.int32)  # Always returns "42"

        def analyzer_tool(tokens: np.ndarray) -> np.ndarray:
            return np.array([99], dtype=np.int32)  # Always returns "99"

        # Store references for verification in tests
        self.calculator_callback = calculator_tool
        self.analyzer_callback = analyzer_tool

        return {
            "calculator": calculator_tool,
            "analyzer": analyzer_tool
        }

    def create_construction_callback(self, template: str, resource_specs: Dict[str, Dict[str, Any]]) -> Callable[[Dict[str, AbstractResource]], str]:
        """Create a construction callback that resolves placeholders."""
        def callback(resources: Dict[str, AbstractResource]) -> str:
            resolved_values = {}
            for placeholder, spec in resource_specs.items():
                resource_name = spec['name']
                arguments = spec.get('arguments')

                if resource_name not in resources:
                    raise ValueError(f"Resource '{resource_name}' not found")

                resource = resources[resource_name]
                if arguments:
                    resolved_values[placeholder] = resource(**arguments)
                else:
                    resolved_values[placeholder] = resource()

            return template.format(**resolved_values)

        return callback

    def create_zcp_node(self, sequence: str, block: int, template: str,
                       resource_specs: Dict[str, Dict[str, Any]], zone_advance_str: str,
                       tags: list, timeout: int = 500) -> ZCPNode:
        """Helper to create ZCP nodes with consistent structure."""
        return ZCPNode(
            sequence=sequence,
            block=block,
            construction_callback=self.create_construction_callback(template, resource_specs),
            resource_specs=resource_specs,
            raw_text=template,
            zone_advance_str=zone_advance_str,
            tags=tags,
            timeout=timeout
        )

    def create_rzcp_node(self, sequence: str, block: int, zone_advance_str: str,
                        tags: list, timeout: int, sampling_callback: Callable[[], str],
                        **kwargs) -> RZCPNode:
        """Helper to create RZCP nodes with required fields."""
        defaults = {
            'escape_strs': ('[Escape]', '[EndEscape]'),  # Added required field
            'input': False,
            'output': False,
            'next_zone': None,
            'jump_advance_str': None,
            'jump_zone': None,
            'tool_name': None
        }
        defaults.update(kwargs)

        return RZCPNode(
            sequence=sequence,
            block=block,
            zone_advance_str=zone_advance_str,
            tags=tags,
            timeout=timeout,
            sampling_callback=sampling_callback,
            **defaults
        )


class TestZCPLoweringIntegration(BaseIntegrationTest):
    """Integration tests for ZCP → RZCP → SZCP → LZCP pipeline with manually created data."""

    def _create_simple_zcp_chain(self) -> ZCPNode:
        """Create a simple linear ZCP chain with no placeholders."""
        # First zone: prompt zone
        node1 = ZCPNode(
            sequence="setup",
            block=0,
            construction_callback=lambda resources: "[Prompt] What is ethics? [Answer]",
            resource_specs={},
            raw_text="[Prompt] What is ethics? [Answer]",
            zone_advance_str="[Answer]",
            tags=["Training"],
            timeout=500
        )

        # Second zone: answer zone
        node2 = ZCPNode(
            sequence="setup",
            block=0,
            construction_callback=lambda resources: " Ethics is about right and wrong. [EOS]",
            resource_specs={},
            raw_text=" Ethics is about right and wrong. [EOS]",
            zone_advance_str="[EOS]",
            tags=["Correct"],
            timeout=500
        )

        # Link them
        node1.next_zone = node2
        return node1

    def _create_zcp_with_resources(self) -> ZCPNode:
        """Create ZCP chain that requires resource resolution."""
        resource_specs = {
            "principle": {
                "name": "constitution",
                "arguments": None,
                "type": "default"
            },
            "feedback": {
                "name": "feedback_sampler",
                "arguments": {"num_samples": 3},
                "type": "default"
            }
        }

        template = "[Prompt] Follow this principle: {principle}. Also consider: {feedback} [Answer]"

        return self.create_zcp_node(
            sequence="reasoning",
            block=0,
            template=template,
            resource_specs=resource_specs,
            zone_advance_str="[Answer]",
            tags=["Training"],
            timeout=1000
        )

    def _create_zcp_with_tools(self) -> ZCPNode:
        """Create ZCP chain that uses tools."""
        # Capture zone (output=True, tool_name set)
        capture_node = ZCPNode(
            sequence="calculation",
            block=0,
            construction_callback=lambda resources: "[Prompt] Calculate 2+2 [Answer]",
            resource_specs={},
            raw_text="[Prompt] Calculate 2+2 [Answer]",
            zone_advance_str="[Answer]",
            tags=[],
            timeout=500
        )

        # Feed zone (input=True)
        feed_node = ZCPNode(
            sequence="calculation",
            block=0,
            construction_callback=lambda resources: " The answer is: [EOS]",
            resource_specs={},
            raw_text=" The answer is: [EOS]",
            zone_advance_str="[EOS]",
            tags=["Correct"],
            timeout=500
        )

        capture_node.next_zone = feed_node
        return capture_node

    def _create_rzcp_with_flow_control(self) -> RZCPNode:
        """Create RZCP graph with flow control (loop)."""
        # Control node - decides whether to loop
        control_callback = lambda: "[Prompt] Continue? Say 'yes' or emit [Escape] [Jump] [EndEscape] to exit [Answer]"

        # Body node - the work being repeated
        body_callback = lambda: " Working on the problem... [EOS]"

        body_node = self.create_rzcp_node(
            sequence="body",
            block=0,
            zone_advance_str="[EOS]",
            tags=["Training"],
            timeout=500,
            sampling_callback=body_callback
        )

        # Control node with proper jump setup
        control_node = self.create_rzcp_node(
            sequence="control",
            block=0,
            zone_advance_str="[Answer]",
            tags=[],
            timeout=500,
            sampling_callback=control_callback,
            jump_advance_str="[Jump]",
            jump_zone=body_node  # Must provide both or neither
        )

        # Wire the loop: control → body → control (jump back)
        control_node.next_zone = body_node
        body_node.next_zone = control_node  # Loop back

        return control_node

    # TEST METHODS

    def test_simple_linear_chain_lowering(self):
        """Test basic ZCP chain → LZCP with no flow control."""
        # Setup: Create ZCP chain manually
        zcp_chain = self._create_simple_zcp_chain()

        # Execute: Lower ZCP → RZCP → SZCP → LZCP (fixed API calls)
        rzcp = zcp_chain.lower(self.resources, self.config)  # Added missing config parameter
        szcp = rzcp.lower()
        lzcp = szcp.lower(self.tokenizer, self.tag_converter, self.tool_registry)

        # Verify: Check final LZCP structure
        self.assertIsInstance(lzcp, LZCPNode)
        self.assertEqual(lzcp.sequence, "setup")
        self.assertEqual(lzcp.block, 0)
        self.assertEqual(lzcp.timeout, 500)

        # Verify chain structure preserved
        self.assertIsNotNone(lzcp.next_zone)
        self.assertIsInstance(lzcp.next_zone, LZCPNode)
        self.assertEqual(lzcp.next_zone.sequence, "setup")
        self.assertIsNone(lzcp.next_zone.next_zone)  # End of chain

        # Verify tokenization occurred
        self.assertIsInstance(lzcp.tokens, np.ndarray)
        self.assertIsInstance(lzcp.zone_advance_tokens, np.ndarray)
        self.assertIsInstance(lzcp.tags, np.ndarray)
        self.assertEqual(lzcp.tags.dtype, np.bool_)

        # Verify escape_tokens field present
        self.assertIsInstance(lzcp.escape_tokens, tuple)
        self.assertEqual(len(lzcp.escape_tokens), 2)

    def test_resource_resolution_integration(self):
        """Test that resources with arguments are called correctly."""
        # Setup: ZCP with placeholders requiring resources with args
        zcp_with_resources = self._create_zcp_with_resources()

        # Execute: Lower with specific resource calls expected (fixed API)
        rzcp = zcp_with_resources.lower(self.resources, self.config)
        szcp = rzcp.lower()

        # Verify: Resources were used correctly (check actual text resolution)
        self.assertIn("Be kind and just", szcp.text)
        self.assertIn("Consider multiple perspectives", szcp.text)
        self.assertNotIn("{principle}", szcp.text)  # Placeholder resolved
        self.assertNotIn("{feedback}", szcp.text)   # Placeholder resolved

    def test_tool_integration(self):
        """Test tool name → callback resolution works."""
        # Setup: Create RZCP with tool name (using helper)
        capture_callback = lambda: "[Prompt] Calculate 2+2 [Answer]"

        capture_node = self.create_rzcp_node(
            sequence="calc",
            block=0,
            zone_advance_str="[Answer]",
            tags=[],
            timeout=500,
            sampling_callback=capture_callback,
            output=True,
            tool_name="calculator"
        )

        # Execute: Lower RZCP → SZCP → LZCP
        szcp = capture_node.lower()
        lzcp = szcp.lower(self.tokenizer, self.tag_converter, self.tool_registry)

        # Verify: Tool name preserved in SZCP
        self.assertEqual(szcp.tool_name, "calculator")

        # Verify: Tool callback attached in LZCP
        self.assertIsNotNone(lzcp.tool_callback)
        self.assertTrue(callable(lzcp.tool_callback))

        # Verify: It's the exact callback from our registry
        self.assertIs(lzcp.tool_callback, self.calculator_callback)

        # Verify: Callback works correctly
        test_tokens = np.array([1, 2, 3], dtype=np.int32)
        result_tokens = lzcp.tool_callback(test_tokens)
        np.testing.assert_array_equal(result_tokens, np.array([42], dtype=np.int32))

    def test_flow_control_topology_preservation(self):
        """Test jump references survive all lowering stages."""
        # Setup: RZCP with simple loop manually wired
        rzcp_head = self._create_rzcp_with_flow_control()

        # Execute: Lower RZCP → SZCP → LZCP
        szcp = rzcp_head.lower()
        lzcp = szcp.lower(self.tokenizer, self.tag_converter, self.tool_registry)

        # Verify: Graph topology preserved at all stages
        # RZCP structure
        self.assertEqual(rzcp_head.sequence, "control")
        self.assertIsNotNone(rzcp_head.next_zone)
        self.assertEqual(rzcp_head.next_zone.sequence, "body")
        self.assertEqual(rzcp_head.next_zone.next_zone, rzcp_head)  # Loop back

        # SZCP structure preserved
        self.assertEqual(szcp.sequence, "control")
        self.assertIsNotNone(szcp.next_zone)
        self.assertEqual(szcp.next_zone.sequence, "body")
        self.assertEqual(szcp.next_zone.next_zone, szcp)  # Loop back preserved

        # LZCP structure preserved
        self.assertEqual(lzcp.sequence, "control")
        self.assertIsNotNone(lzcp.next_zone)
        self.assertEqual(lzcp.next_zone.sequence, "body")
        self.assertEqual(lzcp.next_zone.next_zone, lzcp)  # Loop back preserved

    def test_complex_graph_integration(self):
        """Test realistic workflow with everything: resources, tools, flow control."""
        # Setup: Create ZCP nodes with proper resource specs

        # Control node ZCP with resource spec
        control_specs = {
            "max_tries": {
                "name": "repeat_counter",
                "arguments": None,
                "type": "control"
            }
        }
        control_template = "[Prompt] Try again? Max {max_tries} times. Emit [Escape] [Jump] [EndEscape] to exit [Answer]"

        control_zcp = self.create_zcp_node(
            sequence="control",
            block=0,
            template=control_template,
            resource_specs=control_specs,
            zone_advance_str="[Answer]",
            tags=[],
            timeout=500
        )

        # Work node ZCP
        work_zcp = ZCPNode(
            sequence="work",
            block=0,
            construction_callback=lambda resources: "[Prompt] Solve this problem [Answer] Here's my attempt [EOS]",
            resource_specs={},
            raw_text="[Prompt] Solve this problem [Answer] Here's my attempt [EOS]",
            zone_advance_str="[EOS]",
            tags=["Training"],
            timeout=1000
        )

        # Feedback node ZCP
        feedback_zcp = ZCPNode(
            sequence="feedback",
            block=0,
            construction_callback=lambda resources: "[Prompt] Based on analysis: [Answer] Improved solution [EOS]",
            resource_specs={},
            raw_text="[Prompt] Based on analysis: [Answer] Improved solution [EOS]",
            zone_advance_str="[EOS]",
            tags=["Correct"],
            timeout=500
        )

        # Execute: Lower ZCP → RZCP with resources (fixed API)
        control_rzcp = control_zcp.lower(self.resources, self.config)
        work_rzcp = work_zcp.lower(self.resources, self.config)
        feedback_rzcp = feedback_zcp.lower(self.resources, self.config)

        # Wire RZCP graph with flow control and tool settings
        control_rzcp.next_zone = work_rzcp
        work_rzcp.next_zone = feedback_rzcp
        feedback_rzcp.next_zone = control_rzcp  # Loop back

        # Set up jump control
        control_rzcp.jump_advance_str = "[Jump]"
        control_rzcp.jump_zone = feedback_rzcp  # Jump to end of loop

        # Add tool integration to work node
        work_rzcp.output = True
        work_rzcp.tool_name = "analyzer"

        # Add input flag to feedback node
        feedback_rzcp.input = True

        # Execute: Lower RZCP → SZCP → LZCP
        szcp = control_rzcp.lower()
        lzcp = szcp.lower(self.tokenizer, self.tag_converter, self.tool_registry)

        # Verify: All features integrated correctly

        # 1. Resource resolution worked
        self.assertIn("3", szcp.text)  # Resource was resolved

        # 2. Tool integration
        work_lzcp = lzcp.next_zone
        self.assertIsNotNone(work_lzcp.tool_callback)
        self.assertTrue(work_lzcp.output)
        self.assertIs(work_lzcp.tool_callback, self.analyzer_callback)

        # 3. Input/output flags preserved
        feedback_lzcp = work_lzcp.next_zone
        self.assertTrue(feedback_lzcp.input)

        # 4. Graph topology preserved
        self.assertEqual(feedback_lzcp.next_zone, lzcp)  # Loop back

        # 5. Jump control preserved
        self.assertIsNotNone(lzcp.jump_tokens)
        self.assertEqual(lzcp.jump_zone, feedback_lzcp)

        # 6. Tokenization occurred
        self.assertIsInstance(lzcp.tokens, np.ndarray)
        self.assertIsInstance(lzcp.tags, np.ndarray)
        self.assertEqual(lzcp.tags.dtype, np.bool_)

        # 7. Escape tokens present
        self.assertIsInstance(lzcp.escape_tokens, tuple)

    def test_error_propagation_integration(self):
        """Test that errors propagate correctly through the pipeline."""

        # Setup: Create a mock resource that actually fails
        class FailingResource(AbstractResource):
            def __call__(self, **kwargs) -> str:
                raise RuntimeError("Resource failed!")

        failing_resource = FailingResource()
        bad_resources = {"failing_resource": failing_resource}

        resource_specs = {
            "bad_placeholder": {
                "name": "failing_resource",
                "arguments": None,
                "type": "default"
            }
        }

        zcp_node = self.create_zcp_node(
            sequence="test",
            block=0,
            template="Text {bad_placeholder}",
            resource_specs=resource_specs,
            zone_advance_str="[Answer]",
            tags=[],
            timeout=500
        )

        # Execute: Should fail during resource resolution in sampling callback
        with self.assertRaises(Exception) as context:
            rzcp = zcp_node.lower(bad_resources, self.config)
            rzcp.lower()  # This should trigger sampling callback and fail

        # Verify: Exception was properly raised (specific type depends on implementation)
        self.assertIsNotNone(context.exception)

    def test_serialization_boundary_integration(self):
        """Test that SZCP can be serialized (simulating client/server boundary)."""
        # Setup: Create RZCP and lower to SZCP (fixed API)
        zcp_chain = self._create_simple_zcp_chain()
        rzcp = zcp_chain.lower(self.resources, self.config)
        szcp = rzcp.lower()

        # Execute: Serialize and deserialize
        serialized = szcp.serialize()
        deserialized = SZCPNode.deserialize(serialized)

        # Verify: Deserialized SZCP can continue to LZCP
        lzcp = deserialized.lower(self.tokenizer, self.tag_converter, self.tool_registry)

        # Verify: Final result is identical
        self.assertIsInstance(lzcp, LZCPNode)
        self.assertEqual(lzcp.sequence, "setup")
        self.assertIsNotNone(lzcp.next_zone)

        # Verify: escape_strs preserved through serialization
        self.assertEqual(deserialized.escape_strs, ('[Escape]', '[EndEscape]'))


if __name__ == "__main__":
    unittest.main()