"""
Integration tests for UDPL Parser

These tests use real UDPL examples from the documentation to verify
the entire parsing pipeline works correctly end-to-end.
"""

import unittest
import tempfile
import os
from pathlib import Path

# Import the modules under test
from src.workflow_forge.parsing.main_parser import parse_udpl_file, parse_udpl_folder
from src.workflow_forge.parsing.config_parsing import Config
from src.workflow_forge.ZCP.nodes import ZCPNode


class TestDocumentationExamples(unittest.TestCase):
    """Test parsing of examples from the UDPL documentation."""

    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_simple_philosophical_example(self):
        """Test parsing the main documentation example."""
        # Create the UDPL file from documentation
        udpl_content = '''
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags = ["Training", "Final"]
default_max_token_length = 20000
sequences = ["blocks"]
control_token = "[Jump]"
escape_token = "[Escape]"

[[blocks]]
text = """[Prompt]Consider and resolve a philosophical dilemma
according to the following principles: {placeholder} 
[Answer]
Okay, I should think this through. 
"""
tags = [["Training"], []]

[blocks.placeholder]
name = "constitution_overview"

[[blocks]]
text = """[Prompt]Revise your previous answer after
considering the following additional details.
Make sure to also pretend you are directly answering
the prior response: {details}
[Answer]
Okay, I should begin by thinking through the new point, then
consider if I should revise my answer. Then I give my final 
answer.
"""
tags = [[], []]
repeats = 3

[blocks.details]
name = "constitution_details"
arguments = {"num_samples" = 3}

[[blocks]]
text = """[Prompt]
Consider your reflections so far. State a perfect answer
as though you jumped straight to the right answer.
[Answer]"""
tags = [[], ["Final"]]
'''

        test_file = self.temp_path / "example.toml"
        test_file.write_text(udpl_content)

        # Parse the file
        sequences, config = parse_udpl_file(str(test_file))

        # Verify config was parsed correctly
        self.assertIsInstance(config, Config)
        self.assertEqual(config.zone_tokens, ["[Prompt]", "[Answer]", "[EOS]"])
        self.assertEqual(config.required_tokens, ["[Prompt]", "[Answer]"])
        self.assertEqual(config.valid_tags, ["Training", "Final"])
        self.assertEqual(config.default_max_token_length, 20000)
        self.assertEqual(config.sequences, ["blocks"])
        self.assertEqual(config.control_token, "[Jump]")
        self.assertEqual(config.escape_token, "[Escape]")

        # Verify sequences were parsed
        self.assertIn("blocks", sequences)
        self.assertIsInstance(sequences["blocks"], ZCPNode)

        # Walk through the sequence chain and verify structure
        current_node = sequences["blocks"]
        nodes = []
        while current_node is not None:
            nodes.append(current_node)
            current_node = current_node.next_zone

        # Should have multiple nodes due to repeats
        # First block (1 node) + Second block repeated 3 times (3 nodes) + Third block (1 node) = 5 nodes
        self.assertEqual(len(nodes), 5)

        # Verify first node (first block)
        first_node = nodes[0]
        self.assertEqual(first_node.sequence, "blocks")
        self.assertEqual(first_node.block, 0)
        self.assertIn("{placeholder}", first_node.raw_text)
        self.assertEqual(first_node.tags, ["Training"])
        self.assertIn("placeholder", first_node.resource_specs)
        self.assertEqual(first_node.resource_specs["placeholder"]["name"], "constitution_overview")
        self.assertEqual(first_node.resource_specs["placeholder"]["type"], "default")

        # Verify one of the repeated nodes (second block)
        repeated_node = nodes[1]  # First repetition of second block
        self.assertEqual(repeated_node.sequence, "blocks")
        self.assertEqual(repeated_node.block, 1)
        self.assertIn("{details}", repeated_node.raw_text)
        self.assertEqual(repeated_node.tags, [])
        self.assertIn("details", repeated_node.resource_specs)
        self.assertEqual(repeated_node.resource_specs["details"]["name"], "constitution_details")
        self.assertEqual(repeated_node.resource_specs["details"]["arguments"], {"num_samples": 3})

        # Verify final node (third block)
        final_node = nodes[4]
        self.assertEqual(final_node.sequence, "blocks")
        self.assertEqual(final_node.block, 2)
        self.assertEqual(final_node.tags, ["Final"])
        self.assertEqual(final_node.resource_specs, {})  # No placeholders

    def test_basic_resource_example(self):
        """Test the basic resource example from documentation."""
        udpl_content = '''
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags = ["Training", "Correct"]
default_max_token_length = 1000
sequences = ["solving"]
control_token = "[Jump]"
escape_token = "[Escape]"

[[solving]]
text = """
[Prompt] Consider this: {philosophy}
[Answer]
"""
tags = [["Training"], ["Correct"]]

[solving.philosophy]
name = "principles"
'''

        test_file = self.temp_path / "basic_resource.toml"
        test_file.write_text(udpl_content)

        # Parse the file
        sequences, config = parse_udpl_file(str(test_file))

        # Verify parsing
        self.assertIn("solving", sequences)
        node = sequences["solving"]

        self.assertEqual(node.sequence, "solving")
        self.assertEqual(node.tags, ["Training"])
        self.assertIn("philosophy", node.resource_specs)
        self.assertEqual(node.resource_specs["philosophy"]["name"], "principles")
        self.assertIsNone(node.resource_specs["philosophy"]["arguments"])
        self.assertEqual(node.resource_specs["philosophy"]["type"], "default")

    def test_flow_control_resource_example(self):
        """Test the flow control resource example from documentation."""
        udpl_content = '''
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags = []
default_max_token_length = 1000
sequences = ["loop"]
control_token = "[Jump]"
escape_token = "[Escape]"

[[loop]]
text = """
[Prompt] Repeat {min} to {max} times. Emit the
[Escape] "[Jump]" token when you are ready to break.
Otherwise just say continue. 
[Answer]
"""
tags = [[], []]

[loop.min]
name = "min_value"
type = "flow_control"

[loop.max]
name = "max_value"
type = "flow_control"
'''

        test_file = self.temp_path / "flow_control.toml"
        test_file.write_text(udpl_content)

        # Parse the file
        sequences, config = parse_udpl_file(str(test_file))

        # Verify parsing
        self.assertIn("loop", sequences)
        node = sequences["loop"]

        # Check resource specs include flow_control type
        self.assertIn("min", node.resource_specs)
        self.assertIn("max", node.resource_specs)
        self.assertEqual(node.resource_specs["min"]["type"], "flow_control")
        self.assertEqual(node.resource_specs["max"]["type"], "flow_control")
        self.assertEqual(node.resource_specs["min"]["name"], "min_value")
        self.assertEqual(node.resource_specs["max"]["name"], "max_value")

    def test_tagset_example(self):
        """Test the tagset example from documentation."""
        udpl_content = '''
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags = ["Incorrect1", "Incorrect2"]
default_max_token_length = 1000
sequences = ["contrast"]
control_token = "[Jump]"
escape_token = "[Escape]"

[[contrast]]
text = """
[Prompt] Create a flawed argument based on: {claim}
[Answer]
"""
tagset = [
  [[], ["Incorrect1"]],
  [[], ["Incorrect2"]]
]

[contrast.claim]
name = "claim_sampler"
arguments = {"num_samples" = 1}
'''

        test_file = self.temp_path / "tagset.toml"
        test_file.write_text(udpl_content)

        # Parse the file
        sequences, config = parse_udpl_file(str(test_file))

        # Verify parsing
        self.assertIn("contrast", sequences)

        # Walk the chain to see both repetitions
        current_node = sequences["contrast"]
        nodes = []
        while current_node is not None:
            nodes.append(current_node)
            current_node = current_node.next_zone

        # Should have 2 nodes for the 2 tagset entries
        self.assertEqual(len(nodes), 2)

        # First repetition should have Incorrect1 tag
        self.assertEqual(nodes[0].tags, ["Incorrect1"])
        # Second repetition should have Incorrect2 tag
        self.assertEqual(nodes[1].tags, ["Incorrect2"])

        # Both should have the same resource specs
        for node in nodes:
            self.assertIn("claim", node.resource_specs)
            self.assertEqual(node.resource_specs["claim"]["name"], "claim_sampler")
            self.assertEqual(node.resource_specs["claim"]["arguments"], {"num_samples": 1})


class TestFolderParsing(unittest.TestCase):
    """Test parsing multiple TOML files from a folder."""

    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_split_config_and_sequences(self):
        """Test splitting config and sequences across multiple files."""
        # Create config file
        config_content = '''
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags = ["Training", "Test"]
default_max_token_length = 1000
sequences = ["setup", "main"]
control_token = "[Jump]"
escape_token = "[Escape]"
'''

        # Create sequences file
        sequences_content = '''
[[setup]]
text = """
[Prompt] Initialize the system
[Answer]
"""
tags = [["Training"], []]

[[main]]
text = """
[Prompt] Run the main process with {input}
[Answer]
"""
tags = [[], ["Test"]]

[main.input]
name = "input_resource"
'''

        config_file = self.temp_path / "config.toml"
        sequences_file = self.temp_path / "sequences.toml"

        config_file.write_text(config_content)
        sequences_file.write_text(sequences_content)

        # Parse the folder
        sequences, config = parse_udpl_folder(str(self.temp_path))

        # Verify config
        self.assertIsInstance(config, Config)
        self.assertEqual(config.sequences, ["setup", "main"])

        # Verify sequences
        self.assertIn("setup", sequences)
        self.assertIn("main", sequences)

        # Verify setup sequence
        setup_node = sequences["setup"]
        self.assertEqual(setup_node.sequence, "setup")
        self.assertEqual(setup_node.tags, ["Training"])

        # Verify main sequence
        main_node = sequences["main"]
        self.assertEqual(main_node.sequence, "main")
        self.assertEqual(main_node.tags, ["Test"])
        self.assertIn("input", main_node.resource_specs)
        self.assertEqual(main_node.resource_specs["input"]["name"], "input_resource")

    def test_collision_detection(self):
        """Test that key collisions are properly detected."""
        # Create two files with conflicting keys
        file1_content = '''
[config]
zone_tokens = ["[A]", "[B]"]
required_tokens = ["[A]"]
valid_tags = []
default_max_token_length = 100
sequences = ["test"]
control_token = "[Jump]"
escape_token = "[Escape]"

[[test]]
text = "[A] from file 1 [B]"
tags = [[]]
'''

        file2_content = '''
[config]  # This will collide with file1
zone_tokens = ["[C]", "[D]"]
required_tokens = ["[C]"]
valid_tags = []
default_max_token_length = 200
sequences = ["other"]
control_token = "[Skip]"
escape_token = "[Cancel]"
'''

        file1 = self.temp_path / "file1.toml"
        file2 = self.temp_path / "file2.toml"

        file1.write_text(file1_content)
        file2.write_text(file2_content)

        # Should raise error about collision
        from src.workflow_forge.parsing.main_parser import UDPLParseError
        with self.assertRaises(UDPLParseError) as context:
            parse_udpl_folder(str(self.temp_path))

        self.assertIn("Key collisions found", str(context.exception))
        self.assertIn("config", str(context.exception))


class TestConstructionCallbacks(unittest.TestCase):
    """Test that construction callbacks work correctly."""

    def setUp(self):
        """Set up temporary directory and mock resources."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_construction_callback_execution(self):
        """Test that construction callbacks properly resolve placeholders."""
        udpl_content = '''
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags = ["Test"]
default_max_token_length = 1000
sequences = ["test"]
control_token = "[Jump]"
escape_token = "[Escape]"

[[test]]
text = """
[Prompt] Use principle {principle} and detail {detail}
[Answer]
"""
tags = [["Test"], []]

[test.principle]
name = "main_principle"

[test.detail]
name = "supporting_detail"
arguments = {"count" = 2}
'''

        test_file = self.temp_path / "callback_test.toml"
        test_file.write_text(udpl_content)

        # Parse the file
        sequences, config = parse_udpl_file(str(test_file))

        # Get the node and test its construction callback
        node = sequences["test"]

        # Create mock resources
        from unittest.mock import Mock
        from src.workflow_forge.resources import AbstractResource

        mock_principle = Mock(spec=AbstractResource)
        mock_principle.return_value = "safety first"

        mock_detail = Mock(spec=AbstractResource)
        mock_detail.return_value = "with careful consideration"

        resources = {
            "main_principle": mock_principle,
            "supporting_detail": mock_detail
        }

        # Call the construction callback
        final_text = node.construction_callback(resources)

        # Verify the text was properly constructed
        expected_text = """
[Prompt] Use principle safety first and detail with careful consideration
[Answer]
"""
        self.assertEqual(final_text, expected_text)

        # Verify resources were called correctly
        mock_principle.assert_called_once_with()
        mock_detail.assert_called_once_with(count=2)


if __name__ == "__main__":
    unittest.main()