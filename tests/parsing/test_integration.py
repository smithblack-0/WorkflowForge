"""
Integration tests for UDPL Parser

These tests use real UDPL examples from the documentation to verify
the entire parsing pipeline works correctly end-to-end from TOML → zcp nodes.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

# Import the modules under test
from workflow_forge.frontend.parsing.main_parser import parse_udpl_file, parse_udpl_folder
from workflow_forge.frontend.parsing.config_parsing import Config
from workflow_forge.zcp.nodes import ZCPNode
from workflow_forge.resources import AbstractResource


class BaseIntegrationTest(unittest.TestCase):
    """Base class with common setup and helper methods for integration tests."""

    def create_temp_file(self, content: str, suffix: str = '.toml') -> str:
        """
        Create a temporary file with given content.

        Args:
            content: Content to write to the file
            suffix: File suffix (default .toml)

        Returns:
            Path to the temporary file
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            return f.name

    def cleanup_temp_file(self, path: str):
        """Clean up a temporary file."""
        Path(path).unlink()

    def create_temp_dir(self) -> Path:
        """Create a temporary directory."""
        return Path(tempfile.mkdtemp())

    def cleanup_temp_dir(self, path: Path):
        """Clean up a temporary directory."""
        shutil.rmtree(path)

    def collect_sequence_nodes(self, head_node: ZCPNode) -> list:
        """
        Collect all nodes in a sequence chain.

        Args:
            head_node: First node in the sequence

        Returns:
            List of all nodes in the sequence
        """
        nodes = []
        current = head_node
        while current is not None:
            nodes.append(current)
            current = current.next_zone
        return nodes

    def find_nodes_with_text(self, nodes: list, search_text: str) -> list:
        """Find nodes containing specific text."""
        return [node for node in nodes if search_text in node.raw_text]

    def find_nodes_with_tags(self, nodes: list, tag: str) -> list:
        """Find nodes containing specific tag."""
        return [node for node in nodes if tag in node.tags]

    def get_base_config_content(self) -> str:
        """Get base config content for TOML files."""
        return '''[config]
zone_patterns = ["[Prompt]", "[Answer]", "[EOS]"]
required_patterns = ["[Prompt]", "[Answer]"]
valid_tags = ["Training", "Correct", "Incorrect", "Feedback"]
default_max_token_length = 20000
sequences = ["setup", "loop", "solving", "concluding"]
control_pattern = "[Jump]"
escape_patterns = ["[Escape]", "[EndEscape]"]
tools = []'''


class TestPhilosophicalSelfPlayExample(BaseIntegrationTest):
    """Test parsing the main philosophical self-play example from documentation."""

    def setUp(self):
        """Set up the philosophical self-play TOML content."""
        self.philosophical_toml = self.get_base_config_content() + '''

# This will setup the scenario.
# Notice the tagging on the second
# entry
[[setup]]
text = """
[Prompt] 
Think of an interesting philosophical scenario with unclear
ethical solutions. Reason it out
[Answer]
"""
tags = [[], ["Training"]]

[[setup]]
text = """
[Prompt] 
Clearly state the philosophical scenario, omiting the 
reasoning, like you are going to show this to another
person and ask them to solve it.
[Answer]
"""
tags = [[], ["Training"]]

# This controls 
[[loop]]
text = """
[Prompt]
You are in flow control right now. If you are satisfied
with your current answer emit [Escape] "[Jump]" [EndEscape]. Otherwise
answer "try again". If this is your first time seeing this,
just say "proceed". Repeat at least {min} times and at most
{max} times.
[Answer]
"""
tags = [[],[]]

[loop.min]
name = "min_control_resource"
type = "custom"
[loop.max]
name = "max_control_resource"
type = "custom"

# This will be repeated again and again as needed.

[[solving]]
text = """
[Prompt]
Reason through an answer to the scenario. First
select a small subset of the following principles.
Revise your previous answer if desired and it 
exists.

{placeholder}

You may also want to keep in mind this
feedback you previous created

{feedback}
[Answer]
"""
tags=[[], []]
[solving.placeholder]
name = "constitution"
[solving.feedback]
name = "feedback_backend"
arguments = {"num_samples" = 3}

# This sequence generates output and feedback
[[concluding]]
text = """
[Prompt]
State the scenario and the best way to resolve it directly;
[Answer]
"""
tags =[[],["Correct"]]

[[concluding]]
text = """
[Prompt]
State the scenario and a subtly incorrect way to resolve the scenario;
[Answer]
"""
tags =[[],["Incorrect"]]

[[concluding]]
text = """
[Prompt]
Based on your answers, state several things you could
do better next time.
[Answer]
"""
tags = [[],["Feedback"]]
'''

    def test_config_parsing(self):
        """Test that config is parsed correctly."""
        temp_path = self.create_temp_file(self.philosophical_toml)

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Verify config structure
            self.assertIsInstance(config, Config)
            self.assertEqual(config.zone_patterns, ["[Prompt]", "[Answer]", "[EOS]"])
            self.assertEqual(config.required_patterns, ["[Prompt]", "[Answer]"])
            self.assertEqual(config.valid_tags, ["Training", "Correct", "Incorrect", "Feedback"])
            self.assertEqual(config.default_max_token_length, 20000)
            self.assertEqual(config.sequences, ["setup", "loop", "solving", "concluding"])
            self.assertEqual(config.control_pattern, "[Jump]")
            self.assertEqual(config.escape_patterns, ("[Escape]", "[EndEscape]"))
            self.assertEqual(config.tools, [])
            self.assertEqual(config.num_zones_per_block, 2)  # 3 zone patterns - 1

        finally:
            self.cleanup_temp_file(temp_path)

    def test_sequence_structure(self):
        """Test that all sequences are parsed and have correct structure."""
        temp_path = self.create_temp_file(self.philosophical_toml)

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Verify all sequences exist
            self.assertEqual(set(sequences.keys()), {"setup", "loop", "solving", "concluding"})

            # Verify all sequence heads are ZCPNodes
            for seq_name, seq_head in sequences.items():
                self.assertIsInstance(seq_head, ZCPNode)

        finally:
            self.cleanup_temp_file(temp_path)

    def test_setup_sequence_structure(self):
        """Test setup sequence has correct zone structure and content."""
        temp_path = self.create_temp_file(self.philosophical_toml)

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Walk through setup sequence chain
            setup_nodes = self.collect_sequence_nodes(sequences["setup"])

            # Should have 6 zones total (2 blocks × 3 zones per block)
            self.assertEqual(len(setup_nodes), 6)

            # Check first block zones
            # Zone 0: Initial trigger zone
            self.assertEqual(setup_nodes[0].zone_advance_str, "[Prompt]")
            self.assertEqual(setup_nodes[0].raw_text, "[Prompt]")
            self.assertEqual(setup_nodes[0].tags, [])
            self.assertEqual(setup_nodes[0].sequence, "setup")
            self.assertEqual(setup_nodes[0].block, 0)

            # Zone 1: First block prompt→answer
            self.assertEqual(setup_nodes[1].zone_advance_str, "[Answer]")
            self.assertIn("Think of an interesting philosophical scenario", setup_nodes[1].raw_text)
            self.assertIn("[Answer]", setup_nodes[1].raw_text)
            self.assertEqual(setup_nodes[1].tags, [])
            self.assertEqual(setup_nodes[1].sequence, "setup")
            self.assertEqual(setup_nodes[1].block, 0)

            # Zone 2: First block answer→eos
            self.assertEqual(setup_nodes[2].zone_advance_str, "[EOS]")
            self.assertEqual(setup_nodes[2].tags, ["Training"])

            # Zone 3: Second block initial
            self.assertEqual(setup_nodes[3].zone_advance_str, "[Prompt]")
            self.assertEqual(setup_nodes[3].raw_text, "[Prompt]")
            self.assertEqual(setup_nodes[3].tags, [])
            self.assertEqual(setup_nodes[3].sequence, "setup")
            self.assertEqual(setup_nodes[3].block, 1)

            # Zone 4: Second block prompt→answer
            self.assertEqual(setup_nodes[4].zone_advance_str, "[Answer]")
            self.assertIn("Clearly state the philosophical scenario", setup_nodes[4].raw_text)
            self.assertEqual(setup_nodes[4].tags, [])

            # Zone 5: Second block answer→eos
            self.assertEqual(setup_nodes[5].zone_advance_str, "[EOS]")
            self.assertEqual(setup_nodes[5].tags, ["Training"])

        finally:
            self.cleanup_temp_file(temp_path)

    def test_loop_sequence_resources(self):
        """Test loop sequence has correct resource specifications."""
        temp_path = self.create_temp_file(self.philosophical_toml)

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Walk through loop sequence to find zone with placeholders
            loop_nodes = self.collect_sequence_nodes(sequences["loop"])
            placeholder_zones = self.find_nodes_with_text(loop_nodes, "{min}")

            self.assertTrue(len(placeholder_zones) > 0, "Could not find zone with {min} placeholder")
            prompt_zone = placeholder_zones[0]

            # Check resource specifications
            expected_specs = {
                "min": {
                    "name": "min_control_resource",
                    "arguments": None,
                    "type": "custom"
                },
                "max": {
                    "name": "max_control_resource",
                    "arguments": None,
                    "type": "custom"
                }
            }
            self.assertEqual(prompt_zone.resource_specs, expected_specs)

            # Verify text contains placeholders
            self.assertIn("{min}", prompt_zone.raw_text)
            self.assertIn("{max}", prompt_zone.raw_text)
            self.assertIn("[Escape]", prompt_zone.raw_text)
            self.assertIn("[Jump]", prompt_zone.raw_text)

        finally:
            self.cleanup_temp_file(temp_path)

    def test_solving_sequence_resources(self):
        """Test solving sequence has correct resource specifications."""
        temp_path = self.create_temp_file(self.philosophical_toml)

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Find zone with placeholders in solving sequence
            solving_nodes = self.collect_sequence_nodes(sequences["solving"])
            placeholder_zones = self.find_nodes_with_text(solving_nodes, "{placeholder}")

            self.assertTrue(len(placeholder_zones) > 0, "Could not find zone with placeholders")
            prompt_zone = placeholder_zones[0]

            # Check resource specifications
            expected_specs = {
                "placeholder": {
                    "name": "constitution",
                    "arguments": None,
                    "type": "standard"
                },
                "feedback": {
                    "name": "feedback_backend",
                    "arguments": {"num_samples": 3},
                    "type": "standard"
                }
            }
            self.assertEqual(prompt_zone.resource_specs, expected_specs)

        finally:
            self.cleanup_temp_file(temp_path)

    def test_concluding_sequence_tags(self):
        """Test concluding sequence has correct tag distribution."""
        temp_path = self.create_temp_file(self.philosophical_toml)

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Walk through concluding sequence
            concluding_nodes = self.collect_sequence_nodes(sequences["concluding"])

            # Should have 9 zones (3 blocks × 3 zones per block)
            self.assertEqual(len(concluding_nodes), 9)

            # Check tag distribution
            correct_tagged = self.find_nodes_with_tags(concluding_nodes, "Correct")
            incorrect_tagged = self.find_nodes_with_tags(concluding_nodes, "Incorrect")
            feedback_tagged = self.find_nodes_with_tags(concluding_nodes, "Feedback")

            self.assertEqual(len(correct_tagged), 1, "Should have exactly 1 Correct tagged zone")
            self.assertEqual(len(incorrect_tagged), 1, "Should have exactly 1 Incorrect tagged zone")
            self.assertEqual(len(feedback_tagged), 1, "Should have exactly 1 Feedback tagged zone")

            # Verify they're in different blocks
            self.assertEqual(correct_tagged[0].block, 0)
            self.assertEqual(incorrect_tagged[0].block, 1)
            self.assertEqual(feedback_tagged[0].block, 2)

        finally:
            self.cleanup_temp_file(temp_path)

    def test_construction_callbacks(self):
        """Test that construction callbacks can resolve placeholders."""
        temp_path = self.create_temp_file(self.philosophical_toml)

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Find zone with placeholders in solving sequence
            solving_nodes = self.collect_sequence_nodes(sequences["solving"])
            placeholder_zones = self.find_nodes_with_text(solving_nodes, "{placeholder}")

            self.assertTrue(len(placeholder_zones) > 0, "Could not find zone with placeholders")
            prompt_zone = placeholder_zones[0]

            # Create mock resources
            mock_constitution = Mock(spec=AbstractResource)
            mock_constitution.return_value = "Be kind and just"

            mock_feedback = Mock(spec=AbstractResource)
            mock_feedback.return_value = "Consider multiple perspectives"

            resources = {
                "constitution": mock_constitution,
                "feedback_backend": mock_feedback
            }

            # Test construction callback
            resolved_text = prompt_zone.construction_callback(resources)

            # Verify placeholders were resolved
            self.assertNotIn("{placeholder}", resolved_text)
            self.assertNotIn("{feedback}", resolved_text)
            self.assertIn("Be kind and just", resolved_text)
            self.assertIn("Consider multiple perspectives", resolved_text)

            # Verify resources were called correctly
            mock_constitution.assert_called_once_with()
            mock_feedback.assert_called_once_with(num_samples=3)

        finally:
            self.cleanup_temp_file(temp_path)


class TestFolderParsingIntegration(BaseIntegrationTest):
    """Test parsing UDPL across multiple files."""

    def test_split_config_and_sequences(self):
        """Test parsing when config and sequences are in separate files."""
        temp_dir = self.create_temp_dir()

        try:
            # Create config file
            config_file = temp_dir / "config.toml"
            config_file.write_text('''
[config]
zone_patterns = ["[Prompt]", "[Answer]", "[EOS]"]
required_patterns = ["[Prompt]", "[Answer]"]
valid_tags = ["Training", "Test"]
default_max_token_length = 1000
sequences = ["setup", "main"]
control_pattern = "[Jump]"
escape_patterns = ["[Escape]", "[EndEscape]"]
tools = []
''')

            # Create sequences file
            sequences_file = temp_dir / "sequences.toml"
            sequences_file.write_text('''
[[setup]]
text = """
[Prompt] Initialize the system
[Answer]
"""
tags = [[], ["Training"]]

[[main]]
text = """
[Prompt] Run the main process with {input}
[Answer]
"""
tags = [[], ["Test"]]

[main.input]
name = "input_resource"
''')

            # Parse the folder
            sequences, config = parse_udpl_folder(str(temp_dir))

            # Verify config
            self.assertIsInstance(config, Config)
            self.assertEqual(config.sequences, ["setup", "main"])

            # Verify sequences
            self.assertIn("setup", sequences)
            self.assertIn("main", sequences)

            # Verify setup sequence
            setup_nodes = self.collect_sequence_nodes(sequences["setup"])
            training_zones = self.find_nodes_with_tags(setup_nodes, "Training")
            self.assertEqual(len(training_zones), 1)

            # Verify main sequence has resource
            main_nodes = self.collect_sequence_nodes(sequences["main"])
            placeholder_zones = self.find_nodes_with_text(main_nodes, "{input}")
            self.assertEqual(len(placeholder_zones), 1)

            placeholder_zone = placeholder_zones[0]
            expected_spec = {
                "input": {
                    "name": "input_resource",
                    "arguments": None,
                    "type": "standard"
                }
            }
            self.assertEqual(placeholder_zone.resource_specs, expected_spec)

        finally:
            self.cleanup_temp_dir(temp_dir)


class TestSimpleExamples(BaseIntegrationTest):
    """Test simple UDPL examples for basic functionality."""

    def test_minimal_valid_example(self):
        """Test a minimal but complete UDPL example."""
        minimal_toml = '''
[config]
zone_patterns = ["[Prompt]", "[Answer]", "[EOS]"]
required_patterns = ["[Prompt]", "[Answer]"]
valid_tags = ["Test"]
default_max_token_length = 100
sequences = ["simple"]
control_pattern = "[Jump]"
escape_patterns = ["[Escape]", "[EndEscape]"]
tools = []

[[simple]]
text = """[Prompt] Hello world [Answer]"""
tags = [[], ["Test"]]
'''

        temp_path = self.create_temp_file(minimal_toml)

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Basic validations
            self.assertEqual(len(sequences), 1)
            self.assertIn("simple", sequences)

            # Check sequence structure
            simple_nodes = self.collect_sequence_nodes(sequences["simple"])

            # Should have 3 zones
            self.assertEqual(len(simple_nodes), 3)

            # Check first zone
            self.assertEqual(simple_nodes[0].zone_advance_str, "[Prompt]")
            self.assertEqual(simple_nodes[0].raw_text, "[Prompt]")
            self.assertEqual(simple_nodes[0].tags, [])

            # Check second zone (with prompt content)
            self.assertEqual(simple_nodes[1].zone_advance_str, "[Answer]")
            self.assertEqual(simple_nodes[1].raw_text, " Hello world [Answer]")
            self.assertEqual(simple_nodes[1].tags, [])

            # Check third zone (answer zone)
            self.assertEqual(simple_nodes[2].zone_advance_str, "[EOS]")
            self.assertEqual(simple_nodes[2].raw_text, "")
            self.assertEqual(simple_nodes[2].tags, ["Test"])

        finally:
            self.cleanup_temp_file(temp_path)

    def test_repeated_blocks(self):
        """Test blocks with repeats field."""
        repeated_toml = '''
[config]
zone_patterns = ["[Prompt]", "[Answer]", "[EOS]"]
required_patterns = ["[Prompt]", "[Answer]"]
valid_tags = ["Training"]
default_max_token_length = 100
sequences = ["repeated"]
control_pattern = "[Jump]"
escape_patterns = ["[Escape]", "[EndEscape]"]
tools = []

[[repeated]]
text = """[Prompt] Attempt {attempt} [Answer]"""
tags = [[], ["Training"]]
repeats = 3

[repeated.attempt]
name = "attempt_number"
'''

        temp_path = self.create_temp_file(repeated_toml)

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Check sequence structure
            repeated_nodes = self.collect_sequence_nodes(sequences["repeated"])

            # Should have 9 zones (3 repeats × 3 zones per repeat)
            self.assertEqual(len(repeated_nodes), 9)

            # All zones with placeholders should have same resource spec
            placeholder_zones = self.find_nodes_with_text(repeated_nodes, "{attempt}")
            self.assertEqual(len(placeholder_zones), 3)  # One per repeat

            for zone in placeholder_zones:
                expected_spec = {
                    "attempt": {
                        "name": "attempt_number",
                        "arguments": None,
                        "type": "standard"
                    }
                }
                self.assertEqual(zone.resource_specs, expected_spec)

        finally:
            self.cleanup_temp_file(temp_path)

    def test_escape_patterns_integration(self):
        """Test that escape patterns work correctly in integration."""
        escape_toml = '''
[config]
zone_patterns = ["[Prompt]", "[Answer]", "[EOS]"]
required_patterns = ["[Prompt]", "[Answer]"]
valid_tags = ["Test"]
default_max_token_length = 100
sequences = ["test"]
control_pattern = "[Jump]"
escape_patterns = ["[Escape]", "[EndEscape]"]
tools = []

[[test]]
text = """[Prompt] Use [Escape] [Jump] [EndEscape] to jump [Answer]"""
tags = [[], ["Test"]]
'''

        temp_path = self.create_temp_file(escape_toml)

        try:
            # Should parse without warnings since control pattern is escaped
            sequences, config = parse_udpl_file(temp_path)

            # Basic validation
            self.assertEqual(len(sequences), 1)
            test_nodes = self.collect_sequence_nodes(sequences["test"])

            # Find prompt zone
            prompt_zones = self.find_nodes_with_text(test_nodes, "Use")
            self.assertEqual(len(prompt_zones), 1)

            prompt_zone = prompt_zones[0]
            self.assertIn("[Escape]", prompt_zone.raw_text)
            self.assertIn("[Jump]", prompt_zone.raw_text)
            self.assertIn("[EndEscape]", prompt_zone.raw_text)

        finally:
            self.cleanup_temp_file(temp_path)


if __name__ == "__main__":
    unittest.main()