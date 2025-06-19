"""
Integration tests for UDPL Parser

These tests use real UDPL examples from the documentation to verify
the entire parsing pipeline works correctly end-to-end from TOML → zcp nodes.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock

# Import the modules under test
from src.workflow_forge.parsing.main_parser import parse_udpl_file, parse_udpl_folder
from src.workflow_forge.parsing.config_parsing import Config
from src.workflow_forge.zcp.nodes import ZCPNode
from src.workflow_forge.resources import AbstractResource


class TestPhilosophicalSelfPlayExample(unittest.TestCase):
    """Test parsing the main philosophical self-play example from documentation."""

    def setUp(self):
        """Set up the philosophical self-play TOML content."""
        self.philosophical_toml = '''
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags = ["Training", "Correct", "Incorrect", "Feedback"]
default_max_token_length = 20000
sequences = ["setup", "loop", "solving", "concluding"]
control_token = "[Jump]"
escape_token = "[Escape]"

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
with your current answer emit [Escape] "[Jump]". Otherwise
answer "try again". If this is your first time seeing this,
just say "proceed". Repeat at least {min} times and at most
{max} times.
[Answer]
"""
tags = [[],[]]

[loop.min]
name = "min_control_resource"
type = "control"
[loop.max]
name = "max_control_resource"
type = "control"

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
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(self.philosophical_toml)
            temp_path = f.name

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Verify config structure
            self.assertIsInstance(config, Config)
            self.assertEqual(config.zone_tokens, ["[Prompt]", "[Answer]", "[EOS]"])
            self.assertEqual(config.required_tokens, ["[Prompt]", "[Answer]"])
            self.assertEqual(config.valid_tags, ["Training", "Correct", "Incorrect", "Feedback"])
            self.assertEqual(config.default_max_token_length, 20000)
            self.assertEqual(config.sequences, ["setup", "loop", "solving", "concluding"])
            self.assertEqual(config.control_token, "[Jump]")
            self.assertEqual(config.escape_token, "[Escape]")
            self.assertEqual(config.num_zones_per_block, 2)  # 3 zone tokens - 1

        finally:
            Path(temp_path).unlink()

    def test_sequence_structure(self):
        """Test that all sequences are parsed and have correct structure."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(self.philosophical_toml)
            temp_path = f.name

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Verify all sequences exist
            self.assertEqual(set(sequences.keys()), {"setup", "loop", "solving", "concluding"})

            # Verify all sequence heads are ZCPNodes
            for seq_name, seq_head in sequences.items():
                self.assertIsInstance(seq_head, ZCPNode)

        finally:
            Path(temp_path).unlink()

    def test_setup_sequence_structure(self):
        """Test setup sequence has correct zone structure and content."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(self.philosophical_toml)
            temp_path = f.name

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Walk through setup sequence chain
            setup_nodes = []
            current = sequences["setup"]
            while current is not None:
                setup_nodes.append(current)
                current = current.next_zone

            # Should have 6 zones total (2 blocks × 3 zones per block)
            self.assertEqual(len(setup_nodes), 6)

            # Check first block zones
            # Zone 0: Initial trigger zone
            self.assertEqual(setup_nodes[0].zone_advance_token, "[Prompt]")
            self.assertEqual(setup_nodes[0].raw_text, "[Prompt]")
            self.assertEqual(setup_nodes[0].tags, [])
            self.assertEqual(setup_nodes[0].sequence, "setup")
            self.assertEqual(setup_nodes[0].block, 0)

            # Zone 1: First block prompt→answer
            self.assertEqual(setup_nodes[1].zone_advance_token, "[Answer]")
            self.assertIn("Think of an interesting philosophical scenario", setup_nodes[1].raw_text)
            self.assertIn("[Answer]", setup_nodes[1].raw_text)
            self.assertEqual(setup_nodes[1].tags, [])
            self.assertEqual(setup_nodes[1].sequence, "setup")
            self.assertEqual(setup_nodes[1].block, 0)

            # Zone 2: First block answer→eos
            self.assertEqual(setup_nodes[2].zone_advance_token, "[EOS]")
            self.assertEqual(setup_nodes[2].raw_text, "\n")
            self.assertEqual(setup_nodes[2].tags, ["Training"])

            # Zone 3: Second block initial
            self.assertEqual(setup_nodes[3].zone_advance_token, "[Prompt]")
            self.assertEqual(setup_nodes[3].raw_text, "[Prompt]")
            self.assertEqual(setup_nodes[3].tags, [])
            self.assertEqual(setup_nodes[3].sequence, "setup")
            self.assertEqual(setup_nodes[3].block, 1)

            # Zone 4: Second block prompt→answer
            self.assertEqual(setup_nodes[4].zone_advance_token, "[Answer]")
            self.assertIn("Clearly state the philosophical scenario", setup_nodes[4].raw_text)
            self.assertEqual(setup_nodes[4].tags, [])

            # Zone 5: Second block answer→eos
            self.assertEqual(setup_nodes[5].zone_advance_token, "[EOS]")
            self.assertEqual(setup_nodes[5].tags, ["Training"])

        finally:
            Path(temp_path).unlink()

    def test_loop_sequence_resources(self):
        """Test loop sequence has correct resource specifications."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(self.philosophical_toml)
            temp_path = f.name

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Walk through loop sequence to find zone with placeholders
            current = sequences["loop"]
            prompt_zone = None
            while current is not None:
                if "{min}" in current.raw_text and "{max}" in current.raw_text:
                    prompt_zone = current
                    break
                current = current.next_zone

            self.assertIsNotNone(prompt_zone, "Could not find zone with {min} and {max} placeholders")

            # Check resource specifications
            expected_specs = {
                "min": {
                    "name": "min_control_resource",
                    "arguments": None,
                    "type": "control"
                },
                "max": {
                    "name": "max_control_resource",
                    "arguments": None,
                    "type": "control"
                }
            }
            self.assertEqual(prompt_zone.resource_specs, expected_specs)

            # Verify text contains placeholders
            self.assertIn("{min}", prompt_zone.raw_text)
            self.assertIn("{max}", prompt_zone.raw_text)
            self.assertIn("[Escape]", prompt_zone.raw_text)
            self.assertIn("[Jump]", prompt_zone.raw_text)

        finally:
            Path(temp_path).unlink()

    def test_solving_sequence_resources(self):
        """Test solving sequence has correct resource specifications."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(self.philosophical_toml)
            temp_path = f.name

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Find zone with placeholders in solving sequence
            current = sequences["solving"]
            prompt_zone = None
            while current is not None:
                if "{placeholder}" in current.raw_text and "{feedback}" in current.raw_text:
                    prompt_zone = current
                    break
                current = current.next_zone

            self.assertIsNotNone(prompt_zone, "Could not find zone with placeholders")

            # Check resource specifications
            expected_specs = {
                "placeholder": {
                    "name": "constitution",
                    "arguments": None,
                    "type": "default"
                },
                "feedback": {
                    "name": "feedback_backend",
                    "arguments": {"num_samples": 3},
                    "type": "default"
                }
            }
            self.assertEqual(prompt_zone.resource_specs, expected_specs)

        finally:
            Path(temp_path).unlink()

    def test_concluding_sequence_tags(self):
        """Test concluding sequence has correct tag distribution."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(self.philosophical_toml)
            temp_path = f.name

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Walk through concluding sequence
            concluding_nodes = []
            current = sequences["concluding"]
            while current is not None:
                concluding_nodes.append(current)
                current = current.next_zone

            # Should have 9 zones (3 blocks × 3 zones per block)
            self.assertEqual(len(concluding_nodes), 9)

            # Check tag distribution
            correct_tagged = [node for node in concluding_nodes if "Correct" in node.tags]
            incorrect_tagged = [node for node in concluding_nodes if "Incorrect" in node.tags]
            feedback_tagged = [node for node in concluding_nodes if "Feedback" in node.tags]

            self.assertEqual(len(correct_tagged), 1, "Should have exactly 1 Correct tagged zone")
            self.assertEqual(len(incorrect_tagged), 1, "Should have exactly 1 Incorrect tagged zone")
            self.assertEqual(len(feedback_tagged), 1, "Should have exactly 1 Feedback tagged zone")

            # Verify they're in different blocks
            self.assertEqual(correct_tagged[0].block, 0)
            self.assertEqual(incorrect_tagged[0].block, 1)
            self.assertEqual(feedback_tagged[0].block, 2)

        finally:
            Path(temp_path).unlink()

    def test_construction_callbacks(self):
        """Test that construction callbacks can resolve placeholders."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(self.philosophical_toml)
            temp_path = f.name

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Find zone with placeholders in solving sequence
            current = sequences["solving"]
            prompt_zone = None
            while current is not None:
                if "{placeholder}" in current.raw_text:
                    prompt_zone = current
                    break
                current = current.next_zone

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
            Path(temp_path).unlink()


class TestFolderParsingIntegration(unittest.TestCase):
    """Test parsing UDPL across multiple files."""

    def test_split_config_and_sequences(self):
        """Test parsing when config and sequences are in separate files."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)

        try:
            # Create config file
            config_file = temp_path / "config.toml"
            config_file.write_text('''
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags = ["Training", "Test"]
default_max_token_length = 1000
sequences = ["setup", "main"]
control_token = "[Jump]"
escape_token = "[Escape]"
''')

            # Create sequences file
            sequences_file = temp_path / "sequences.toml"
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
            sequences, config = parse_udpl_folder(str(temp_path))

            # Verify config
            self.assertIsInstance(config, Config)
            self.assertEqual(config.sequences, ["setup", "main"])

            # Verify sequences
            self.assertIn("setup", sequences)
            self.assertIn("main", sequences)

            # Verify setup sequence
            setup_node = sequences["setup"]
            setup_nodes = []
            current = setup_node
            while current is not None:
                setup_nodes.append(current)
                current = current.next_zone

            # Find tagged zone
            training_zones = [node for node in setup_nodes if "Training" in node.tags]
            self.assertEqual(len(training_zones), 1)

            # Verify main sequence has resource
            main_node = sequences["main"]
            main_nodes = []
            current = main_node
            while current is not None:
                main_nodes.append(current)
                current = current.next_zone

            # Find zone with placeholder
            placeholder_zones = [node for node in main_nodes if "{input}" in node.raw_text]
            self.assertEqual(len(placeholder_zones), 1)

            placeholder_zone = placeholder_zones[0]
            expected_spec = {
                "input": {
                    "name": "input_resource",
                    "arguments": None,
                    "type": "default"
                }
            }
            self.assertEqual(placeholder_zone.resource_specs, expected_spec)

        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestSimpleExamples(unittest.TestCase):
    """Test simple UDPL examples for basic functionality."""

    def test_minimal_valid_example(self):
        """Test a minimal but complete UDPL example."""
        minimal_toml = '''
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags = ["Test"]
default_max_token_length = 100
sequences = ["simple"]
control_token = "[Jump]"
escape_token = "[Escape]"

[[simple]]
text = """[Prompt] Hello world [Answer]"""
tags = [[], ["Test"]]
'''

        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(minimal_toml)
            temp_path = f.name

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Basic validations
            self.assertEqual(len(sequences), 1)
            self.assertIn("simple", sequences)

            # Check sequence structure
            simple_nodes = []
            current = sequences["simple"]
            while current is not None:
                simple_nodes.append(current)
                current = current.next_zone

            # Should have 3 zones
            self.assertEqual(len(simple_nodes), 3)

            # Check first zone
            self.assertEqual(simple_nodes[0].zone_advance_token, "[Prompt]")
            self.assertEqual(simple_nodes[0].raw_text, "[Prompt]")
            self.assertEqual(simple_nodes[0].tags, [])

            # Check second zone (with prompt content)
            self.assertEqual(simple_nodes[1].zone_advance_token, "[Answer]")
            self.assertEqual(simple_nodes[1].raw_text, " Hello world [Answer]")
            self.assertEqual(simple_nodes[1].tags, [])

            # Check third zone (answer zone)
            self.assertEqual(simple_nodes[2].zone_advance_token, "[EOS]")
            self.assertEqual(simple_nodes[2].raw_text, "")
            self.assertEqual(simple_nodes[2].tags, ["Test"])

        finally:
            Path(temp_path).unlink()

    def test_repeated_blocks(self):
        """Test blocks with repeats field."""
        repeated_toml = '''
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags = ["Training"]
default_max_token_length = 100
sequences = ["repeated"]
control_token = "[Jump]"
escape_token = "[Escape]"

[[repeated]]
text = """[Prompt] Attempt {attempt} [Answer]"""
tags = [[], ["Training"]]
repeats = 3

[repeated.attempt]
name = "attempt_number"
'''

        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(repeated_toml)
            temp_path = f.name

        try:
            sequences, config = parse_udpl_file(temp_path)

            # Check sequence structure
            repeated_nodes = []
            current = sequences["repeated"]
            while current is not None:
                repeated_nodes.append(current)
                current = current.next_zone

            # Should have 9 zones (3 repeats × 3 zones per repeat)
            self.assertEqual(len(repeated_nodes), 9)

            # All zones with placeholders should have same resource spec
            placeholder_zones = [node for node in repeated_nodes if "{attempt}" in node.raw_text]
            self.assertEqual(len(placeholder_zones), 3)  # One per repeat

            for zone in placeholder_zones:
                expected_spec = {
                    "attempt": {
                        "name": "attempt_number",
                        "arguments": None,
                        "type": "default"
                    }
                }
                self.assertEqual(zone.resource_specs, expected_spec)

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    unittest.main()