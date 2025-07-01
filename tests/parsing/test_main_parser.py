"""
Unit tests for Main UDPL Parser

Tests cover:
1. File parser tests - single TOML file parsing
2. Folder parser tests - multiple TOML file parsing and merging
3. Collision detection tests - key conflicts between files
4. Core parse logic tests - internal parsing orchestration
5. Error handling tests - exception chaining and context
6. Edge cases - empty files, large folders, etc.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, mock_open

# Import the modules under test
from workflow_forge.frontend.parsing.main_parser import (
    parse_udpl_file, parse_udpl_folder, _parse, _check_for_collisions,
    _create_block_parser, UDPLParseError
)
from workflow_forge.frontend.parsing.config_parsing import Config
from workflow_forge.zcp.nodes import ZCPNode


class TestFileParser(unittest.TestCase):
    """Test single TOML file parsing functionality."""

    @patch('src.workflow_forge.parsing.main_parser._parse')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.workflow_forge.parsing.main_parser.toml.load')
    def test_valid_single_file(self, mock_toml_load, mock_file, mock_parse):
        """Test parsing a valid single TOML file."""
        # Setup mocks
        mock_toml_data = {"config": {"sequences": ["test"]}, "test": [{"text": "content"}]}
        mock_toml_load.return_value = mock_toml_data

        mock_config = Mock(spec=Config)
        mock_sequences = {"test": Mock(spec=ZCPNode)}
        mock_parse.return_value = (mock_sequences, mock_config)

        # Test
        sequences, config = parse_udpl_file("test.toml")

        # Verify
        mock_file.assert_called_once_with("test.toml", 'r')
        mock_toml_load.assert_called_once()
        mock_parse.assert_called_once_with(mock_toml_data)
        self.assertEqual(sequences, mock_sequences)
        self.assertEqual(config, mock_config)

    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_file_not_found(self, mock_file):
        """Test error when file doesn't exist."""
        with self.assertRaises(UDPLParseError) as context:
            parse_udpl_file("nonexistent.toml")

        self.assertIn("UDPL file not found: nonexistent.toml", str(context.exception))

    @patch('builtins.open', new_callable=mock_open)
    @patch('src.workflow_forge.parsing.main_parser.toml.load')
    def test_invalid_toml_syntax(self, mock_toml_load, mock_file):
        """Test error for invalid TOML syntax."""
        import toml
        mock_toml_load.side_effect = toml.TomlDecodeError("Invalid TOML", "test", 1)

        with self.assertRaises(UDPLParseError) as context:
            parse_udpl_file("invalid.toml")

        self.assertIn("Invalid TOML syntax in invalid.toml", str(context.exception))

    @patch('src.workflow_forge.parsing.main_parser._parse')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.workflow_forge.parsing.main_parser.toml.load')
    def test_parsing_error_propagation(self, mock_toml_load, mock_file, mock_parse):
        """Test that parsing errors are properly wrapped."""
        mock_toml_load.return_value = {"test": "data"}
        mock_parse.side_effect = ValueError("Parse failed")

        with self.assertRaises(UDPLParseError) as context:
            parse_udpl_file("test.toml")

        self.assertIn("Error parsing UDPL file test.toml", str(context.exception))
        self.assertIsInstance(context.exception.__cause__, ValueError)


class TestFolderParser(unittest.TestCase):
    """Test folder parsing with multiple TOML files."""

    def setUp(self):
        """Set up temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('src.workflow_forge.parsing.main_parser._parse')
    def test_valid_folder_single_file(self, mock_parse):
        """Test folder with single TOML file."""
        # Create test file
        test_file = self.temp_path / "config.toml"
        test_file.write_text("""
[config]
sequences = ["test"]
zone_tokens = ["[A]", "[B]"]
required_tokens = ["[A]"]
valid_tags = ["tag"]
default_max_token_length = 100
control_token = "[Jump]"
escape_token = "[Escape]"

[[test]]
text = "content"
tags = [[]]
""")

        mock_config = Mock(spec=Config)
        mock_sequences = {"test": Mock(spec=ZCPNode)}
        mock_parse.return_value = (mock_sequences, mock_config)

        # Test
        sequences, config = parse_udpl_folder(str(self.temp_path))

        # Verify
        mock_parse.assert_called_once()
        self.assertEqual(sequences, mock_sequences)
        self.assertEqual(config, mock_config)

    @patch('src.workflow_forge.parsing.main_parser._parse')
    def test_valid_folder_multiple_files(self, mock_parse):
        """Test folder with multiple non-conflicting TOML files."""
        # Create config file
        config_file = self.temp_path / "config.toml"
        config_file.write_text("""
[config]
sequences = ["setup", "main"]
zone_tokens = ["[A]", "[B]"]
required_tokens = ["[A]"]
valid_tags = ["tag"]
default_max_token_length = 100
control_token = "[Jump]"
escape_token = "[Escape]"
""")

        # Create sequences file
        sequences_file = self.temp_path / "sequences.toml"
        sequences_file.write_text("""
[[setup]]
text = "setup content"
tags = [[]]

[[main]]
text = "main content"  
tags = [[]]
""")

        mock_config = Mock(spec=Config)
        mock_sequences = {"setup": Mock(spec=ZCPNode), "main": Mock(spec=ZCPNode)}
        mock_parse.return_value = (mock_sequences, mock_config)

        # Test
        sequences, config = parse_udpl_folder(str(self.temp_path))

        # Verify _parse was called with merged data
        mock_parse.assert_called_once()
        call_args = mock_parse.call_args[0][0]  # First positional argument
        self.assertIn("config", call_args)
        self.assertIn("setup", call_args)
        self.assertIn("main", call_args)

    def test_folder_not_found(self):
        """Test error when folder doesn't exist."""
        with self.assertRaises(UDPLParseError) as context:
            parse_udpl_folder("/nonexistent/folder")

        self.assertIn("Folder not found", str(context.exception))

    def test_path_is_not_directory(self):
        """Test error when path is a file, not directory."""
        # Create a file instead of directory
        test_file = self.temp_path / "notadir.txt"
        test_file.write_text("content")

        with self.assertRaises(UDPLParseError) as context:
            parse_udpl_folder(str(test_file))

        self.assertIn("Path is not a directory", str(context.exception))

    def test_empty_folder(self):
        """Test error when folder has no TOML files."""
        # Create some non-TOML files
        (self.temp_path / "readme.txt").write_text("readme")
        (self.temp_path / "script.py").write_text("# python")

        with self.assertRaises(UDPLParseError) as context:
            parse_udpl_folder(str(self.temp_path))

        self.assertIn("No TOML files found", str(context.exception))

    @patch('src.workflow_forge.parsing.main_parser._parse')
    def test_folder_ignores_non_toml_files(self, mock_parse):
        """Test that non-TOML files are ignored."""
        # Create TOML file
        toml_file = self.temp_path / "config.toml"
        toml_file.write_text("""
[config]
sequences = []
zone_tokens = ["[A]", "[B]"]
required_tokens = ["[A]"]
valid_tags = ["tag"]
default_max_token_length = 100
control_token = "[Jump]"
escape_token = "[Escape]"
""")

        # Create non-TOML files that should be ignored
        (self.temp_path / "readme.txt").write_text("readme")
        (self.temp_path / "script.py").write_text("# python")
        (self.temp_path / "backup.toml.bak").write_text("backup")

        mock_parse.return_value = ({}, Mock(spec=Config))

        # Test
        parse_udpl_folder(str(self.temp_path))

        # Should only process the .toml file
        mock_parse.assert_called_once()


class TestCollisionDetection(unittest.TestCase):
    """Test key collision detection between TOML files."""

    def test_no_collisions(self):
        """Test when there are no key collisions."""
        new_data = {"setup": "data1", "main": "data2"}
        existing_data = {"config": "data3", "other": "data4"}
        file_sources = {"config": "file1.toml", "other": "file1.toml"}

        # Should not raise exception
        _check_for_collisions(new_data, existing_data, file_sources, "file2.toml")

    def test_single_collision(self):
        """Test detection of single key collision."""
        new_data = {"setup": "data1", "main": "data2"}
        existing_data = {"config": "data3", "setup": "data4"}  # 'setup' collision
        file_sources = {"config": "file1.toml", "setup": "file1.toml"}

        with self.assertRaises(UDPLParseError) as context:
            _check_for_collisions(new_data, existing_data, file_sources, "file2.toml")

        error_msg = str(context.exception)
        self.assertIn("Key collisions found", error_msg)
        self.assertIn("Key 'setup' defined in both file1.toml and file2.toml", error_msg)

    def test_multiple_collisions(self):
        """Test detection of multiple key collisions."""
        new_data = {"setup": "data1", "main": "data2", "config": "data3"}
        existing_data = {"setup": "data4", "other": "data5", "config": "data6"}
        file_sources = {"setup": "file1.toml", "other": "file1.toml", "config": "file1.toml"}

        with self.assertRaises(UDPLParseError) as context:
            _check_for_collisions(new_data, existing_data, file_sources, "file2.toml")

        error_msg = str(context.exception)
        self.assertIn("Key 'setup' defined in both file1.toml and file2.toml", error_msg)
        self.assertIn("Key 'config' defined in both file1.toml and file2.toml", error_msg)

    def test_config_collision(self):
        """Test specific case of config section collision."""
        new_data = {"config": {"sequences": ["new"]}}
        existing_data = {"config": {"sequences": ["old"]}}
        file_sources = {"config": "main.toml"}

        with self.assertRaises(UDPLParseError) as context:
            _check_for_collisions(new_data, existing_data, file_sources, "other.toml")

        self.assertIn("Key 'config' defined in both", str(context.exception))


class TestCoreParseLogic(unittest.TestCase):
    """Test the core _parse function logic."""

    @patch('src.workflow_forge.parsing.main_parser.parse_sequences')
    @patch('src.workflow_forge.parsing.main_parser.parse_config')
    def test_valid_complete_parsing(self, mock_parse_config, mock_parse_sequences):
        """Test successful end-to-end parsing."""
        # Setup mocks
        mock_config = Mock(spec=Config)
        mock_config.sequences = ["setup", "main"]
        mock_parse_config.return_value = mock_config

        mock_sequences = {"setup": Mock(spec=ZCPNode), "main": Mock(spec=ZCPNode)}
        mock_parse_sequences.return_value = mock_sequences

        toml_data = {"config": {"sequences": ["setup", "main"]}, "setup": [], "main": []}

        # Test
        sequences, config = _parse(toml_data)

        # Verify
        mock_parse_config.assert_called_once_with(toml_data)
        mock_parse_sequences.assert_called_once()
        self.assertEqual(sequences, mock_sequences)
        self.assertEqual(config, mock_config)

    @patch('src.workflow_forge.parsing.main_parser.parse_sequences')
    @patch('src.workflow_forge.parsing.main_parser.parse_config')
    def test_missing_sequences_from_toml(self, mock_parse_config, mock_parse_sequences):
        """Test error when sequences declared in config but not found in TOML."""
        mock_config = Mock(spec=Config)
        mock_config.sequences = ["setup", "main", "missing"]
        mock_parse_config.return_value = mock_config

        # Only return setup and main, missing 'missing'
        mock_sequences = {"setup": Mock(spec=ZCPNode), "main": Mock(spec=ZCPNode)}
        mock_parse_sequences.return_value = mock_sequences

        toml_data = {"config": {}, "setup": [], "main": []}

        with self.assertRaises(UDPLParseError) as context:
            _parse(toml_data)

        self.assertIn("Sequences declared in config but not found: missing", str(context.exception))

    @patch('src.workflow_forge.parsing.main_parser.parse_config')
    def test_config_parsing_error_propagation(self, mock_parse_config):
        """Test that config parsing errors are properly wrapped."""
        mock_parse_config.side_effect = ValueError("Config parse failed")

        with self.assertRaises(UDPLParseError) as context:
            _parse({"invalid": "data"})

        self.assertIn("Error during UDPL parsing", str(context.exception))
        self.assertIsInstance(context.exception.__cause__, ValueError)


class TestBlockParserIntegration(unittest.TestCase):
    """Test the block parser creation and integration."""

    def test_create_block_parser(self):
        """Test that block parser creation returns callable."""
        block_parser = _create_block_parser()
        self.assertTrue(callable(block_parser))

    @patch('src.workflow_forge.parsing.main_parser.parse_block')
    @patch('src.workflow_forge.parsing.main_parser.parse_zone')
    def test_block_parser_calls_parse_block(self, mock_parse_zone, mock_parse_block):
        """Test that created block parser calls parse_block correctly."""
        mock_zcp_node = Mock(spec=ZCPNode)
        mock_parse_block.return_value = mock_zcp_node

        block_parser = _create_block_parser()

        # Test calling the block parser
        block_data = {"text": "test", "tags": [[]]}
        config = Mock(spec=Config)

        result = block_parser(block_data, config, "test_seq", 0)

        # Verify parse_block was called with correct arguments
        mock_parse_block.assert_called_once_with(
            block_data=block_data,
            config=config,
            sequence_name="test_seq",
            block_index=0,
            zone_parser=mock_parse_zone
        )
        self.assertEqual(result, mock_zcp_node)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and exception chaining."""

    def test_exception_chaining_file_parser(self):
        """Test that file parser preserves exception chains."""
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with self.assertRaises(UDPLParseError) as context:
                parse_udpl_file("restricted.toml")

            self.assertIsInstance(context.exception.__cause__, PermissionError)

    def test_exception_chaining_folder_parser(self):
        """Test that folder parser preserves exception chains."""
        with patch('pathlib.Path.exists', side_effect=OSError("Filesystem error")):
            with self.assertRaises(UDPLParseError) as context:
                parse_udpl_folder("/some/path")

            self.assertIsInstance(context.exception.__cause__, OSError)

    def test_error_context_includes_file_path(self):
        """Test that error messages include file path context."""
        with patch('builtins.open', side_effect=FileNotFoundError()):
            with self.assertRaises(UDPLParseError) as context:
                parse_udpl_file("specific_file.toml")

            self.assertIn("specific_file.toml", str(context.exception))

    def test_error_context_includes_folder_path(self):
        """Test that error messages include folder path context."""
        with patch('pathlib.Path.exists', return_value=False):
            with self.assertRaises(UDPLParseError) as context:
                parse_udpl_folder("/specific/folder")

            self.assertIn("/specific/folder", str(context.exception))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and unusual scenarios."""

    def setUp(self):
        """Set up temporary directory for edge case testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('src.workflow_forge.parsing.main_parser._parse')
    def test_empty_toml_file(self, mock_parse):
        """Test handling of valid but empty TOML file."""
        test_file = self.temp_path / "empty.toml"
        test_file.write_text("")  # Valid TOML but empty

        mock_parse.return_value = ({}, Mock(spec=Config))

        # Should not crash, just pass empty dict to _parse
        parse_udpl_file(str(test_file))
        mock_parse.assert_called_once_with({})

    @patch('src.workflow_forge.parsing.main_parser._parse')
    def test_large_folder_many_files(self, mock_parse):
        """Test folder with many TOML files."""
        # Create 10 TOML files
        for i in range(10):
            test_file = self.temp_path / f"file_{i}.toml"
            test_file.write_text(f'section_{i} = "content_{i}"')

        mock_parse.return_value = ({}, Mock(spec=Config))

        # Should handle many files without issue
        parse_udpl_folder(str(self.temp_path))

        # Verify all files were merged
        mock_parse.assert_called_once()
        merged_data = mock_parse.call_args[0][0]

        # Should have all 10 sections
        for i in range(10):
            self.assertIn(f"section_{i}", merged_data)

    @patch('src.workflow_forge.parsing.main_parser._parse')
    def test_toml_with_only_non_sequence_data(self, mock_parse):
        """Test TOML with config but no actual sequences."""
        test_file = self.temp_path / "config_only.toml"
        test_file.write_text("""
[config]
sequences = []
zone_tokens = ["[A]", "[B]"]
required_tokens = ["[A]"]
valid_tags = ["tag"]
default_max_token_length = 100
control_token = "[Jump]"
escape_token = "[Escape]"

[other_section]
data = "value"
""")

        mock_parse.return_value = ({}, Mock(spec=Config))

        # Should parse successfully even with no sequences
        parse_udpl_file(str(test_file))
        mock_parse.assert_called_once()


if __name__ == "__main__":
    unittest.main()