"""
Tests for the CLI functionality.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ottoman_ner.cli import (
    create_parser,
    handle_train_command,
    handle_eval_command,
    handle_predict_command,
    main,
)


class TestCreateParser:
    """Tests for argument parser creation."""
    
    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "ottoman-ner"
    
    def test_parser_help(self):
        """Test that help message is available."""
        parser = create_parser()
        # Check that description exists
        assert "Ottoman Turkish" in parser.description
    
    def test_train_subcommand(self):
        """Test train subcommand parsing."""
        parser = create_parser()
        args = parser.parse_args(["train", "--config", "config.json"])
        
        assert args.command == "train"
        assert args.config == "config.json"
    
    def test_train_requires_config(self):
        """Test train subcommand requires config."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(["train"])
    
    def test_eval_subcommand(self):
        """Test eval subcommand parsing."""
        parser = create_parser()
        args = parser.parse_args([
            "eval",
            "--model-path", "models/test",
            "--test-file", "data/test.conll"
        ])
        
        assert args.command == "eval"
        assert args.model_path == "models/test"
        assert args.test_file == "data/test.conll"
        assert args.output_dir is None
    
    def test_eval_with_output_dir(self):
        """Test eval subcommand with output directory."""
        parser = create_parser()
        args = parser.parse_args([
            "eval",
            "--model-path", "models/test",
            "--test-file", "data/test.conll",
            "--output-dir", "results/"
        ])
        
        assert args.output_dir == "results/"
    
    def test_eval_requires_model_and_test(self):
        """Test eval subcommand requires model and test file."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(["eval"])
        
        with pytest.raises(SystemExit):
            parser.parse_args(["eval", "--model-path", "models/test"])
    
    def test_predict_subcommand_with_text(self):
        """Test predict subcommand with text."""
        parser = create_parser()
        args = parser.parse_args([
            "predict",
            "--text", "Sultan Abdülhamid"
        ])
        
        assert args.command == "predict"
        assert args.text == "Sultan Abdülhamid"
        assert args.input_file is None
    
    def test_predict_subcommand_with_input_file(self):
        """Test predict subcommand with input file."""
        parser = create_parser()
        args = parser.parse_args([
            "predict",
            "--input-file", "input.txt"
        ])
        
        assert args.command == "predict"
        assert args.input_file == "input.txt"
        assert args.text is None
    
    def test_predict_requires_text_or_input(self):
        """Test predict subcommand requires text or input file."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(["predict"])
    
    def test_predict_mutually_exclusive(self):
        """Test that --text and --input-file are mutually exclusive."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args([
                "predict",
                "--text", "hello",
                "--input-file", "input.txt"
            ])
    
    def test_predict_optional_model_path(self):
        """Test predict with optional model path."""
        parser = create_parser()
        args = parser.parse_args([
            "predict",
            "--model-path", "custom/model",
            "--text", "hello"
        ])
        
        assert args.model_path == "custom/model"
    
    def test_predict_with_output_file(self):
        """Test predict with output file."""
        parser = create_parser()
        args = parser.parse_args([
            "predict",
            "--text", "hello",
            "--output-file", "output.json"
        ])
        
        assert args.output_file == "output.json"
    
    def test_verbose_flag(self):
        """Test verbose flag."""
        parser = create_parser()
        args = parser.parse_args(["--verbose", "predict", "--text", "hello"])
        
        assert args.verbose is True


class TestHandleTrainCommand:
    """Tests for handle_train_command function."""
    
    def test_train_config_not_found(self, temp_dir, caplog):
        """Test training with non-existent config file."""
        caplog.set_level(logging.ERROR)
        
        args = MagicMock()
        args.config = str(temp_dir / "nonexistent.json")
        
        result = handle_train_command(args)
        
        assert result == 1
        assert "Configuration file not found" in caplog.text
    
    def test_train_invalid_json(self, temp_dir, caplog):
        """Test training with invalid JSON config."""
        caplog.set_level(logging.ERROR)
        
        config_path = temp_dir / "invalid.json"
        config_path.write_text("not valid json{", encoding="utf-8")
        
        args = MagicMock()
        args.config = str(config_path)
        
        result = handle_train_command(args)
        
        assert result == 1
        assert "Invalid JSON" in caplog.text
    
    @patch("ottoman_ner.cli.OttomanNER")
    def test_train_success(self, mock_ner_class, temp_dir, caplog):
        """Test successful training."""
        caplog.set_level(logging.INFO)
        
        # Setup mock
        mock_ner = MagicMock()
        mock_ner.train_from_config.return_value = {"eval_f1": 0.95}
        mock_ner_class.return_value = mock_ner
        
        # Create config file
        config = {
            "model": {"model_name_or_path": "test-model"},
            "data": {"train_file": "train.conll", "dev_file": "dev.conll"},
            "training": {"output_dir": "output"}
        }
        config_path = temp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        args = MagicMock()
        args.config = str(config_path)
        
        result = handle_train_command(args)
        
        assert result == 0
        assert "Training completed successfully" in caplog.text
        mock_ner.train_from_config.assert_called_once()
    
    @patch("ottoman_ner.cli.OttomanNER")
    def test_train_with_none_f1(self, mock_ner_class, temp_dir, caplog):
        """Test training when F1 score is None."""
        caplog.set_level(logging.INFO)
        
        mock_ner = MagicMock()
        mock_ner.train_from_config.return_value = {"eval_f1": None}
        mock_ner_class.return_value = mock_ner
        
        config = {"model": {}, "data": {}, "training": {}}
        config_path = temp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        args = MagicMock()
        args.config = str(config_path)
        
        result = handle_train_command(args)
        
        assert result == 0
        assert "Final F1 Score: N/A" in caplog.text


class TestHandleEvalCommand:
    """Tests for handle_eval_command function."""
    
    @patch("ottoman_ner.cli.OttomanNER")
    def test_eval_success(self, mock_ner_class, caplog):
        """Test successful evaluation."""
        caplog.set_level(logging.INFO)
        
        mock_ner = MagicMock()
        mock_ner.evaluate.return_value = {
            "overall_f1": 0.87,
            "overall_precision": 0.90,
            "overall_recall": 0.85
        }
        mock_ner_class.return_value = mock_ner
        
        args = MagicMock()
        args.model_path = "models/test"
        args.test_file = "data/test.conll"
        args.output_dir = None
        
        result = handle_eval_command(args)
        
        assert result == 0
        assert "Evaluation completed" in caplog.text
        assert "Overall F1: 0.8700" in caplog.text
        mock_ner.evaluate.assert_called_once_with(
            model_path="models/test",
            test_file="data/test.conll",
            output_dir=None
        )
    
    @patch("ottoman_ner.cli.OttomanNER")
    def test_eval_with_output_dir(self, mock_ner_class):
        """Test evaluation with output directory."""
        mock_ner = MagicMock()
        mock_ner.evaluate.return_value = {
            "overall_f1": 0.87,
            "overall_precision": 0.90,
            "overall_recall": 0.85
        }
        mock_ner_class.return_value = mock_ner
        
        args = MagicMock()
        args.model_path = "models/test"
        args.test_file = "data/test.conll"
        args.output_dir = "results/"
        
        result = handle_eval_command(args)
        
        assert result == 0
        mock_ner.evaluate.assert_called_once_with(
            model_path="models/test",
            test_file="data/test.conll",
            output_dir="results/"
        )
    
    @patch("ottoman_ner.cli.OttomanNER")
    def test_eval_failure(self, mock_ner_class, caplog):
        """Test evaluation failure."""
        caplog.set_level(logging.ERROR)
        
        mock_ner = MagicMock()
        mock_ner.evaluate.side_effect = Exception("Model not found")
        mock_ner_class.return_value = mock_ner
        
        args = MagicMock()
        args.model_path = "invalid/model"
        args.test_file = "data/test.conll"
        args.output_dir = None
        
        result = handle_eval_command(args)
        
        assert result == 1
        assert "Evaluation failed" in caplog.text


class TestHandlePredictCommand:
    """Tests for handle_predict_command function."""
    
    @patch("ottoman_ner.cli.OttomanNER")
    def test_predict_with_text(self, mock_ner_class, temp_dir, caplog):
        """Test prediction with text input."""
        caplog.set_level(logging.INFO)
        
        mock_ner = MagicMock()
        mock_ner.default_model_path = "default-model"
        mock_ner.predict.return_value = [
            {"text": "Sultan Abdülhamid", "label": "PER", "start": 0, "end": 2, "confidence": 0.95}
        ]
        mock_ner_class.return_value = mock_ner
        
        args = MagicMock()
        args.model_path = None
        args.text = "Sultan Abdülhamid"
        args.input_file = None
        args.output_file = None
        
        result = handle_predict_command(args)
        
        assert result == 0
        mock_ner.load_model.assert_called_once_with("default-model")
        mock_ner.predict.assert_called_once_with("Sultan Abdülhamid")
    
    @patch("ottoman_ner.cli.OttomanNER")
    def test_predict_with_custom_model(self, mock_ner_class, caplog):
        """Test prediction with custom model path."""
        caplog.set_level(logging.INFO)
        
        mock_ner = MagicMock()
        mock_ner.predict.return_value = []
        mock_ner_class.return_value = mock_ner
        
        args = MagicMock()
        args.model_path = "custom/model"
        args.text = "hello"
        args.input_file = None
        args.output_file = None
        
        result = handle_predict_command(args)
        
        mock_ner.load_model.assert_called_once_with("custom/model")
    
    @patch("ottoman_ner.cli.OttomanNER")
    def test_predict_with_input_file(self, mock_ner_class, temp_dir, caplog):
        """Test prediction with input file."""
        caplog.set_level(logging.INFO)
        
        # Create input file
        input_file = temp_dir / "input.txt"
        input_file.write_text("Line 1\nLine 2\n", encoding="utf-8")
        
        mock_ner = MagicMock()
        mock_ner.predict.return_value = []
        mock_ner_class.return_value = mock_ner
        
        args = MagicMock()
        args.model_path = None
        args.text = None
        args.input_file = str(input_file)
        args.output_file = None
        
        result = handle_predict_command(args)
        
        assert result == 0
        assert mock_ner.predict.call_count == 2
    
    @patch("ottoman_ner.cli.OttomanNER")
    def test_predict_input_file_not_found(self, mock_ner_class, temp_dir, caplog):
        """Test prediction with non-existent input file."""
        caplog.set_level(logging.ERROR)
        
        args = MagicMock()
        args.model_path = None
        args.text = None
        args.input_file = str(temp_dir / "nonexistent.txt")
        args.output_file = None
        
        result = handle_predict_command(args)
        
        assert result == 1
        assert "Input file not found" in caplog.text
    
    @patch("ottoman_ner.cli.OttomanNER")
    def test_predict_with_output_file(self, mock_ner_class, temp_dir, caplog):
        """Test prediction with output file."""
        caplog.set_level(logging.INFO)
        
        mock_ner = MagicMock()
        mock_ner.predict.return_value = [
            {"text": "Test", "label": "PER", "start": 0, "end": 1, "confidence": 0.9}
        ]
        mock_ner_class.return_value = mock_ner
        
        output_file = temp_dir / "output.json"
        
        args = MagicMock()
        args.model_path = None
        args.text = "Test"
        args.input_file = None
        args.output_file = str(output_file)
        
        result = handle_predict_command(args)
        
        assert result == 0
        assert output_file.exists()
        
        # Verify JSON output
        with open(output_file) as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["text"] == "Test"
    
    @patch("ottoman_ner.cli.OttomanNER")
    def test_predict_skips_empty_lines(self, mock_ner_class, temp_dir, caplog):
        """Test prediction skips empty lines in input file."""
        caplog.set_level(logging.INFO)
        
        input_file = temp_dir / "input.txt"
        input_file.write_text("Line 1\n\n   \nLine 2\n", encoding="utf-8")
        
        mock_ner = MagicMock()
        mock_ner.predict.return_value = []
        mock_ner_class.return_value = mock_ner
        
        args = MagicMock()
        args.model_path = None
        args.text = None
        args.input_file = str(input_file)
        args.output_file = None
        
        result = handle_predict_command(args)
        
        assert result == 0
        # Should only call predict for non-empty lines
        assert mock_ner.predict.call_count == 2
    
    @patch("ottoman_ner.cli.OttomanNER")
    def test_predict_failure(self, mock_ner_class, caplog):
        """Test prediction failure."""
        caplog.set_level(logging.ERROR)
        
        mock_ner = MagicMock()
        mock_ner.load_model.side_effect = Exception("Model error")
        mock_ner_class.return_value = mock_ner
        
        args = MagicMock()
        args.model_path = None
        args.text = "hello"
        args.input_file = None
        args.output_file = None
        
        result = handle_predict_command(args)
        
        assert result == 1
        assert "Prediction failed" in caplog.text


class TestMain:
    """Tests for main entry point."""
    
    def test_main_no_command(self, capsys):
        """Test main with no command prints help."""
        # When no command is given, the CLI prints help and returns 1
        with patch("sys.argv", ["ottoman-ner"]):
            result = main()
        
        # Should return 1 when no command given
        assert result == 1
    
    def test_main_help(self, capsys):
        """Test main with --help."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["ottoman-ner", "--help"]):
                main()
        
        # --help exits with code 0
        assert exc_info.value.code == 0
    
    @patch("ottoman_ner.cli.handle_train_command")
    def test_main_train_command(self, mock_train_handler):
        """Test main routes to train command."""
        mock_train_handler.return_value = 0
        
        with patch("sys.argv", ["ottoman-ner", "train", "--config", "config.json"]):
            result = main()
        
        assert result == 0
        mock_train_handler.assert_called_once()
    
    @patch("ottoman_ner.cli.handle_eval_command")
    def test_main_eval_command(self, mock_eval_handler):
        """Test main routes to eval command."""
        mock_eval_handler.return_value = 0
        
        with patch("sys.argv", [
            "ottoman-ner", "eval",
            "--model-path", "models/test",
            "--test-file", "test.conll"
        ]):
            result = main()
        
        assert result == 0
        mock_eval_handler.assert_called_once()
    
    @patch("ottoman_ner.cli.handle_predict_command")
    def test_main_predict_command(self, mock_predict_handler):
        """Test main routes to predict command."""
        mock_predict_handler.return_value = 0
        
        with patch("sys.argv", ["ottoman-ner", "predict", "--text", "hello"]):
            result = main()
        
        assert result == 0
        mock_predict_handler.assert_called_once()
    
    def test_main_verbose_flag(self):
        """Test main sets verbose logging."""
        with patch("ottoman_ner.cli.handle_predict_command") as mock_handler:
            mock_handler.return_value = 0
            with patch("logging.getLogger") as mock_get_logger:
                mock_root = MagicMock()
                mock_get_logger.return_value = mock_root
                
                with patch("sys.argv", ["ottoman-ner", "--verbose", "predict", "--text", "hello"]):
                    main()
                
                # Logging level should be set to DEBUG
                mock_root.setLevel.assert_called_with(logging.DEBUG)
    
    def test_main_keyboard_interrupt(self, caplog):
        """Test main handles keyboard interrupt."""
        caplog.set_level(logging.INFO)
        
        with patch("ottoman_ner.cli.handle_predict_command") as mock_handler:
            mock_handler.side_effect = KeyboardInterrupt()
            
            with patch("sys.argv", ["ottoman-ner", "predict", "--text", "hello"]):
                result = main()
            
            assert result == 1
            assert "cancelled by user" in caplog.text
    
    def test_main_unexpected_exception(self, caplog):
        """Test main handles unexpected exceptions."""
        caplog.set_level(logging.ERROR)
        
        with patch("ottoman_ner.cli.handle_predict_command") as mock_handler:
            mock_handler.side_effect = Exception("Unexpected error")
            
            with patch("sys.argv", ["ottoman-ner", "predict", "--text", "hello"]):
                result = main()
            
            assert result == 1
            assert "Command failed" in caplog.text
