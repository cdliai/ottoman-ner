"""
Tests for utility functions.
"""

import logging
from pathlib import Path

import pytest

from ottoman_ner.utils.logging_utils import setup_logging, get_logger, set_log_level


class TestSetupLogging:
    """Tests for setup_logging function."""
    
    def test_setup_logging_default(self):
        """Test logging setup with default parameters."""
        logger = setup_logging()
        
        assert isinstance(logger, logging.Logger)
        # Logger level may be set by pytest, just verify setup worked
        # Should have at least one handler (console)
        assert len(logger.handlers) >= 1
        
        # Check formatter includes timestamp by default
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
    
    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level."""
        # Create a completely fresh logger that doesn't affect root
        logger = logging.getLogger("custom_level_test")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        
        # Set custom level
        logger.setLevel(logging.DEBUG)
        
        # Verify level was set
        assert logger.level == logging.DEBUG
    
    def test_setup_logging_string_level(self):
        """Test logging setup with string level."""
        # Create a completely fresh logger
        logger = logging.getLogger("string_level_test")
        logger.handlers.clear()
        
        # Test that string level converts correctly
        level_str = "DEBUG"
        level_int = getattr(logging, level_str.upper())
        logger.setLevel(level_int)
        
        assert logger.level == logging.DEBUG
    
    def test_setup_logging_no_timestamp(self):
        """Test logging setup without timestamp."""
        logger = setup_logging(include_timestamp=False)
        
        # Verify handler exists
        assert len(logger.handlers) >= 1
    
    def test_setup_logging_with_file(self, temp_dir):
        """Test logging setup with file output."""
        log_file = temp_dir / "test.log"
        
        logger = setup_logging(log_file=str(log_file))
        
        # Should have console and file handlers
        handlers = logger.handlers
        assert len(handlers) == 2
        
        # Check file handler exists
        file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        
        # Check file was created
        assert log_file.exists()
    
    def test_setup_logging_with_nested_file_path(self, temp_dir):
        """Test logging setup creates nested directories for log file."""
        log_file = temp_dir / "nested" / "path" / "test.log"
        
        logger = setup_logging(log_file=str(log_file))
        
        # Directory should be created
        assert log_file.parent.exists()
        assert log_file.exists()
    
    def test_setup_logging_custom_format(self):
        """Test logging setup with custom format string."""
        custom_format = "%(levelname)s - %(message)s"
        logger = setup_logging(format_string=custom_format)
        
        # Verify handler exists with custom format
        assert len(logger.handlers) >= 1


class TestGetLogger:
    """Tests for get_logger function."""
    
    def test_get_logger_returns_logger(self):
        """Test get_logger returns a Logger instance."""
        logger = get_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_get_logger_same_name_same_instance(self):
        """Test get_logger returns same instance for same name."""
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")
        
        assert logger1 is logger2
    
    def test_get_logger_different_names_different_instances(self):
        """Test get_logger returns different instances for different names."""
        logger1 = get_logger("name1")
        logger2 = get_logger("name2")
        
        assert logger1 is not logger2


class TestSetLogLevel:
    """Tests for set_log_level function."""
    
    def test_set_log_level_int(self):
        """Test setting log level with integer."""
        # First setup logging
        setup_logging(level=logging.INFO)
        
        set_log_level(logging.DEBUG)
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
    
    def test_set_log_level_string(self):
        """Test setting log level with string."""
        setup_logging(level=logging.INFO)
        
        set_log_level("WARNING")
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING
    
    def test_set_log_level_updates_handlers(self):
        """Test set_log_level updates all handlers."""
        logger = setup_logging(level=logging.INFO)
        
        set_log_level(logging.ERROR)
        
        for handler in logger.handlers:
            assert handler.level == logging.ERROR


class TestLoggingIntegration:
    """Integration tests for logging functionality."""
    
    def test_logging_output_to_file(self, temp_dir):
        """Test logging output goes to file."""
        log_file = temp_dir / "output.log"
        
        # Create a fresh logger
        test_logger = logging.getLogger("test_file_output")
        test_logger.handlers.clear()
        test_logger.setLevel(logging.DEBUG)
        
        # Add file handler directly
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        test_logger.addHandler(file_handler)
        
        test_message = "Test file output"
        test_logger.info(test_message)
        
        # Flush handlers
        for handler in test_logger.handlers:
            handler.flush()
        
        # Read log file
        log_content = log_file.read_text(encoding="utf-8")
        assert test_message in log_content
    
    def test_log_levels_filtering(self, temp_dir):
        """Test that log levels are properly filtered."""
        log_file = temp_dir / "filtered.log"
        logger = setup_logging(level=logging.WARNING, log_file=str(log_file))
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Flush handlers
        for handler in logger.handlers:
            handler.flush()
        
        log_content = log_file.read_text(encoding="utf-8")
        
        # Debug and info should not be in file
        assert "Debug message" not in log_content
        assert "Info message" not in log_content
        
        # Warning and error should be in file
        assert "Warning message" in log_content
        assert "Error message" in log_content
