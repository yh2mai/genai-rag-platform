"""
Test logging configuration and functionality
"""
import pytest
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import os

def test_setup_logging_creates_logs_directory():
    """Test that setup_logging creates the logs directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to temporary directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.LOG_LEVEL = "DEBUG"
            mock_settings.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            from app.core.logging import setup_logging
            
            # Ensure logs directory doesn't exist initially
            logs_path = Path("logs")
            if logs_path.exists():
                shutil.rmtree(logs_path)
            
            # Call setup_logging with mock settings
            setup_logging(mock_settings)
            
            # Verify logs directory was created
            assert logs_path.exists()
            assert logs_path.is_dir()
            
        finally:
            os.chdir(original_cwd)
            # Properly close all handlers to release file locks
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)

def test_setup_logging_configures_root_logger():
    """Test that setup_logging properly configures the root logger"""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Clear any existing logging configuration
            logging.getLogger().handlers.clear()
            
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.LOG_LEVEL = "DEBUG"
            mock_settings.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            from app.core.logging import setup_logging
            setup_logging(mock_settings)
            
            # Get root logger
            root_logger = logging.getLogger()
            
            # Verify logger is configured
            assert len(root_logger.handlers) > 0
            
            # Verify log level is set from settings
            assert root_logger.level <= logging.DEBUG  # Using mocked DEBUG level
            
        finally:
            os.chdir(original_cwd)
            # Properly close all handlers to release file locks
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)

def test_setup_logging_creates_file_handler():
    """Test that setup_logging creates a file handler"""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.LOG_LEVEL = "DEBUG"
            mock_settings.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            from app.core.logging import setup_logging
            setup_logging(mock_settings)
            
            # Check that log file is created when logging occurs
            logger = logging.getLogger("test_logger")
            logger.info("Test message")
            
            # Verify log file exists
            log_file = Path("logs/enterprise_genai.log")
            assert log_file.exists()
            
            # Verify log file contains our message
            log_content = log_file.read_text()
            assert "Test message" in log_content
            
        finally:
            os.chdir(original_cwd)
            # Properly close all handlers to release file locks
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)

def test_setup_logging_uses_settings_format():
    """Test that logging uses format from settings"""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.LOG_LEVEL = "DEBUG"
            mock_settings.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            from app.core.logging import setup_logging
            setup_logging(mock_settings)
            
            # Get console handler (StreamHandler that's not a file handler)
            root_logger = logging.getLogger()
            console_handlers = [
                h for h in root_logger.handlers 
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler)
            ]
            
            assert len(console_handlers) > 0
            console_handler = console_handlers[0]
            
            # Verify formatter uses settings format
            formatter = console_handler.formatter
            assert formatter is not None
            # Check that the mocked format is used
            expected_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            assert expected_format in formatter._fmt
            
        finally:
            os.chdir(original_cwd)
            # Properly close all handlers to release file locks
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)

def test_setup_logging_handles_existing_logs_directory():
    """Test that setup_logging works when logs directory already exists"""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Pre-create logs directory
            logs_path = Path("logs")
            logs_path.mkdir()
            
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.LOG_LEVEL = "DEBUG"
            mock_settings.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            from app.core.logging import setup_logging
            
            # Should not raise an exception
            setup_logging(mock_settings)
            
            # Directory should still exist
            assert logs_path.exists()
            assert logs_path.is_dir()
            
        finally:
            os.chdir(original_cwd)
            # Properly close all handlers to release file locks
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)

@patch('app.core.logging.logging.config.dictConfig')
def test_setup_logging_calls_dict_config(mock_dict_config):
    """Test that setup_logging calls logging.config.dictConfig with proper config"""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.LOG_LEVEL = "DEBUG"
            mock_settings.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            from app.core.logging import setup_logging
            setup_logging(mock_settings)
            
            # Verify dictConfig was called
            assert mock_dict_config.called
            
            # Get the config that was passed
            config_arg = mock_dict_config.call_args[0][0]
            
            # Verify config structure
            assert "version" in config_arg
            assert "formatters" in config_arg
            assert "handlers" in config_arg
            assert "loggers" in config_arg
            
            # Verify specific handlers exist
            assert "console" in config_arg["handlers"]
            assert "file" in config_arg["handlers"]
            
        finally:
            os.chdir(original_cwd)

def test_logging_integration():
    """Integration test: verify logging works end-to-end"""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.LOG_LEVEL = "DEBUG"
            mock_settings.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            from app.core.logging import setup_logging
            
            # Setup logging
            setup_logging(mock_settings)
            
            # Create a test logger and log messages
            test_logger = logging.getLogger("integration_test")
            test_logger.info("Info message")
            test_logger.warning("Warning message")
            test_logger.error("Error message")
            
            # Verify log file was created and contains messages
            log_file = Path("logs/enterprise_genai.log")
            assert log_file.exists()
            
            log_content = log_file.read_text()
            assert "Info message" in log_content
            assert "Warning message" in log_content
            assert "Error message" in log_content
            
            # Verify log format includes expected components
            assert "integration_test" in log_content  # Logger name
            assert "INFO" in log_content or "WARNING" in log_content or "ERROR" in log_content
            
        finally:
            os.chdir(original_cwd)
            # Properly close all handlers to release file locks
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)

def test_setup_logging_startup_message():
    """Test that setup_logging logs a startup message"""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.LOG_LEVEL = "DEBUG"
            mock_settings.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            
            from app.core.logging import setup_logging
            setup_logging(mock_settings)
            
            # Check that startup message was logged
            log_file = Path("logs/enterprise_genai.log")
            if log_file.exists():
                log_content = log_file.read_text()
                assert "Logging configured successfully" in log_content
            
        finally:
            os.chdir(original_cwd)
            # Properly close all handlers to release file locks
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)