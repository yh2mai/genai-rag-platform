"""
Logging configuration for Enterprise GenAI Platform
"""
import logging
import logging.config
import sys
from pathlib import Path

def setup_logging(settings=None):
    """Configure logging for the application"""
    
    # Import settings here to avoid circular imports and allow mocking
    if settings is None:
        from app.core.config import settings as app_settings
        settings = app_settings
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": settings.LOG_FORMAT,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "default",
                "stream": sys.stdout,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "detailed",
                "filename": "logs/enterprise_genai.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "": {  # Root logger
                "level": settings.LOG_LEVEL,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "fastapi": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            # Suppress pdfminer debug messages
            "pdfminer": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "pdfminer.pdfpage": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "pdfminer.pdfinterp": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "pdfminer.converter": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "pdfminer.layout": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            # Suppress pdfplumber debug messages
            "pdfplumber": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            # Suppress transformers debug messages
            "transformers": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "transformers.tokenization_utils_base": {
                "level": "ERROR",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            # Suppress sentence-transformers debug messages
            "sentence_transformers": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            # Suppress torch debug messages
            "torch": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            # Suppress urllib3 debug messages
            "urllib3": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
    }
    
    logging.config.dictConfig(logging_config)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")