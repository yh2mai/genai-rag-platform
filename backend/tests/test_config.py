"""
Test configuration loading and basic setup
"""
import pytest
from app.core.config import settings

def test_config_loading():
    """Test that configuration loads successfully"""
    assert settings.HOST is not None
    assert settings.PORT is not None
    assert isinstance(settings.PORT, int)
    assert settings.PORT > 0

def test_default_values():
    """Test that configuration values are loaded from .env file"""
    # Values now come from .env file, not Python defaults
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8000
    # DEBUG is True in development .env file
    assert settings.DEBUG == True  
    assert settings.LOG_LEVEL == "DEBUG"  # Set to DEBUG in dev .env
    assert settings.LLM_MODEL == "deepseek-chat"
    assert settings.MAX_TOKENS == 8192

def test_llm_configuration():
    """Test that LLM configuration is properly set"""
    assert settings.LLM_MODEL is not None
    assert settings.LLM_BASE_URL is not None
    assert settings.DEEPSEEK_API_KEY is not None
    assert settings.MAX_TOKENS > 0
    assert settings.TEMPERATURE >= 0.0
    assert settings.TEMPERATURE <= 2.0

def test_paths_configuration():
    """Test that storage paths are configured"""
    assert settings.VECTOR_DB_PATH is not None
    assert settings.DOCUMENT_STORE_PATH is not None
    assert len(settings.VECTOR_DB_PATH) > 0
    assert len(settings.DOCUMENT_STORE_PATH) > 0