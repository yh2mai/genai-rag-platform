"""
Configuration management for Enterprise GenAI Platform

This module defines the configuration schema and validation rules.
All default values are stored in .env.example file to avoid duplication.
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field
from typing import List

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    
    IMPORTANT: Default values are defined in .env.example file.
    This class only defines types and validation rules.
    """
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # Server Configuration
    HOST: str
    PORT: int  
    DEBUG: bool
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str]
    
    # Database Configuration
    DATABASE_URL: str
    
    # Vector Database Configuration
    VECTOR_DB_PATH: str
    
    # Document Storage Configuration
    DOCUMENT_STORE_PATH: str
    
    # Embedding Configuration
    EMBEDDING_MODEL: str
    EMBEDDING_DIMENSION: int
    
    # LLM Configuration
    LLM_MODEL: str
    LLM_BASE_URL: str
    MAX_TOKENS: int
    TEMPERATURE: float
    DEEPSEEK_API_KEY: str = Field(
        description="DeepSeek API key - get from https://platform.deepseek.com/",
        min_length=1
    )
    
    # Logging Configuration
    LOG_LEVEL: str
    LOG_FORMAT: str
    
    # Cache Configuration
    CACHE_TTL: int
    REDIS_URL: str
    
    # Security Configuration
    SECRET_KEY: str = Field(
        description="Secret key for cryptographic operations - generate with generate_secret_key.py",
        min_length=32
    )

# Global settings instance
settings = Settings()