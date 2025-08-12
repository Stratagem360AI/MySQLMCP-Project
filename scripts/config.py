# Modularized MySQL MCP Server with LLM Integration

# ==================================================
# config.py - Configuration Management
# ==================================================

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 3306
    database: str = "test_db"
    username: str = "root"
    password: str = "password"
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Load configuration from environment variables"""
        return cls(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '3306')),
            database=os.getenv('DB_NAME', 'test_db'),
            username=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', 'password')
        )

@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str = "openai"  # openai, anthropic, ollama
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    base_url: Optional[str] = None  # For local models like Ollama
    max_tokens: int = 1000
    temperature: float = 0.1
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load LLM configuration from environment variables"""
        return cls(
            provider=os.getenv('LLM_PROVIDER', 'openai'),
            api_key=os.getenv('LLM_API_KEY'),
            model=os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
            base_url=os.getenv('LLM_BASE_URL'),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '1000')),
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.1'))
        )