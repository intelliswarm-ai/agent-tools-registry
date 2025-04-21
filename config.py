from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    # OpenAI API configuration
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0

    # Tools Registry configuration
    tools_registry_url: str = "http://localhost:8000/tools"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

def verify_api_key() -> bool:
    """Verify that the OpenAI API key is properly configured."""
    settings = get_settings()
    return bool(settings.openai_api_key and settings.openai_api_key.startswith("sk-")) 