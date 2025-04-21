from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, List
import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings.
    """
    # Project Info
    PROJECT_NAME: str = "Agent Tools Registry"
    VERSION: str = "0.1.0"
    API_VERSION_STR: str = "/api/v1"
    DESCRIPTION: str = """
    Agent Tools Registry API provides a centralized registry for managing and executing agent tools.
    """

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    WORKERS: int = 1

    # CORS Settings
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:4200",  # Angular dev server
        "http://localhost:8000",  # FastAPI dev server
        "http://localhost",
        "https://localhost",
        "http://localhost:3000",  # For potential frontend dev server
    ]

    # OpenAI Settings
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4-1106-preview"
    OPENAI_TEMPERATURE: float = 0.7

    # Tools Settings
    TOOLS_REGISTRY_URL: str = "http://localhost:8000/tools"
    TOOLS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools")
    TOOLS_TIMEOUT: int = 30  # seconds

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        case_sensitive = True
        env_file = ".env"

    @property
    def openai_api_key(self) -> str:
        return self.OPENAI_API_KEY

    @property
    def openai_model(self) -> str:
        return self.OPENAI_MODEL

    @property
    def openai_temperature(self) -> float:
        return self.OPENAI_TEMPERATURE

    @property
    def tools_registry_url(self) -> str:
        return self.TOOLS_REGISTRY_URL

    @property
    def tools_dir(self) -> str:
        return self.TOOLS_DIR

    @property
    def tools_timeout(self) -> int:
        return self.TOOLS_TIMEOUT

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

def verify_api_key() -> bool:
    """Verify that the OpenAI API key is properly configured."""
    settings = get_settings()
    return bool(settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your-api-key-here")

# Create settings instance
settings = Settings() 