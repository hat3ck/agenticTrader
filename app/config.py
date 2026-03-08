"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration loaded from environment / .env file."""

    llm_model: str = Field(default="openai:gemma-3", description="PydanticAI model identifier")
    llm_base_url: str | None = Field(default="http://localhost:1234/v1", description="Custom base URL for the LLM provider")

    app_env: str = "development"
    log_level: str = "INFO"

    market_data_cache_ttl: int = 900       # 15 minutes
    fundamental_data_cache_ttl: int = 86400  # 24 hours
    sentiment_cache_ttl: int = 3600        # 1 hour
    macro_cache_ttl: int = 86400           # 24 hours

    database_url: str = "sqlite+aiosqlite:///./agentic_trader.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton – import this everywhere
settings = Settings()
