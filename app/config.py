"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration loaded from environment / .env file."""

    llm_model: str = Field(default="openai:gemma-3", description="PydanticAI model identifier")
    llm_base_url: str | None = Field(default="http://localhost:1234/v1", description="Custom base URL for the LLM provider")

    app_env: str = "development"
    log_level: str = "INFO"

    alpha_vantage_api_key: str | None = Field(default=None, description="Alpha Vantage API key (free tier, 500 calls/day) — used as fallback when yfinance is unreliable")
    sec_edgar_user_agent: str = Field(default="AgenticTrader/0.1 (agentic-trader@example.com)", description="SEC EDGAR requires a User-Agent with app name and contact email")

    news_api_key: str | None = Field(default=None, description="NewsAPI.org API key (free tier, 100 requests/day) — recent headlines & sentiment")
    fred_api_key: str | None = Field(default=None, description="FRED API key (free, api.stlouisfed.org) — macroeconomic indicators")

    market_data_cache_ttl: int = 900       # 15 minutes
    fundamental_data_cache_ttl: int = 86400  # 24 hours
    sentiment_cache_ttl: int = 3600        # 1 hour
    macro_cache_ttl: int = 86400           # 24 hours

    database_url: str = "sqlite+aiosqlite:///./agentic_trader.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton – import this everywhere
settings = Settings()
