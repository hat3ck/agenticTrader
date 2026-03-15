"""Request schemas for the Agentic Trader API."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class InvestmentHorizon(str, Enum):
    """Supported investment horizons."""
    ONE_WEEK = "1_week"
    ONE_MONTH = "1_month"
    THREE_MONTHS = "3_months"
    ONE_YEAR = "1_year"
    THREE_YEARS_PLUS = "3_years_plus"


class RiskTolerance(str, Enum):
    """Risk tolerance levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class DataSource(str, Enum):
    """Data source for stock universe constituents."""
    AUTO = "auto"            # Default fallback chain: FMP → Wikipedia → Hardcoded
    FMP = "fmp"              # FMP API only (requires FMP_API_KEY)
    WIKIPEDIA = "wikipedia"  # Wikipedia scraping only
    HARDCODED = "hardcoded"  # Built-in static universe


class SuggestRequest(BaseModel):
    """POST /api/v1/suggest — main recommendation request."""
    funds: float = Field(..., gt=0, description="Available capital in USD")
    horizon: InvestmentHorizon = Field(..., description="Investment time horizon")
    risk_tolerance: RiskTolerance = Field(default=RiskTolerance.MODERATE)
    sector_preferences: list[str] | None = Field(default=None, description="Optional sector preferences")
    excluded_tickers: list[str] | None = Field(default=None, description="Tickers to exclude")
    data_source: DataSource = Field(default=DataSource.AUTO, description="Data source for the stock universe (auto, fmp, wikipedia, hardcoded)")


class AnalyzeRequest(BaseModel):
    """POST /api/v1/analyze/{ticker} — deep-dive analysis."""
    horizon: InvestmentHorizon = Field(default=InvestmentHorizon.THREE_MONTHS)
    include_technicals: bool = True
    include_fundamentals: bool = True
    include_sentiment: bool = True
