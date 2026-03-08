"""Response schemas for the Agentic Trader API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class KeyMetrics(BaseModel):
    """Subset of metrics surfaced per recommendation."""
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    peg_ratio: float | None = None
    roe: float | None = None
    debt_to_equity: float | None = None
    free_cash_flow: float | None = None
    rsi: float | None = None
    macd_signal: str | None = None  # "bullish" / "bearish" / "neutral"
    bollinger_position: str | None = None  # "upper" / "middle" / "lower"
    sma_trend: str | None = None  # "golden_cross" / "death_cross" / "neutral"


class StockRecommendation(BaseModel):
    """A single stock recommendation with allocation."""
    ticker: str
    company_name: str = ""
    allocation_usd: float = Field(..., ge=0, description="Dollar amount to allocate")
    allocation_pct: float = Field(..., ge=0, le=100, description="Percentage of total funds")
    confidence: float = Field(..., ge=0, le=1, description="Agent confidence 0-1")
    rationale: str = Field(..., description="Natural-language explanation for picking this stock")
    strategy_applied: str = ""
    key_metrics: KeyMetrics = Field(default_factory=KeyMetrics)


class TradeRecommendationResponse(BaseModel):
    """Full response from POST /api/v1/suggest."""
    strategy_applied: str = Field(..., description="Primary strategy or blend used")
    macro_context: str = Field(default="", description="Summary of current macro environment")
    recommendations: list[StockRecommendation] = Field(default_factory=list)
    cash_reserve_usd: float = Field(..., ge=0)
    cash_reserve_pct: float = Field(..., ge=0, le=100)
    risk_warnings: list[str] = Field(default_factory=list)
    disclaimer: str = Field(
        default="This is AI-generated analysis for educational purposes only. "
        "It is NOT financial advice. Always do your own research before investing."
    )


class AnalysisResponse(BaseModel):
    """Full response from POST /api/v1/analyze/{ticker}."""
    ticker: str
    company_name: str = ""
    summary: str = ""
    fundamental_analysis: str = ""
    technical_analysis: str = ""
    sentiment_analysis: str = ""
    overall_rating: str = ""  # "strong_buy" / "buy" / "hold" / "sell" / "strong_sell"
    confidence: float = Field(default=0.5, ge=0, le=1)
    key_metrics: KeyMetrics = Field(default_factory=KeyMetrics)
    risk_warnings: list[str] = Field(default_factory=list)
    disclaimer: str = Field(
        default="This is AI-generated analysis for educational purposes only. "
        "It is NOT financial advice. Always do your own research before investing."
    )


class StrategyInfo(BaseModel):
    """Describes a codified trading strategy."""
    name: str
    best_horizon: str
    key_metrics: list[str]
    description: str


class HealthResponse(BaseModel):
    """GET /api/v1/health response."""
    status: str = "ok"
    llm_connected: bool = False
    cache_status: str = "in-memory"
    db_status: str = "ok"
