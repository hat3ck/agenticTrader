"""Codified trading strategy definitions.

Strategies are structured Pydantic models — NOT LLM prompts.
The agent references these to decide which tools to invoke,
which metrics to prioritise, and what thresholds signal buy / avoid.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MetricThreshold(BaseModel):
    """A single metric threshold rule."""
    metric: str
    favorable_operator: str  # "<", ">", "between", "=="
    favorable_value: float | None = None
    favorable_range: tuple[float, float] | None = None
    description: str = ""


class TradingStrategy(BaseModel):
    """Full definition of a codified trading strategy."""
    name: str
    description: str
    best_horizons: list[str] = Field(description="e.g. ['1_year', '3_years_plus']")
    key_metrics: list[str] = Field(description="Metrics the strategy relies on")
    tools_required: list[str] = Field(description="Tool names to invoke")
    fundamental_weight: float = Field(ge=0, le=1, description="0-1 weight for fundamentals")
    technical_weight: float = Field(ge=0, le=1, description="0-1 weight for technicals")
    sentiment_weight: float = Field(ge=0, le=1, description="0-1 weight for sentiment")
    thresholds: list[MetricThreshold] = Field(default_factory=list)
    kelly_fraction: str = Field(description="'quarter', 'half', or 'full'")


# ── Pre-defined Strategies ───────────────────────────────

VALUE_INVESTING = TradingStrategy(
    name="Value Investing",
    description=(
        "Identifies undervalued stocks trading below intrinsic value. "
        "Focuses on low P/E, low P/B, strong FCF, and margin of safety. "
        "Best for patient investors with 1-year+ horizons."
    ),
    best_horizons=["1_year", "3_years_plus"],
    key_metrics=["pe_ratio", "pb_ratio", "free_cash_flow", "roe", "debt_to_equity", "dividend_yield"],
    tools_required=["fundamentals", "sentiment", "portfolio"],
    fundamental_weight=0.7,
    technical_weight=0.1,
    sentiment_weight=0.2,
    thresholds=[
        MetricThreshold(metric="pe_ratio", favorable_operator="<", favorable_value=20,
                        description="P/E below 20 suggests reasonable valuation"),
        MetricThreshold(metric="pb_ratio", favorable_operator="<", favorable_value=3,
                        description="P/B below 3 for value stocks"),
        MetricThreshold(metric="roe", favorable_operator=">", favorable_value=15,
                        description="ROE above 15% indicates quality business"),
        MetricThreshold(metric="debt_to_equity", favorable_operator="<", favorable_value=1.5,
                        description="D/E below 1.5 for manageable leverage"),
    ],
    kelly_fraction="half",
)

GROWTH_INVESTING = TradingStrategy(
    name="Growth Investing",
    description=(
        "Targets high-growth companies with strong revenue acceleration. "
        "Accepts higher valuations if backed by robust growth rates. "
        "Best for 3-month to 3-year horizons."
    ),
    best_horizons=["3_months", "1_year", "3_years_plus"],
    key_metrics=["peg_ratio", "revenue_growth", "roe", "free_cash_flow"],
    tools_required=["fundamentals", "technicals", "sentiment", "portfolio"],
    fundamental_weight=0.5,
    technical_weight=0.25,
    sentiment_weight=0.25,
    thresholds=[
        MetricThreshold(metric="peg_ratio", favorable_operator="<", favorable_value=1.5,
                        description="PEG below 1.5 means growth isn't overpriced"),
        MetricThreshold(metric="revenue_growth", favorable_operator=">", favorable_value=15,
                        description="Revenue growth above 15% YoY"),
    ],
    kelly_fraction="half",
)

MOMENTUM_TRADING = TradingStrategy(
    name="Momentum Trading",
    description=(
        "Rides short-term price trends using technical indicators. "
        "Relies on RSI, MACD, volume, and EMA crossovers. "
        "Best for 1-week to 3-month horizons."
    ),
    best_horizons=["1_week", "1_month", "3_months"],
    key_metrics=["rsi", "macd", "ema_crossover", "volume_trend", "atr"],
    tools_required=["technicals", "sentiment", "portfolio"],
    fundamental_weight=0.1,
    technical_weight=0.7,
    sentiment_weight=0.2,
    thresholds=[
        MetricThreshold(metric="rsi", favorable_operator="between", favorable_range=(30, 70),
                        description="RSI between 30-70 for trend-following (avoid extremes)"),
        MetricThreshold(metric="macd", favorable_operator=">", favorable_value=0,
                        description="MACD above signal line indicates bullish momentum"),
    ],
    kelly_fraction="half",
)

DIVIDEND_INCOME = TradingStrategy(
    name="Dividend Income",
    description=(
        "Focuses on reliable dividend-paying stocks with sustainable payouts. "
        "Best for income-seeking investors with 1-year+ horizons."
    ),
    best_horizons=["1_year", "3_years_plus"],
    key_metrics=["dividend_yield", "payout_ratio", "free_cash_flow", "debt_to_equity"],
    tools_required=["fundamentals", "portfolio"],
    fundamental_weight=0.8,
    technical_weight=0.05,
    sentiment_weight=0.15,
    thresholds=[
        MetricThreshold(metric="dividend_yield", favorable_operator=">", favorable_value=2.0,
                        description="Dividend yield above 2% for income"),
        MetricThreshold(metric="debt_to_equity", favorable_operator="<", favorable_value=1.0,
                        description="Low leverage ensures dividend sustainability"),
    ],
    kelly_fraction="half",
)

MEAN_REVERSION = TradingStrategy(
    name="Mean Reversion",
    description=(
        "Buys oversold stocks expecting a bounce-back to the mean. "
        "Relies on Bollinger Bands, RSI extremes, and volume confirmation. "
        "Best for 1-week to 1-month horizons."
    ),
    best_horizons=["1_week", "1_month"],
    key_metrics=["rsi", "bollinger_position", "volume_trend"],
    tools_required=["technicals", "fundamentals", "portfolio"],
    fundamental_weight=0.2,
    technical_weight=0.7,
    sentiment_weight=0.1,
    thresholds=[
        MetricThreshold(metric="rsi", favorable_operator="<", favorable_value=30,
                        description="RSI below 30 signals oversold"),
        MetricThreshold(metric="bollinger_position", favorable_operator="==", favorable_value=0,
                        description="Price near or below lower Bollinger Band"),
    ],
    kelly_fraction="quarter",
)

QUALITY_FACTOR = TradingStrategy(
    name="Quality Factor",
    description=(
        "Filters for high-quality businesses with consistent earnings, "
        "strong ROE, low debt, and stable FCF. Used as a filter on top "
        "of other strategies for any horizon."
    ),
    best_horizons=["3_months", "1_year", "3_years_plus"],
    key_metrics=["roe", "debt_to_equity", "free_cash_flow", "earnings_consistency"],
    tools_required=["fundamentals"],
    fundamental_weight=0.85,
    technical_weight=0.05,
    sentiment_weight=0.1,
    thresholds=[
        MetricThreshold(metric="roe", favorable_operator=">", favorable_value=15,
                        description="Consistently high ROE signals quality"),
        MetricThreshold(metric="debt_to_equity", favorable_operator="<", favorable_value=1.0,
                        description="Low leverage for financial stability"),
    ],
    kelly_fraction="half",
)

# Convenience list of all strategies
ALL_STRATEGIES: list[TradingStrategy] = [
    VALUE_INVESTING,
    GROWTH_INVESTING,
    MOMENTUM_TRADING,
    DIVIDEND_INCOME,
    MEAN_REVERSION,
    QUALITY_FACTOR,
]
