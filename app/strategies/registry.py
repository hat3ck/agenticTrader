"""Strategy selection logic.

Given a user's investment horizon and risk tolerance, selects the
applicable strategies and returns them ranked by relevance.
"""

from __future__ import annotations

from app.models.requests import InvestmentHorizon, RiskTolerance
from app.strategies.models import (
    ALL_STRATEGIES,
    DIVIDEND_INCOME,
    GROWTH_INVESTING,
    MEAN_REVERSION,
    MOMENTUM_TRADING,
    QUALITY_FACTOR,
    VALUE_INVESTING,
    TradingStrategy,
)

# ── Horizon → primary strategies mapping ─────────────────

_HORIZON_STRATEGY_MAP: dict[InvestmentHorizon, list[TradingStrategy]] = {
    InvestmentHorizon.ONE_WEEK: [MOMENTUM_TRADING, MEAN_REVERSION],
    InvestmentHorizon.ONE_MONTH: [MOMENTUM_TRADING, MEAN_REVERSION],
    InvestmentHorizon.THREE_MONTHS: [MOMENTUM_TRADING, GROWTH_INVESTING],
    InvestmentHorizon.ONE_YEAR: [VALUE_INVESTING, GROWTH_INVESTING, DIVIDEND_INCOME],
    InvestmentHorizon.THREE_YEARS_PLUS: [VALUE_INVESTING, DIVIDEND_INCOME, GROWTH_INVESTING],
}

# ── Risk adjustments ─────────────────────────────────────

_RISK_ADJUSTMENTS: dict[RiskTolerance, list[str]] = {
    RiskTolerance.CONSERVATIVE: ["Value Investing", "Dividend Income", "Quality Factor"],
    RiskTolerance.MODERATE: [],  # no filtering — keep horizon selection
    RiskTolerance.AGGRESSIVE: ["Momentum Trading", "Growth Investing", "Mean Reversion"],
}


def select_strategies(
    horizon: InvestmentHorizon,
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE,
) -> list[TradingStrategy]:
    """Return ordered list of applicable strategies.

    Quality Factor is always appended as a secondary filter.
    """
    primary = list(_HORIZON_STRATEGY_MAP.get(horizon, [GROWTH_INVESTING]))

    # For conservative / aggressive, prefer strategies that match both
    # horizon AND risk preference when possible
    preferred_names = _RISK_ADJUSTMENTS.get(risk_tolerance, [])
    if preferred_names:
        # Re-order: matching strategies first, then others
        matching = [s for s in primary if s.name in preferred_names]
        others = [s for s in primary if s.name not in preferred_names]
        primary = matching + others if matching else primary

    # Always add Quality Factor as a secondary overlay if not already present
    if QUALITY_FACTOR not in primary:
        primary.append(QUALITY_FACTOR)

    return primary


def get_kelly_mode(
    horizon: InvestmentHorizon,
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE,
) -> str:
    """Return the Kelly fraction mode based on horizon + risk."""
    if horizon == InvestmentHorizon.ONE_WEEK:
        return "quarter"
    if risk_tolerance == RiskTolerance.CONSERVATIVE:
        return "quarter"
    if horizon in (InvestmentHorizon.THREE_YEARS_PLUS,) and risk_tolerance == RiskTolerance.AGGRESSIVE:
        return "full"
    return "half"


def get_cash_reserve_pct(
    horizon: InvestmentHorizon,
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE,
) -> float:
    """Recommended cash reserve percentage (5-20%)."""
    base = {
        InvestmentHorizon.ONE_WEEK: 20.0,
        InvestmentHorizon.ONE_MONTH: 15.0,
        InvestmentHorizon.THREE_MONTHS: 10.0,
        InvestmentHorizon.ONE_YEAR: 7.5,
        InvestmentHorizon.THREE_YEARS_PLUS: 5.0,
    }.get(horizon, 10.0)

    if risk_tolerance == RiskTolerance.CONSERVATIVE:
        base += 5.0
    elif risk_tolerance == RiskTolerance.AGGRESSIVE:
        base = max(base - 5.0, 5.0)

    return min(base, 25.0)


def list_all_strategies() -> list[dict]:
    """Return all strategies as serialisable dicts (for GET /strategies)."""
    return [
        {
            "name": s.name,
            "description": s.description,
            "best_horizons": s.best_horizons,
            "key_metrics": s.key_metrics,
            "fundamental_weight": s.fundamental_weight,
            "technical_weight": s.technical_weight,
            "sentiment_weight": s.sentiment_weight,
        }
        for s in ALL_STRATEGIES
    ]
