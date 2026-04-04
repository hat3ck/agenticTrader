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
    dividend_investing: bool = True,
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

    # Filter out Dividend Income strategy when dividend investing is disabled
    if not dividend_investing:
        primary = [s for s in primary if s is not DIVIDEND_INCOME]
        # Ensure at least one growth-oriented strategy is included
        if GROWTH_INVESTING not in primary:
            primary.insert(0, GROWTH_INVESTING)

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


def compute_recommendation_range(
    funds: float,
    horizon: InvestmentHorizon,
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE,
) -> tuple[int, int]:
    """Return (min, max) number of stock recommendations.

    The range adapts to:
    - **Funds**: More capital allows broader diversification.
    - **Horizon**: Short-term trades favour concentration; long-term
      portfolios benefit from diversification.
    - **Risk tolerance**: Conservative investors diversify more;
      aggressive investors concentrate.
    """
    # ── Base range from available funds ─────────────────────
    if funds < 1_000:
        lo, hi = 1, 2
    elif funds < 5_000:
        lo, hi = 3, 5
    elif funds < 25_000:
        lo, hi = 4, 6
    elif funds < 100_000:
        lo, hi = 5, 9
    else:
        lo, hi = 6, 10

    # ── Horizon adjustment ─────────────────────────────────
    if horizon in (InvestmentHorizon.ONE_WEEK, InvestmentHorizon.ONE_MONTH):
        # Short-term: tighter, favour fewer picks
        hi = max(lo, hi - 2)
    elif horizon in (InvestmentHorizon.ONE_YEAR, InvestmentHorizon.THREE_YEARS_PLUS):
        # Long-term: allow a wider portfolio
        lo = min(lo + 1, hi)
        hi = hi + 2

    # ── Risk tolerance adjustment ──────────────────────────
    if risk_tolerance == RiskTolerance.CONSERVATIVE:
        # Diversify more
        lo = min(lo + 1, hi)
        hi = hi + 1
    elif risk_tolerance == RiskTolerance.AGGRESSIVE:
        # Concentrate
        hi = max(lo, hi - 1)

    # Sanity clamps
    lo = max(1, lo)
    hi = max(lo, min(hi, 20))
    return lo, hi


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
