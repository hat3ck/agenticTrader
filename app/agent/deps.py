"""Agent dependencies — injected into every tool call via RunContext."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models.requests import InvestmentHorizon, RiskTolerance


@dataclass
class TraderDeps:
    """Dependencies available to every PydanticAI tool via ctx.deps."""

    # User request parameters
    funds: float = 10_000.0
    horizon: InvestmentHorizon = InvestmentHorizon.THREE_MONTHS
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    sector_preferences: list[str] = field(default_factory=list)
    excluded_tickers: list[str] = field(default_factory=list)

    # Resolved by the strategy registry before agent run
    selected_strategies: list[str] = field(default_factory=list)
    kelly_mode: str = "half"
    cash_reserve_pct: float = 10.0
