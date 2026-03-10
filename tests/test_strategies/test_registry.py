"""Tests for strategy selection logic."""

import pytest
from app.models.requests import InvestmentHorizon, RiskTolerance
from app.strategies.registry import (
    compute_recommendation_range,
    get_cash_reserve_pct,
    get_kelly_mode,
    select_strategies,
    list_all_strategies,
)


def test_short_horizon_selects_momentum():
    strategies = select_strategies(InvestmentHorizon.ONE_WEEK)
    names = [s.name for s in strategies]
    assert "Momentum Trading" in names
    assert "Mean Reversion" in names


def test_long_horizon_selects_value():
    strategies = select_strategies(InvestmentHorizon.ONE_YEAR)
    names = [s.name for s in strategies]
    assert "Value Investing" in names


def test_quality_always_included():
    for horizon in InvestmentHorizon:
        strategies = select_strategies(horizon)
        names = [s.name for s in strategies]
        assert "Quality Factor" in names


def test_conservative_risk_adjusts():
    strategies = select_strategies(InvestmentHorizon.THREE_MONTHS, RiskTolerance.CONSERVATIVE)
    # Quality Factor should be present for conservative
    names = [s.name for s in strategies]
    assert "Quality Factor" in names


def test_kelly_mode_short_horizon():
    assert get_kelly_mode(InvestmentHorizon.ONE_WEEK) == "quarter"


def test_kelly_mode_long_aggressive():
    assert get_kelly_mode(InvestmentHorizon.THREE_YEARS_PLUS, RiskTolerance.AGGRESSIVE) == "full"


def test_kelly_mode_moderate():
    assert get_kelly_mode(InvestmentHorizon.THREE_MONTHS, RiskTolerance.MODERATE) == "half"


def test_cash_reserve_short_horizon():
    pct = get_cash_reserve_pct(InvestmentHorizon.ONE_WEEK)
    assert pct >= 15.0


def test_cash_reserve_long_horizon_aggressive():
    pct = get_cash_reserve_pct(InvestmentHorizon.THREE_YEARS_PLUS, RiskTolerance.AGGRESSIVE)
    assert pct >= 5.0
    assert pct <= 10.0


def test_list_all_strategies():
    all_strats = list_all_strategies()
    assert len(all_strats) == 6
    assert all("name" in s for s in all_strats)


# ── compute_recommendation_range tests ───────────────────


class TestComputeRecommendationRange:
    """Tests for dynamic recommendation count logic."""

    def test_small_funds_few_picks(self):
        lo, hi = compute_recommendation_range(500, InvestmentHorizon.THREE_MONTHS)
        assert lo == 1
        assert hi <= 2

    def test_moderate_funds_moderate_picks(self):
        lo, hi = compute_recommendation_range(10_000, InvestmentHorizon.THREE_MONTHS)
        assert 3 <= lo
        assert hi <= 8

    def test_large_funds_many_picks(self):
        lo, hi = compute_recommendation_range(150_000, InvestmentHorizon.THREE_MONTHS)
        assert lo >= 5
        assert hi >= 10

    def test_short_horizon_reduces_count(self):
        _, hi_short = compute_recommendation_range(10_000, InvestmentHorizon.ONE_WEEK)
        _, hi_long = compute_recommendation_range(10_000, InvestmentHorizon.ONE_YEAR)
        assert hi_short < hi_long

    def test_long_horizon_increases_count(self):
        lo_short, _ = compute_recommendation_range(10_000, InvestmentHorizon.ONE_MONTH)
        lo_long, _ = compute_recommendation_range(10_000, InvestmentHorizon.THREE_YEARS_PLUS)
        assert lo_long >= lo_short

    def test_conservative_diversifies_more(self):
        _, hi_agg = compute_recommendation_range(
            10_000, InvestmentHorizon.THREE_MONTHS, RiskTolerance.AGGRESSIVE,
        )
        _, hi_con = compute_recommendation_range(
            10_000, InvestmentHorizon.THREE_MONTHS, RiskTolerance.CONSERVATIVE,
        )
        assert hi_con >= hi_agg

    def test_min_never_below_one(self):
        lo, hi = compute_recommendation_range(100, InvestmentHorizon.ONE_WEEK, RiskTolerance.AGGRESSIVE)
        assert lo >= 1
        assert hi >= lo

    def test_max_capped_at_twenty(self):
        _, hi = compute_recommendation_range(
            1_000_000, InvestmentHorizon.THREE_YEARS_PLUS, RiskTolerance.CONSERVATIVE,
        )
        assert hi <= 20
