"""Tests for strategy selection logic."""

import pytest
from app.models.requests import InvestmentHorizon, RiskTolerance
from app.strategies.registry import (
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
