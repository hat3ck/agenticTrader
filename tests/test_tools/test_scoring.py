"""Tests for the quantitative scoring engine."""

from __future__ import annotations

import pytest

from app.strategies.models import MetricThreshold, TradingStrategy
from app.tools.scoring import compute_confidence, evaluate_threshold


# ── evaluate_threshold ───────────────────────────────────────────────────────

class TestEvaluateThreshold:
    """Unit tests for individual threshold evaluation."""

    def _fund(self, **kw) -> dict:
        return kw

    def _tech(self, **kw) -> dict:
        return kw

    def test_less_than_pass(self):
        t = MetricThreshold(metric="pe_ratio", favorable_operator="<", favorable_value=20)
        assert evaluate_threshold(t, self._fund(pe_ratio=15), {}) is True

    def test_less_than_fail(self):
        t = MetricThreshold(metric="pe_ratio", favorable_operator="<", favorable_value=20)
        assert evaluate_threshold(t, self._fund(pe_ratio=25), {}) is False

    def test_greater_than_pass(self):
        t = MetricThreshold(metric="roe", favorable_operator=">", favorable_value=15)
        assert evaluate_threshold(t, self._fund(roe=20), {}) is True

    def test_greater_than_fail(self):
        t = MetricThreshold(metric="roe", favorable_operator=">", favorable_value=15)
        assert evaluate_threshold(t, self._fund(roe=10), {}) is False

    def test_between_pass(self):
        t = MetricThreshold(metric="rsi", favorable_operator="between", favorable_range=(30, 70))
        assert evaluate_threshold(t, {}, self._tech(rsi=50)) is True

    def test_between_fail_low(self):
        t = MetricThreshold(metric="rsi", favorable_operator="between", favorable_range=(30, 70))
        assert evaluate_threshold(t, {}, self._tech(rsi=20)) is False

    def test_between_fail_high(self):
        t = MetricThreshold(metric="rsi", favorable_operator="between", favorable_range=(30, 70))
        assert evaluate_threshold(t, {}, self._tech(rsi=80)) is False

    def test_missing_data_returns_none(self):
        t = MetricThreshold(metric="pe_ratio", favorable_operator="<", favorable_value=20)
        assert evaluate_threshold(t, {}, {}) is None

    def test_macd_maps_to_macd_line(self):
        t = MetricThreshold(metric="macd", favorable_operator=">", favorable_value=0)
        assert evaluate_threshold(t, {}, self._tech(macd_line=0.5)) is True
        assert evaluate_threshold(t, {}, self._tech(macd_line=-0.3)) is False

    def test_bollinger_position_string(self):
        t = MetricThreshold(metric="bollinger_position", favorable_operator="==", favorable_value=0)
        assert evaluate_threshold(t, {}, self._tech(bollinger_position="lower")) is True
        assert evaluate_threshold(t, {}, self._tech(bollinger_position="upper")) is False
        assert evaluate_threshold(t, {}, self._tech(bollinger_position="middle")) is False

    def test_non_numeric_value_returns_none(self):
        t = MetricThreshold(metric="pe_ratio", favorable_operator="<", favorable_value=20)
        assert evaluate_threshold(t, {"pe_ratio": "N/A"}, {}) is None


# ── compute_confidence ───────────────────────────────────────────────────────

def _make_strategy(
    name: str = "Test",
    thresholds: list[MetricThreshold] | None = None,
    fund_w: float = 0.5,
    tech_w: float = 0.3,
    sent_w: float = 0.2,
) -> TradingStrategy:
    return TradingStrategy(
        name=name,
        description="test",
        best_horizons=["1_year"],
        key_metrics=[],
        tools_required=[],
        fundamental_weight=fund_w,
        technical_weight=tech_w,
        sentiment_weight=sent_w,
        thresholds=thresholds or [],
        kelly_fraction="half",
    )


class TestComputeConfidence:
    """Unit tests for the full confidence computation."""

    def test_no_strategies_returns_neutral(self):
        result = compute_confidence([], {}, {}, 0.0)
        assert result["confidence"] == 0.5

    def test_all_thresholds_pass(self):
        strategy = _make_strategy(
            thresholds=[
                MetricThreshold(metric="pe_ratio", favorable_operator="<", favorable_value=20),
                MetricThreshold(metric="roe", favorable_operator=">", favorable_value=15),
            ],
            fund_w=1.0, tech_w=0.0, sent_w=0.0,
        )
        result = compute_confidence(
            [strategy],
            {"pe_ratio": 15, "roe": 20},
            {},
            0.0,
        )
        assert result["confidence"] == 1.0

    def test_no_thresholds_pass(self):
        strategy = _make_strategy(
            thresholds=[
                MetricThreshold(metric="pe_ratio", favorable_operator="<", favorable_value=20),
                MetricThreshold(metric="roe", favorable_operator=">", favorable_value=15),
            ],
            fund_w=1.0, tech_w=0.0, sent_w=0.0,
        )
        result = compute_confidence(
            [strategy],
            {"pe_ratio": 30, "roe": 5},
            {},
            0.0,
        )
        assert result["confidence"] == 0.0

    def test_half_thresholds_pass(self):
        strategy = _make_strategy(
            thresholds=[
                MetricThreshold(metric="pe_ratio", favorable_operator="<", favorable_value=20),
                MetricThreshold(metric="roe", favorable_operator=">", favorable_value=15),
            ],
            fund_w=1.0, tech_w=0.0, sent_w=0.0,
        )
        result = compute_confidence(
            [strategy],
            {"pe_ratio": 15, "roe": 5},  # pe passes, roe fails
            {},
            0.0,
        )
        assert result["confidence"] == 0.5

    def test_sentiment_weight_positive(self):
        """Positive sentiment (score=1.0) maps to pass_rate=1.0."""
        strategy = _make_strategy(
            thresholds=[],
            fund_w=0.0, tech_w=0.0, sent_w=1.0,
        )
        result = compute_confidence([strategy], {}, {}, 1.0)
        assert result["confidence"] == 1.0

    def test_sentiment_weight_negative(self):
        """Negative sentiment (score=-1.0) maps to pass_rate=0.0."""
        strategy = _make_strategy(
            thresholds=[],
            fund_w=0.0, tech_w=0.0, sent_w=1.0,
        )
        result = compute_confidence([strategy], {}, {}, -1.0)
        assert result["confidence"] == 0.0

    def test_sentiment_neutral(self):
        """Neutral sentiment (score=0.0) maps to pass_rate=0.5."""
        strategy = _make_strategy(
            thresholds=[],
            fund_w=0.0, tech_w=0.0, sent_w=1.0,
        )
        result = compute_confidence([strategy], {}, {}, 0.0)
        assert result["confidence"] == 0.5

    def test_mixed_weights(self):
        """50% fund weight, 50% tech weight — fund passes, tech fails."""
        strategy = _make_strategy(
            thresholds=[
                MetricThreshold(metric="pe_ratio", favorable_operator="<", favorable_value=20),
                MetricThreshold(metric="rsi", favorable_operator="between", favorable_range=(30, 70)),
            ],
            fund_w=0.5, tech_w=0.5, sent_w=0.0,
        )
        result = compute_confidence(
            [strategy],
            {"pe_ratio": 15},      # pass
            {"rsi": 80},           # fail (outside 30-70)
            0.0,
        )
        assert result["confidence"] == 0.5

    def test_multiple_strategies_averaged(self):
        s1 = _make_strategy(
            name="AllPass",
            thresholds=[
                MetricThreshold(metric="pe_ratio", favorable_operator="<", favorable_value=20),
            ],
            fund_w=1.0, tech_w=0.0, sent_w=0.0,
        )
        s2 = _make_strategy(
            name="AllFail",
            thresholds=[
                MetricThreshold(metric="roe", favorable_operator=">", favorable_value=50),
            ],
            fund_w=1.0, tech_w=0.0, sent_w=0.0,
        )
        result = compute_confidence(
            [s1, s2],
            {"pe_ratio": 15, "roe": 10},
            {},
            0.0,
        )
        # s1 → 1.0, s2 → 0.0, average → 0.5
        assert result["confidence"] == 0.5

    def test_unevaluable_thresholds_default_neutral(self):
        """When data is missing for all thresholds, pass_rate defaults to 0.5."""
        strategy = _make_strategy(
            thresholds=[
                MetricThreshold(metric="pe_ratio", favorable_operator="<", favorable_value=20),
            ],
            fund_w=1.0, tech_w=0.0, sent_w=0.0,
        )
        result = compute_confidence([strategy], {}, {}, 0.0)
        assert result["confidence"] == 0.5

    def test_breakdown_included(self):
        strategy = _make_strategy(
            thresholds=[
                MetricThreshold(metric="pe_ratio", favorable_operator="<", favorable_value=20),
            ],
            fund_w=1.0, tech_w=0.0, sent_w=0.0,
        )
        result = compute_confidence([strategy], {"pe_ratio": 15}, {}, 0.0)
        assert len(result["breakdown"]) == 1
        bd = result["breakdown"][0]
        assert bd["strategy"] == "Test"
        assert bd["fundamental_pass_rate"] == 1.0
        assert len(bd["thresholds"]) == 1
        assert bd["thresholds"][0]["passed"] is True

    def test_deterministic_same_inputs_same_output(self):
        """Same inputs always produce the same confidence — key property."""
        strategy = _make_strategy(
            thresholds=[
                MetricThreshold(metric="pe_ratio", favorable_operator="<", favorable_value=20),
                MetricThreshold(metric="roe", favorable_operator=">", favorable_value=15),
                MetricThreshold(metric="rsi", favorable_operator="between", favorable_range=(30, 70)),
            ],
            fund_w=0.5, tech_w=0.3, sent_w=0.2,
        )
        fund = {"pe_ratio": 18, "roe": 12}
        tech = {"rsi": 55}
        for _ in range(10):
            r = compute_confidence([strategy], fund, tech, 0.3)
            assert r["confidence"] == r["confidence"]  # no NaN
        results = [compute_confidence([strategy], fund, tech, 0.3)["confidence"] for _ in range(10)]
        assert len(set(results)) == 1

    def test_with_real_value_investing_strategy(self):
        """Smoke test using the actual VALUE_INVESTING strategy."""
        from app.strategies.models import VALUE_INVESTING

        fund = {"pe_ratio": 15, "pb_ratio": 2, "roe": 20, "debt_to_equity": 0.8}
        result = compute_confidence([VALUE_INVESTING], fund, {}, 0.2)
        # All 4 thresholds pass, sentiment slightly positive → high confidence
        assert result["confidence"] > 0.7
