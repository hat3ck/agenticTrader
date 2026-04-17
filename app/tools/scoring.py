"""Quantitative Scoring Engine — deterministic confidence scoring.

Replaces LLM-generated confidence with a reproducible score computed by
evaluating how many strategy thresholds each stock passes, weighted by
fundamental_weight / technical_weight / sentiment_weight from the strategy model.
"""

from __future__ import annotations

import logging

from app.strategies.models import MetricThreshold, TradingStrategy

logger = logging.getLogger(__name__)

# ── Metric → data-source category ───────────────────────────────────────────

_FUNDAMENTAL_METRICS = frozenset({
    "pe_ratio", "pb_ratio", "peg_ratio", "roe", "debt_to_equity",
    "free_cash_flow", "dividend_yield", "revenue_growth",
    "earnings_per_share", "profit_margin", "payout_ratio",
    "earnings_consistency",
})

_TECHNICAL_METRICS = frozenset({
    "rsi", "macd", "bollinger_position", "ema_crossover",
    "volume_trend", "atr", "sma_trend",
})

# Threshold metric names that differ from tool-returned dict keys.
_METRIC_KEY_MAP: dict[str, str] = {
    "macd": "macd_line",
    "volume_trend": "volume_ratio",
}

# String-valued metrics: metric → {threshold_value → set of favorable strings}
_STRING_FAVORABLE_MAP: dict[str, dict[float | None, set[str]]] = {
    # bollinger_position == 0 in Mean Reversion → "lower" is favorable
    "bollinger_position": {0: {"lower"}},
}


def _metric_category(metric: str) -> str:
    """Classify a metric as 'fundamental', 'technical', or 'sentiment'."""
    if metric in _FUNDAMENTAL_METRICS:
        return "fundamental"
    if metric in _TECHNICAL_METRICS:
        return "technical"
    return "sentiment"


def evaluate_threshold(
    threshold: MetricThreshold,
    fundamentals: dict,
    technicals: dict,
) -> bool | None:
    """Evaluate one MetricThreshold against real data.

    Returns True (passes), False (fails), or None (data unavailable).
    """
    metric = threshold.metric
    key = _METRIC_KEY_MAP.get(metric, metric)

    # Resolve data source
    cat = _metric_category(metric)
    source = fundamentals if cat == "fundamental" else technicals
    value = source.get(key)

    # ── String-valued metrics (e.g. bollinger_position) ──────────────
    if metric in _STRING_FAVORABLE_MAP:
        if value is None:
            return None
        favorable = _STRING_FAVORABLE_MAP[metric].get(threshold.favorable_value)
        if favorable is None:
            return None
        return str(value).lower() in favorable

    # ── Numeric comparison ───────────────────────────────────────────
    if value is None:
        return None

    try:
        num = float(value)
    except (TypeError, ValueError):
        return None

    op = threshold.favorable_operator
    if op == "<" and threshold.favorable_value is not None:
        return num < threshold.favorable_value
    if op == ">" and threshold.favorable_value is not None:
        return num > threshold.favorable_value
    if op == "==" and threshold.favorable_value is not None:
        return abs(num - threshold.favorable_value) < 1e-9
    if op == "between" and threshold.favorable_range is not None:
        lo, hi = threshold.favorable_range
        return lo <= num <= hi

    return None


def compute_confidence(
    strategies: list[TradingStrategy],
    fundamentals: dict,
    technicals: dict,
    sentiment_score: float,
) -> dict:
    """Compute a deterministic 0–1 confidence score for a stock.

    Algorithm
    ---------
    1. For each strategy, classify its thresholds into fundamental / technical
       buckets and evaluate each against real data.
    2. Compute *pass_rate* per bucket  (passed / evaluable).
       If a bucket has no evaluable thresholds, default to 0.5 (neutral).
    3. Sentiment pass_rate is the news sentiment_score mapped from [-1, 1]
       to [0, 1].
    4. Weighted score = fund_weight × fund_rate + tech_weight × tech_rate
                        + sent_weight × sent_rate   (normalised by total weight).
    5. Final confidence = average across all applicable strategies.

    Returns
    -------
    dict with keys ``confidence`` (float 0–1) and ``breakdown`` (list of
    per-strategy detail dicts including threshold-level results).
    """
    if not strategies:
        return {"confidence": 0.5, "breakdown": []}

    # sentiment_score ∈ [-1, 1] → [0, 1]
    sentiment_pass = max(0.0, min(1.0, (sentiment_score + 1) / 2))

    breakdown: list[dict] = []
    strategy_scores: list[float] = []

    for strategy in strategies:
        fund_results: list[dict] = []
        tech_results: list[dict] = []

        for t in strategy.thresholds:
            result = evaluate_threshold(t, fundamentals, technicals)
            entry = {
                "metric": t.metric,
                "operator": t.favorable_operator,
                "target": (
                    t.favorable_value
                    if t.favorable_range is None
                    else list(t.favorable_range)
                ),
                "passed": result,
            }
            if _metric_category(t.metric) == "fundamental":
                fund_results.append(entry)
            else:
                tech_results.append(entry)

        fund_evaluable = [r for r in fund_results if r["passed"] is not None]
        tech_evaluable = [r for r in tech_results if r["passed"] is not None]

        fund_rate = (
            sum(1 for r in fund_evaluable if r["passed"]) / len(fund_evaluable)
            if fund_evaluable
            else 0.5
        )
        tech_rate = (
            sum(1 for r in tech_evaluable if r["passed"]) / len(tech_evaluable)
            if tech_evaluable
            else 0.5
        )

        total_w = (
            strategy.fundamental_weight
            + strategy.technical_weight
            + strategy.sentiment_weight
        )
        if total_w == 0:
            score = 0.5
        else:
            score = (
                strategy.fundamental_weight * fund_rate
                + strategy.technical_weight * tech_rate
                + strategy.sentiment_weight * sentiment_pass
            ) / total_w

        strategy_scores.append(score)
        breakdown.append({
            "strategy": strategy.name,
            "score": round(score, 4),
            "fundamental_pass_rate": round(fund_rate, 4),
            "technical_pass_rate": round(tech_rate, 4),
            "sentiment_pass_rate": round(sentiment_pass, 4),
            "thresholds": fund_results + tech_results,
        })

    confidence = round(sum(strategy_scores) / len(strategy_scores), 4)
    return {"confidence": confidence, "breakdown": breakdown}
