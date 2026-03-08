"""Portfolio Optimization & Kelly Criterion Tool.

Implements:
- Kelly Criterion:  f* = (bp - q) / b
- Half-Kelly / Quarter-Kelly adjustments
- Horizon-adjusted allocation
- Correlation-aware sizing
- Cash reserve logic
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize  # type: ignore


# ──────────────────────────────────────────────────────────
# Kelly Criterion
# ──────────────────────────────────────────────────────────

def kelly_fraction(
    win_prob: float,
    win_loss_ratio: float,
    mode: str = "half",
) -> float:
    """Calculate the Kelly fraction for optimal bet sizing.

    The Kelly formula: f* = (bp - q) / b
    where b = win/loss ratio, p = win probability, q = 1 - p.

    Args:
        win_prob: Estimated probability of the trade being profitable (0-1).
        win_loss_ratio: Ratio of average win to average loss.
        mode: 'full', 'half', or 'quarter' Kelly.

    Returns:
        Fraction of capital to allocate (0.0 to 1.0).
    """
    p = max(0.0, min(1.0, win_prob))
    q = 1.0 - p
    b = max(0.01, win_loss_ratio)

    f_star = (b * p - q) / b
    f_star = max(0.0, f_star)  # never go negative

    multiplier = {"full": 1.0, "half": 0.5, "quarter": 0.25}.get(mode, 0.5)
    return round(f_star * multiplier, 4)


# ──────────────────────────────────────────────────────────
# Portfolio Allocation
# ──────────────────────────────────────────────────────────

async def optimize_portfolio(
    candidates: list[dict],
    total_funds: float,
    kelly_mode: str = "half",
    cash_reserve_pct: float = 10.0,
    max_single_position_pct: float = 30.0,
) -> dict:
    """Allocate capital across candidate stocks using Kelly Criterion.

    Args:
        candidates: List of dicts, each with:
            - ticker (str)
            - confidence (float, 0-1): Agent's confidence this stock will profit
            - expected_return (float): Estimated return if profitable
            - expected_loss (float): Estimated loss if unprofitable (positive number)
            - sector (str): For correlation-aware sizing
        total_funds: Total available capital in USD.
        kelly_mode: 'full', 'half', or 'quarter' Kelly sizing.
        cash_reserve_pct: Percentage of funds to hold as cash (5-25).
        max_single_position_pct: Maximum allocation to any single stock.

    Returns a dict with:
        - allocations: List of dicts (ticker, allocation_usd, allocation_pct, kelly_f)
        - cash_reserve_usd: Dollar amount held as cash
        - cash_reserve_pct: Percentage held as cash
        - total_invested: Sum of all stock allocations
        - diversification_score: 0-1 measure of actual diversification
        - explanation: Natural-language explanation of the sizing logic

    The Kelly Criterion maximises long-run geometric growth rate.
    Half-Kelly reduces variance by ~75% while only reducing expected
    growth by ~25% — this is the recommended default.
    """
    cash_reserve_pct = max(5.0, min(25.0, cash_reserve_pct))
    cash_reserve = round(total_funds * (cash_reserve_pct / 100), 2)
    investable = total_funds - cash_reserve

    if not candidates:
        return {
            "allocations": [],
            "cash_reserve_usd": total_funds,
            "cash_reserve_pct": 100.0,
            "total_invested": 0.0,
            "diversification_score": 0.0,
            "explanation": "No candidates provided — all funds held as cash.",
        }

    # ── Step 1: Compute raw Kelly fractions ──
    raw_allocations = []
    for c in candidates:
        confidence = c.get("confidence", 0.5)
        exp_return = c.get("expected_return", 0.10)
        exp_loss = c.get("expected_loss", 0.05)
        win_loss = exp_return / exp_loss if exp_loss > 0 else 2.0

        kf = kelly_fraction(confidence, win_loss, kelly_mode)
        raw_allocations.append({
            "ticker": c["ticker"],
            "sector": c.get("sector", "Unknown"),
            "kelly_f": kf,
            "confidence": confidence,
        })

    # ── Step 2: Correlation-aware adjustment ──
    # Penalise same-sector concentration
    sector_counts: dict[str, int] = {}
    for a in raw_allocations:
        sector_counts[a["sector"]] = sector_counts.get(a["sector"], 0) + 1

    for a in raw_allocations:
        count = sector_counts.get(a["sector"], 1)
        if count > 1:
            # Reduce fraction proportionally for correlated positions
            a["kelly_f"] = round(a["kelly_f"] / (1 + 0.3 * (count - 1)), 4)

    # ── Step 3: Normalise so total ≤ investable ──
    total_raw = sum(a["kelly_f"] for a in raw_allocations)
    if total_raw <= 0:
        # No positive Kelly fractions — hold cash
        return {
            "allocations": [],
            "cash_reserve_usd": total_funds,
            "cash_reserve_pct": 100.0,
            "total_invested": 0.0,
            "diversification_score": 0.0,
            "explanation": "No candidates had positive expected value — all funds held as cash.",
        }

    # Cap individual position
    max_pct = max_single_position_pct / 100.0
    for a in raw_allocations:
        weight = a["kelly_f"] / total_raw
        if weight > max_pct:
            a["kelly_f"] = total_raw * max_pct  # cap it

    # Re-normalise after capping
    total_raw = sum(a["kelly_f"] for a in raw_allocations)

    allocations = []
    for a in raw_allocations:
        weight = a["kelly_f"] / total_raw if total_raw > 0 else 0
        alloc_usd = round(investable * weight, 2)
        alloc_pct = round(weight * (100 - cash_reserve_pct), 2)
        if alloc_usd > 0:
            allocations.append({
                "ticker": a["ticker"],
                "allocation_usd": alloc_usd,
                "allocation_pct": alloc_pct,
                "kelly_fraction": a["kelly_f"],
                "confidence": a["confidence"],
            })

    # Sort by allocation descending
    allocations.sort(key=lambda x: x["allocation_usd"], reverse=True)
    total_invested = sum(a["allocation_usd"] for a in allocations)

    # ── Diversification score ──
    weights = np.array([a["allocation_usd"] / total_invested for a in allocations]) if total_invested > 0 else np.array([])
    hhi = float(np.sum(weights ** 2)) if len(weights) > 0 else 1.0
    diversification_score = round(1.0 - hhi, 2)  # 0 = concentrated, 1 = diversified

    n = len(allocations)
    explanation = (
        f"Applied {kelly_mode}-Kelly sizing across {n} positions. "
        f"Cash reserve: ${cash_reserve:,.0f} ({cash_reserve_pct:.0f}%). "
        f"Total invested: ${total_invested:,.0f}. "
        f"Diversification score: {diversification_score:.2f}/1.00. "
        f"Same-sector positions were penalised to ensure actual diversification."
    )

    return {
        "allocations": allocations,
        "cash_reserve_usd": cash_reserve,
        "cash_reserve_pct": cash_reserve_pct,
        "total_invested": round(total_invested, 2),
        "diversification_score": diversification_score,
        "explanation": explanation,
    }
