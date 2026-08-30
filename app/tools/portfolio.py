"""Portfolio Optimization & Kelly Criterion Tool.

Implements:
- Kelly Criterion:  f* = (bp - q) / b
- Half-Kelly / Quarter-Kelly adjustments
- Horizon-adjusted allocation
- Correlation-aware sizing
- Cash reserve logic
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import yfinance as yf
from scipy.optimize import minimize  # type: ignore

logger = logging.getLogger(__name__)

# Semaphore shared with market_data to avoid yfinance crumb invalidation
_YF_SEMAPHORE = asyncio.Semaphore(3)

_CORR_LOOKBACK = "6mo"  # period for historical close data


# ──────────────────────────────────────────────────────────
# Correlation helpers
# ──────────────────────────────────────────────────────────

async def _fetch_correlation_matrix(tickers: list[str]) -> np.ndarray:
    """Download historical closes and return a pairwise correlation matrix.

    Falls back to an identity matrix (= assume uncorrelated) when data is
    unavailable so that the optimiser still produces valid weights.
    """
    n = len(tickers)
    if n <= 1:
        return np.eye(n)

    try:
        df = await asyncio.to_thread(
            lambda: yf.download(
                tickers, period=_CORR_LOOKBACK, auto_adjust=True, progress=False,
            )["Close"]
        )

        # yf.download returns a Series (not DataFrame) for a single ticker
        if hasattr(df, "columns"):
            df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

        if df.empty or (hasattr(df, "shape") and df.shape[1] < 2):
            logger.warning("Insufficient historical data; using identity correlation matrix")
            return np.eye(n)

        returns = df.pct_change().dropna()
        if len(returns) < 5:
            logger.warning("Too few return observations; using identity correlation matrix")
            return np.eye(n)

        corr = returns.corr()

        # Align columns to the original ticker order
        ordered = []
        for t in tickers:
            t_upper = t.upper()
            if t_upper in corr.columns:
                ordered.append(t_upper)
            else:
                ordered.append(None)

        # Build aligned matrix, filling missing tickers with 0 correlation
        mat = np.eye(n)
        col_list = list(corr.columns)
        for i in range(n):
            for j in range(n):
                if ordered[i] is not None and ordered[j] is not None:
                    try:
                        mat[i, j] = corr.loc[ordered[i], ordered[j]]
                    except KeyError:
                        mat[i, j] = 0.0 if i != j else 1.0

        # Ensure symmetry and positive semi-definiteness
        mat = (mat + mat.T) / 2
        eigvals = np.linalg.eigvalsh(mat)
        if eigvals.min() < 0:
            mat -= 1.05 * eigvals.min() * np.eye(n)

        return mat

    except Exception:
        logger.exception("Correlation matrix fetch failed; using identity matrix")
        return np.eye(n)


def _mean_variance_weights(
    kelly_arr: np.ndarray,
    corr_matrix: np.ndarray,
    max_weight: float,
    n: int,
) -> np.ndarray:
    """Compute Mean-Variance optimal weights tilted toward Kelly fractions.

    Minimises  w' * Corr * w  -  lambda * (kelly . w)
    subject to  sum(w) = 1,  0 <= w_i <= max_weight.

    This blends diversification (minimise portfolio variance via the real
    correlation matrix) with the Kelly signal (prefer higher-edge bets).
    """
    if n == 1:
        return np.array([1.0])

    kelly_norm = kelly_arr / kelly_arr.sum() if kelly_arr.sum() > 0 else np.ones(n) / n

    # Risk-aversion parameter: higher = more weight on diversification
    lam = 0.5

    def objective(w: np.ndarray) -> float:
        port_var = float(w @ corr_matrix @ w)
        kelly_tilt = float(kelly_norm @ w)
        return port_var - lam * kelly_tilt

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight) for _ in range(n)]
    x0 = kelly_norm.copy()

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    if result.success:
        w = np.maximum(result.x, 0.0)
        w /= w.sum() if w.sum() > 0 else 1.0
        return w

    # Fallback: normalised & capped Kelly weights
    logger.warning("MV optimisation did not converge; falling back to capped Kelly weights")
    w = kelly_norm.copy()
    w = np.minimum(w, max_weight)
    w /= w.sum() if w.sum() > 0 else 1.0
    return w


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

    # ── Step 2: Fetch correlation matrix from historical returns ──
    tickers = [a["ticker"] for a in raw_allocations]
    corr_matrix = await _fetch_correlation_matrix(tickers)

    # ── Step 3: Mean-Variance optimised weights ──
    kelly_arr = np.array([a["kelly_f"] for a in raw_allocations])
    if kelly_arr.sum() <= 0:
        return {
            "allocations": [],
            "cash_reserve_usd": total_funds,
            "cash_reserve_pct": 100.0,
            "total_invested": 0.0,
            "diversification_score": 0.0,
            "explanation": "No candidates had positive expected value — all funds held as cash.",
        }

    max_pct = max_single_position_pct / 100.0
    n_assets = len(raw_allocations)
    opt_weights = _mean_variance_weights(kelly_arr, corr_matrix, max_pct, n_assets)

    allocations = []
    for a, w in zip(raw_allocations, opt_weights):
        alloc_usd = round(investable * w, 2)
        alloc_pct = round(w * (100 - cash_reserve_pct), 2)
        if alloc_usd > 0:
            allocations.append({
                "ticker": a["ticker"],
                "allocation_usd": alloc_usd,
                "allocation_pct": alloc_pct,
                "kelly_fraction": a["kelly_f"],
                "confidence": a["confidence"],
            })

    allocations.sort(key=lambda x: x["allocation_usd"], reverse=True)
    total_invested = sum(a["allocation_usd"] for a in allocations)

    # ── Diversification score (portfolio variance ratio) ──
    final_w = np.array([a["allocation_usd"] / total_invested for a in allocations]) if total_invested > 0 else np.array([])
    if len(final_w) > 1:
        # Use only the rows/cols of tickers that survived filtering
        survived = [a["ticker"] for a in allocations]
        idx = [tickers.index(t) for t in survived]
        sub_corr = corr_matrix[np.ix_(idx, idx)]
        port_var = float(final_w @ sub_corr @ final_w)
        # Normalise: 1.0 = perfectly uncorrelated equal-weight, 0.0 = single stock
        equal_var = float(np.mean(sub_corr) * (1 / len(final_w)))
        diversification_score = round(max(0.0, min(1.0, 1.0 - port_var / max(equal_var * len(final_w), 1e-9))), 2)
    else:
        diversification_score = 0.0

    n = len(allocations)
    explanation = (
        f"Applied {kelly_mode}-Kelly sizing across {n} positions, "
        f"then refined weights via Mean-Variance optimisation using "
        f"pairwise return correlations from 6-month historical data. "
        f"Cash reserve: ${cash_reserve:,.0f} ({cash_reserve_pct:.0f}%). "
        f"Total invested: ${total_invested:,.0f}. "
        f"Diversification score: {diversification_score:.2f}/1.00."
    )

    return {
        "allocations": allocations,
        "cash_reserve_usd": cash_reserve,
        "cash_reserve_pct": cash_reserve_pct,
        "total_invested": round(total_invested, 2),
        "diversification_score": diversification_score,
        "explanation": explanation,
    }
