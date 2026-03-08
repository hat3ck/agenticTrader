"""Fundamental Analysis Tool — Financial metrics calculator.

Currently returns MOCK data.  Will be replaced with yfinance later.
Metrics: Market Cap, P/E, P/B, PEG, D/E, FCF, ROE, Dividend Yield,
Revenue Growth Rate.
"""

from __future__ import annotations

import random

from app.data.cache import fundamental_cache, async_get_or_set


async def get_fundamental_metrics(ticker: str) -> dict:
    """Calculate key fundamental metrics for a stock.

    Returns a dict containing:
    - market_cap: Total market value in USD
    - pe_ratio: Price-to-Earnings ratio
    - pb_ratio: Price-to-Book ratio
    - peg_ratio: P/E adjusted for earnings growth
    - debt_to_equity: Financial leverage ratio
    - free_cash_flow: FCF in USD
    - roe: Return on Equity (percentage)
    - dividend_yield: Annual dividend yield (percentage)
    - revenue_growth: Year-over-year revenue growth (percentage)
    - earnings_per_share: EPS in USD
    - profit_margin: Net profit margin (percentage)
    - payout_ratio: Dividend payout ratio (percentage)

    Use this to evaluate a stock's valuation, financial health, and quality.
    P/E < 20 is generally reasonable; PEG < 1 suggests undervaluation relative
    to growth; ROE > 15% signals quality; D/E > 2 is risky.
    """

    async def _fetch():
        # ── MOCK DATA — replace with yfinance later ──
        mock_fundamentals = {
            "AAPL": {"pe": 28.5, "pb": 45.2, "peg": 2.1, "de": 1.8, "fcf": 111e9, "roe": 160, "div": 0.5, "rev_g": 3.5, "eps": 6.50, "margin": 26.3, "payout": 15.0},
            "MSFT": {"pe": 35.2, "pb": 12.8, "peg": 2.0, "de": 0.4, "fcf": 63e9, "roe": 43, "div": 0.7, "rev_g": 15.2, "eps": 11.90, "margin": 36.5, "payout": 25.0},
            "GOOGL": {"pe": 24.1, "pb": 6.5, "peg": 1.3, "de": 0.1, "fcf": 69e9, "roe": 28, "div": 0.0, "rev_g": 12.8, "eps": 7.30, "margin": 25.1, "payout": 0},
            "AMZN": {"pe": 62.3, "pb": 8.2, "peg": 1.5, "de": 0.7, "fcf": 35e9, "roe": 22, "div": 0.0, "rev_g": 11.5, "eps": 3.15, "margin": 6.2, "payout": 0},
            "NVDA": {"pe": 65.0, "pb": 55.0, "peg": 0.9, "de": 0.4, "fcf": 28e9, "roe": 115, "div": 0.02, "rev_g": 122.0, "eps": 13.60, "margin": 55.0, "payout": 1.5},
            "META": {"pe": 27.5, "pb": 8.3, "peg": 1.1, "de": 0.2, "fcf": 43e9, "roe": 33, "div": 0.3, "rev_g": 24.7, "eps": 18.50, "margin": 35.0, "payout": 8.0},
            "TSLA": {"pe": 70.0, "pb": 16.0, "peg": 3.5, "de": 0.1, "fcf": 4.4e9, "roe": 22, "div": 0.0, "rev_g": 18.8, "eps": 3.50, "margin": 10.2, "payout": 0},
            "JPM": {"pe": 12.0, "pb": 1.8, "peg": 1.5, "de": 1.2, "fcf": 30e9, "roe": 17, "div": 2.4, "rev_g": 8.5, "eps": 16.50, "margin": 33.0, "payout": 27.0},
            "V": {"pe": 30.5, "pb": 13.5, "peg": 1.8, "de": 0.6, "fcf": 18e9, "roe": 47, "div": 0.7, "rev_g": 10.5, "eps": 9.35, "margin": 53.0, "payout": 22.0},
            "JNJ": {"pe": 15.2, "pb": 5.8, "peg": 2.5, "de": 0.4, "fcf": 18e9, "roe": 25, "div": 3.0, "rev_g": 4.2, "eps": 10.20, "margin": 21.5, "payout": 45.0},
            "UNH": {"pe": 22.0, "pb": 6.2, "peg": 1.5, "de": 0.7, "fcf": 22e9, "roe": 25, "div": 1.4, "rev_g": 13.0, "eps": 23.70, "margin": 6.5, "payout": 30.0},
            "HD": {"pe": 24.0, "pb": 1000, "peg": 2.2, "de": 50.0, "fcf": 16e9, "roe": 1500, "div": 2.3, "rev_g": 3.0, "eps": 15.80, "margin": 10.5, "payout": 55.0},
            "PG": {"pe": 26.0, "pb": 7.8, "peg": 3.0, "de": 0.7, "fcf": 15e9, "roe": 30, "div": 2.4, "rev_g": 3.5, "eps": 6.35, "margin": 18.5, "payout": 62.0},
            "MA": {"pe": 35.0, "pb": 60.0, "peg": 1.9, "de": 2.0, "fcf": 12e9, "roe": 180, "div": 0.5, "rev_g": 12.0, "eps": 13.15, "margin": 46.0, "payout": 18.0},
            "XOM": {"pe": 12.5, "pb": 2.1, "peg": 2.0, "de": 0.2, "fcf": 36e9, "roe": 20, "div": 3.3, "rev_g": -5.0, "eps": 8.85, "margin": 10.8, "payout": 40.0},
        }

        if ticker.upper() in mock_fundamentals:
            d = mock_fundamentals[ticker.upper()]
        else:
            # Generate random but plausible fundamentals
            d = {
                "pe": round(random.uniform(8, 80), 1),
                "pb": round(random.uniform(0.5, 20), 1),
                "peg": round(random.uniform(0.5, 4), 1),
                "de": round(random.uniform(0.1, 3), 1),
                "fcf": round(random.uniform(1e9, 50e9), 0),
                "roe": round(random.uniform(5, 50), 0),
                "div": round(random.uniform(0, 4), 1),
                "rev_g": round(random.uniform(-10, 40), 1),
                "eps": round(random.uniform(1, 20), 2),
                "margin": round(random.uniform(5, 40), 1),
                "payout": round(random.uniform(0, 60), 0),
            }

        return {
            "ticker": ticker.upper(),
            "market_cap": round(random.uniform(50e9, 3e12), 0),
            "pe_ratio": d["pe"],
            "pb_ratio": d["pb"],
            "peg_ratio": d["peg"],
            "debt_to_equity": d["de"],
            "free_cash_flow": d["fcf"],
            "roe": d["roe"],
            "dividend_yield": d["div"],
            "revenue_growth": d["rev_g"],
            "earnings_per_share": d["eps"],
            "profit_margin": d["margin"],
            "payout_ratio": d["payout"],
            "valuation_summary": _valuation_summary(d["pe"], d["peg"], d["pb"]),
            "health_summary": _health_summary(d["de"], d["fcf"]),
            "quality_summary": _quality_summary(d["roe"], d["rev_g"], d["div"]),
        }

    return await async_get_or_set(fundamental_cache, f"fund:{ticker.upper()}", _fetch)


def _valuation_summary(pe: float, peg: float, pb: float) -> str:
    parts = []
    if pe < 15:
        parts.append("attractively valued (low P/E)")
    elif pe < 25:
        parts.append("reasonably valued")
    else:
        parts.append("premium valuation (high P/E)")

    if peg < 1:
        parts.append("undervalued relative to growth (PEG < 1)")
    elif peg < 1.5:
        parts.append("fairly priced for growth")
    else:
        parts.append("growth may be priced in")

    return "; ".join(parts)


def _health_summary(de: float, fcf: float) -> str:
    parts = []
    if de < 0.5:
        parts.append("very low leverage")
    elif de < 1.5:
        parts.append("manageable debt levels")
    else:
        parts.append("high leverage — elevated financial risk")

    if fcf > 10e9:
        parts.append("strong cash generation")
    elif fcf > 0:
        parts.append("positive free cash flow")
    else:
        parts.append("negative FCF — cash burn risk")

    return "; ".join(parts)


def _quality_summary(roe: float, rev_g: float, div_yield: float) -> str:
    parts = []
    if roe > 20:
        parts.append("high-quality business (strong ROE)")
    elif roe > 10:
        parts.append("decent returns on equity")
    else:
        parts.append("low ROE — management effectiveness questionable")

    if rev_g > 15:
        parts.append("strong revenue growth")
    elif rev_g > 0:
        parts.append("moderate growth")
    else:
        parts.append("declining revenue")

    if div_yield > 2:
        parts.append(f"attractive {div_yield}% dividend yield")

    return "; ".join(parts)
