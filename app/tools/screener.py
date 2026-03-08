"""Screening & Filtering Tool — Pre-filter the stock universe.

Currently returns MOCK data.  Will be integrated with a proper data
source later.
"""

from __future__ import annotations

import random

from app.data.cache import screener_cache, async_get_or_set


async def screen_stocks(
    risk_tolerance: str = "moderate",
    sector_preferences: list[str] | None = None,
    excluded_tickers: list[str] | None = None,
    min_market_cap: float = 10e9,
    max_results: int = 25,
) -> dict:
    """Screen the stock universe and return a filtered list of candidates.

    Args:
        risk_tolerance: 'conservative', 'moderate', or 'aggressive'
        sector_preferences: Optional list of preferred sectors
        excluded_tickers: Tickers to exclude from results
        min_market_cap: Minimum market cap filter in USD (default $10B)
        max_results: Maximum number of candidates to return

    Returns a dict with:
    - candidates: List of ticker dicts (ticker, name, sector, market_cap)
    - total_screened: How many stocks were evaluated
    - filters_applied: Description of filters used

    This is the FIRST tool the agent should call in any recommendation
    workflow.  It narrows the universe from thousands of stocks to a
    manageable 10-30 candidates that the other tools then evaluate deeply.
    """

    async def _fetch():
        excluded = set(t.upper() for t in (excluded_tickers or []))

        # ── MOCK UNIVERSE — replace with proper data source later ──
        universe = [
            {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "market_cap": 2.8e12},
            {"ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Technology", "market_cap": 3.1e12},
            {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "market_cap": 2.1e12},
            {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "market_cap": 1.9e12},
            {"ticker": "NVDA", "name": "NVIDIA Corp.", "sector": "Technology", "market_cap": 2.2e12},
            {"ticker": "META", "name": "Meta Platforms", "sector": "Technology", "market_cap": 1.3e12},
            {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Cyclical", "market_cap": 780e9},
            {"ticker": "JPM", "name": "JPMorgan Chase", "sector": "Financial Services", "market_cap": 570e9},
            {"ticker": "V", "name": "Visa Inc.", "sector": "Financial Services", "market_cap": 530e9},
            {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "market_cap": 375e9},
            {"ticker": "UNH", "name": "UnitedHealth", "sector": "Healthcare", "market_cap": 480e9},
            {"ticker": "HD", "name": "Home Depot", "sector": "Consumer Cyclical", "market_cap": 370e9},
            {"ticker": "PG", "name": "Procter & Gamble", "sector": "Consumer Defensive", "market_cap": 390e9},
            {"ticker": "MA", "name": "Mastercard", "sector": "Financial Services", "market_cap": 420e9},
            {"ticker": "XOM", "name": "Exxon Mobil", "sector": "Energy", "market_cap": 450e9},
            {"ticker": "LLY", "name": "Eli Lilly", "sector": "Healthcare", "market_cap": 720e9},
            {"ticker": "AVGO", "name": "Broadcom Inc.", "sector": "Technology", "market_cap": 620e9},
            {"ticker": "COST", "name": "Costco Wholesale", "sector": "Consumer Defensive", "market_cap": 350e9},
            {"ticker": "ABBV", "name": "AbbVie Inc.", "sector": "Healthcare", "market_cap": 310e9},
            {"ticker": "WMT", "name": "Walmart Inc.", "sector": "Consumer Defensive", "market_cap": 430e9},
            {"ticker": "CRM", "name": "Salesforce Inc.", "sector": "Technology", "market_cap": 265e9},
            {"ticker": "NFLX", "name": "Netflix Inc.", "sector": "Technology", "market_cap": 260e9},
            {"ticker": "AMD", "name": "AMD Inc.", "sector": "Technology", "market_cap": 230e9},
            {"ticker": "ORCL", "name": "Oracle Corp.", "sector": "Technology", "market_cap": 320e9},
            {"ticker": "MRK", "name": "Merck & Co.", "sector": "Healthcare", "market_cap": 280e9},
        ]

        # Apply filters
        filtered = [s for s in universe if s["ticker"] not in excluded]
        filtered = [s for s in filtered if s["market_cap"] >= min_market_cap]

        if sector_preferences:
            sector_prefs = set(sp.lower() for sp in sector_preferences)
            preferred = [s for s in filtered if s["sector"].lower() in sector_prefs]
            others = [s for s in filtered if s["sector"].lower() not in sector_prefs]
            filtered = preferred + others

        # Risk-based adjustments
        if risk_tolerance == "conservative":
            # Favor large-cap, defensive
            filtered.sort(key=lambda x: x["market_cap"], reverse=True)
        elif risk_tolerance == "aggressive":
            # Include more volatile names
            random.shuffle(filtered)

        candidates = filtered[:max_results]

        filters_desc = f"risk={risk_tolerance}, min_cap=${min_market_cap/1e9:.0f}B"
        if sector_preferences:
            filters_desc += f", sectors={sector_preferences}"
        if excluded_tickers:
            filters_desc += f", excluded={excluded_tickers}"

        return {
            "candidates": candidates,
            "total_screened": len(universe),
            "candidates_returned": len(candidates),
            "filters_applied": filters_desc,
        }

    cache_key = f"screen:{risk_tolerance}:{sorted(sector_preferences or [])}:{sorted(excluded_tickers or [])}"
    return await async_get_or_set(screener_cache, cache_key, _fetch)
