"""Market Data Tool — prices, OHLCV, company info.

Currently returns MOCK data.  Will be replaced with yfinance integration later.
"""

from __future__ import annotations

import datetime as dt
import random

from app.data.cache import market_data_cache, async_get_or_set


async def get_stock_price(ticker: str) -> dict:
    """Get current price and basic quote data for a ticker.

    Returns dict with keys: ticker, price, change_pct, volume,
    market_cap, fifty_two_week_high, fifty_two_week_low, currency.
    """

    async def _fetch():
        # ── MOCK DATA — replace with yfinance later ──
        base_prices = {
            "AAPL": 185.50, "MSFT": 420.10, "GOOGL": 175.30,
            "AMZN": 195.80, "NVDA": 890.50, "META": 510.20,
            "TSLA": 245.00, "JPM": 198.60, "V": 285.40,
            "JNJ": 155.30, "UNH": 520.80, "HD": 380.50,
            "PG": 165.20, "MA": 460.30, "XOM": 110.50,
        }
        price = base_prices.get(ticker.upper(), round(random.uniform(20, 500), 2))
        return {
            "ticker": ticker.upper(),
            "price": price,
            "change_pct": round(random.uniform(-3, 3), 2),
            "volume": random.randint(5_000_000, 80_000_000),
            "market_cap": round(price * random.uniform(1e9, 3e10), 0),
            "fifty_two_week_high": round(price * random.uniform(1.05, 1.30), 2),
            "fifty_two_week_low": round(price * random.uniform(0.60, 0.90), 2),
            "currency": "USD",
        }

    return await async_get_or_set(market_data_cache, f"price:{ticker.upper()}", _fetch)


async def get_historical_data(ticker: str, period: str = "6mo") -> dict:
    """Get historical OHLCV data for a ticker.

    Args:
        ticker: Stock ticker symbol.
        period: How far back — e.g. '1mo', '3mo', '6mo', '1y', '5y'.

    Returns dict with keys: ticker, period, data_points (int),
    sample_data (list of recent OHLCV bars).
    """

    async def _fetch():
        # ── MOCK DATA ──
        base = random.uniform(100, 500)
        bars = []
        for i in range(30):
            o = round(base + random.uniform(-5, 5), 2)
            h = round(o + random.uniform(0, 8), 2)
            l = round(o - random.uniform(0, 8), 2)  # noqa: E741
            c = round(random.uniform(l, h), 2)
            bars.append({
                "date": (dt.date.today() - dt.timedelta(days=30 - i)).isoformat(),
                "open": o, "high": h, "low": l, "close": c,
                "volume": random.randint(5_000_000, 80_000_000),
            })
            base = c  # next day opens near previous close
        return {
            "ticker": ticker.upper(),
            "period": period,
            "data_points": len(bars),
            "sample_data": bars,
        }

    return await async_get_or_set(market_data_cache, f"hist:{ticker.upper()}:{period}", _fetch)


async def get_company_info(ticker: str) -> dict:
    """Get basic company information.

    Returns dict with keys: ticker, name, sector, industry,
    description, employees, website.
    """

    async def _fetch():
        # ── MOCK DATA ──
        companies = {
            "AAPL": {"name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics"},
            "MSFT": {"name": "Microsoft Corp.", "sector": "Technology", "industry": "Software"},
            "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "industry": "Internet Services"},
            "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "industry": "E-Commerce"},
            "NVDA": {"name": "NVIDIA Corp.", "sector": "Technology", "industry": "Semiconductors"},
            "META": {"name": "Meta Platforms Inc.", "sector": "Technology", "industry": "Social Media"},
            "TSLA": {"name": "Tesla Inc.", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
            "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Financial Services", "industry": "Banks"},
            "V": {"name": "Visa Inc.", "sector": "Financial Services", "industry": "Credit Services"},
            "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare", "industry": "Drug Manufacturers"},
            "UNH": {"name": "UnitedHealth Group", "sector": "Healthcare", "industry": "Health Care Plans"},
            "HD": {"name": "Home Depot Inc.", "sector": "Consumer Cyclical", "industry": "Home Improvement"},
            "PG": {"name": "Procter & Gamble", "sector": "Consumer Defensive", "industry": "Household Products"},
            "MA": {"name": "Mastercard Inc.", "sector": "Financial Services", "industry": "Credit Services"},
            "XOM": {"name": "Exxon Mobil Corp.", "sector": "Energy", "industry": "Oil & Gas"},
        }
        info = companies.get(ticker.upper(), {
            "name": f"{ticker.upper()} Corp.",
            "sector": "Unknown",
            "industry": "Unknown",
        })
        return {
            "ticker": ticker.upper(),
            **info,
            "description": f"A leading company in the {info.get('industry', 'Unknown')} industry.",
            "employees": random.randint(5000, 200000),
            "website": f"https://www.{ticker.lower()}.com",
        }

    return await async_get_or_set(market_data_cache, f"info:{ticker.upper()}", _fetch)
