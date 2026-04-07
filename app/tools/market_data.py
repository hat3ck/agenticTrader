"""Market Data Tool — prices, OHLCV, company info.

Provider : yfinance (free, no API key)
Fallback : Alpha Vantage free tier (500 calls/day) when yfinance fails
Extension: Swap in Polygon.io / Alpaca for real-time streaming later —
           the tool interface stays the same, only the implementation changes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import yfinance as yf

from app.config import settings
from app.data.cache import market_data_cache, async_get_or_set

logger = logging.getLogger(__name__)

# Semaphore to limit concurrent yfinance calls — prevents crumb invalidation
# from too many simultaneous requests sharing the same session/cookie.
_YF_SEMAPHORE = asyncio.Semaphore(3)

_VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe(val: Any, fallback: Any = None) -> Any:
    """Return *val* unless it is None / NaN, in which case return *fallback*."""
    if val is None:
        return fallback
    try:
        # Catch float NaN
        if val != val:  # noqa: PLR0124  (NaN is the only value ≠ itself)
            return fallback
    except (TypeError, ValueError):
        pass
    return val


def _yf_ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker.upper())


def _reset_yf_crumb() -> None:
    """Clear the cached crumb/cookie so yfinance re-authenticates on the next call."""
    try:
        from yfinance.data import YfData
        yd = YfData()
        yd._crumb = None
        yd._cookie = None
        logger.debug("Cleared yfinance crumb/cookie cache")
    except Exception:  # noqa: BLE001
        pass


async def _yf_info_with_retry(ticker: str, max_retries: int = 2) -> dict:
    """Fetch ``yf.Ticker(ticker).info`` with retry on crumb/auth failures.

    yfinance shares a global crumb/cookie that can be invalidated by
    concurrent requests.  On 401 / auth errors we reset the crumb and retry.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with _YF_SEMAPHORE:
                t = _yf_ticker(ticker)
                info = await asyncio.to_thread(lambda: t.info)
            if info is None:
                info = {}
            return info
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            err_str = str(exc).lower()
            if "unauthorized" in err_str or "invalid crumb" in err_str or "401" in err_str:
                logger.debug(
                    "yfinance auth error for %s (attempt %d/%d): %s — resetting crumb",
                    ticker, attempt + 1, max_retries + 1, exc,
                )
                await asyncio.to_thread(_reset_yf_crumb)
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            raise
    raise last_exc


# ── Alpha Vantage fallback helpers ───────────────────────────────────────────

async def _av_quote(ticker: str) -> dict | None:
    """Alpha Vantage GLOBAL_QUOTE fallback. Returns None when unavailable."""
    api_key = settings.alpha_vantage_api_key
    if not api_key:
        return None
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={ticker.upper()}&apikey={api_key}"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json().get("Global Quote", {})
            if not data:
                return None
            price = float(data.get("05. price", 0))
            prev_close = float(data.get("08. previous close", 0))
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
            return {
                "ticker": ticker.upper(),
                "price": round(price, 2),
                "change_pct": round(change_pct, 2),
                "volume": int(data.get("06. volume", 0)),
                "market_cap": None,
                "fifty_two_week_high": None,
                "fifty_two_week_low": None,
                "currency": "USD",
            }
    except Exception as exc:
        logger.warning("Alpha Vantage quote failed for %s: %s", ticker, exc)
        return None


async def _av_history(ticker: str, period: str) -> dict | None:
    """Alpha Vantage TIME_SERIES_DAILY fallback. Returns None when unavailable."""
    api_key = settings.alpha_vantage_api_key
    if not api_key:
        return None
    size = "full" if period in {"1y", "2y", "5y", "10y", "max"} else "compact"
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={ticker.upper()}"
        f"&outputsize={size}&apikey={api_key}"
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            ts = resp.json().get("Time Series (Daily)", {})
            if not ts:
                return None
            bars = [
                {
                    "date": date,
                    "open": float(vals["1. open"]),
                    "high": float(vals["2. high"]),
                    "low": float(vals["3. low"]),
                    "close": float(vals["4. close"]),
                    "volume": int(vals["5. volume"]),
                }
                for date, vals in sorted(ts.items())
            ]
            return {
                "ticker": ticker.upper(),
                "period": period,
                "data_points": len(bars),
                "sample_data": bars,
            }
    except Exception as exc:
        logger.warning("Alpha Vantage history failed for %s: %s", ticker, exc)
        return None


# ── public API ───────────────────────────────────────────────────────────────

async def get_stock_price(ticker: str) -> dict:
    """Get current price and basic quote data for a ticker.

    Returns dict with keys: ticker, price, change_pct, volume,
    market_cap, fifty_two_week_high, fifty_two_week_low, currency.
    """

    async def _fetch() -> dict:
        try:
            async with _YF_SEMAPHORE:
                t = _yf_ticker(ticker)
                info: dict = await asyncio.to_thread(lambda: t.fast_info)
            # fast_info is a dict-like object; fall back to .info for extras
            price = _safe(getattr(info, "last_price", None))
            prev_close = _safe(getattr(info, "previous_close", None))
            change_pct = (
                round((price - prev_close) / prev_close * 100, 2)
                if price is not None and prev_close
                else None
            )
            return {
                "ticker": ticker.upper(),
                "price": round(price, 2) if price is not None else None,
                "change_pct": change_pct,
                "volume": _safe(getattr(info, "last_volume", None)),
                "market_cap": _safe(getattr(info, "market_cap", None)),
                "fifty_two_week_high": _safe(getattr(info, "year_high", None)),
                "fifty_two_week_low": _safe(getattr(info, "year_low", None)),
                "currency": _safe(getattr(info, "currency", None), "USD"),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance price failed for %s: %s — trying Alpha Vantage", ticker, exc)
            av = await _av_quote(ticker)
            if av is not None:
                return av
            raise RuntimeError(f"Could not fetch price for {ticker}: {exc}") from exc

    return await async_get_or_set(market_data_cache, f"price:{ticker.upper()}", _fetch)


async def get_historical_data(ticker: str, period: str = "6mo") -> dict:
    """Get historical OHLCV data for a ticker.

    Args:
        ticker: Stock ticker symbol.
        period: How far back — e.g. '1mo', '3mo', '6mo', '1y', '5y'.

    Returns dict with keys: ticker, period, data_points (int),
    sample_data (list of recent OHLCV bars).
    """
    if period not in _VALID_PERIODS:
        period = "6mo"

    async def _fetch() -> dict:
        try:
            async with _YF_SEMAPHORE:
                t = _yf_ticker(ticker)
                df = await asyncio.to_thread(lambda: t.history(period=period, auto_adjust=True))
            if df.empty:
                raise ValueError(f"No historical data returned for {ticker}")

            bars = [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                }
                for idx, row in df.iterrows()
            ]
            return {
                "ticker": ticker.upper(),
                "period": period,
                "data_points": len(bars),
                "sample_data": bars,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance history failed for %s: %s — trying Alpha Vantage", ticker, exc)
            av = await _av_history(ticker, period)
            if av is not None:
                return av
            raise RuntimeError(f"Could not fetch history for {ticker}: {exc}") from exc

    return await async_get_or_set(market_data_cache, f"hist:{ticker.upper()}:{period}", _fetch)


async def get_company_info(ticker: str) -> dict:
    """Get basic company information.

    Returns dict with keys: ticker, name, sector, industry,
    description, employees, website.
    """

    async def _fetch() -> dict:
        try:
            info = await _yf_info_with_retry(ticker)
            return {
                "ticker": ticker.upper(),
                "name": _safe(info.get("longName") or info.get("shortName"), f"{ticker.upper()} Corp."),
                "sector": _safe(info.get("sector"), "Unknown"),
                "industry": _safe(info.get("industry"), "Unknown"),
                "description": _safe(info.get("longBusinessSummary"), ""),
                "employees": _safe(info.get("fullTimeEmployees")),
                "website": _safe(info.get("website"), ""),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance company info failed for %s: %s", ticker, exc)
            # Return default data instead of crashing — allows the pipeline to continue
            return {
                "ticker": ticker.upper(),
                "name": f"{ticker.upper()} Corp.",
                "sector": "Unknown",
                "industry": "Unknown",
                "description": "",
                "employees": None,
                "website": "",
            }

    return await async_get_or_set(market_data_cache, f"info:{ticker.upper()}", _fetch)
