"""Stock universe provider — dynamic with hardcoded fallback.

Fallback chain (first success wins):
  1. FMP API  — S&P 500 + NASDAQ constituents (requires FMP_API_KEY)
  2. Wikipedia — scrapes current S&P 500 & NASDAQ-100 tables (no key)
  3. Hardcoded — ~260 stocks across all GICS sectors

All dynamic data is cached for 24 h so the universe automatically picks
up index rebalances without manual edits.

Sector names follow Yahoo Finance conventions so they line up with the
rest of the application.
"""

from __future__ import annotations

import asyncio
import logging
from io import StringIO
from typing import TypedDict

import httpx
import pandas as pd

from app.config import settings
from app.data.cache import constituents_cache, async_get_or_set

logger = logging.getLogger(__name__)


class StockEntry(TypedDict):
    ticker: str
    name: str
    sector: str
    exchange: str          # "NYSE" | "NASDAQ"
    cap_tier: str          # "mega" | "large" | "mid" | "small"
    indices: list[str]     # e.g. ["sp500", "nasdaq100"]


# ── FMP API ──────────────────────────────────────────────────────────────────

_FMP_BASE = "https://financialmodelingprep.com/api/v3"

# Map FMP exchange strings → our canonical names
_EXCHANGE_MAP: dict[str, str] = {
    "NASDAQ": "NASDAQ",
    "NYSE": "NYSE",
    "New York Stock Exchange": "NYSE",
    "NasdaqGS": "NASDAQ",
    "NasdaqGM": "NASDAQ",
    "NasdaqCM": "NASDAQ",
    "AMEX": "NYSE",           # rolled into NYSE
    "BATS": "NYSE",           # CBOE/BATS-listed ≈ NYSE for our purposes
}


def _classify_cap_tier(market_cap: float | None) -> str:
    """Classify market cap into mega / large / mid / small."""
    if market_cap is None:
        return "large"  # safe default
    if market_cap >= 200e9:
        return "mega"
    if market_cap >= 10e9:
        return "large"
    if market_cap >= 2e9:
        return "mid"
    return "small"


async def _fmp_get(endpoint: str) -> list[dict] | None:
    """Call an FMP API endpoint.  Returns None on any failure."""
    api_key = settings.fmp_api_key
    if not api_key:
        return None
    url = f"{_FMP_BASE}/{endpoint}?apikey={api_key}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            return None
    except httpx.HTTPStatusError as exc:
        logger.warning("FMP API HTTP error for %s: %s", endpoint, exc.response.status_code)
        return None
    except Exception:
        logger.exception("FMP API request failed for %s", endpoint)
        return None


def _normalize_fmp_constituent(raw: dict, index_tag: str) -> StockEntry | None:
    """Convert a raw FMP constituent dict into our StockEntry format."""
    ticker = raw.get("symbol")
    if not ticker:
        return None
    name = raw.get("name", "")
    sector = raw.get("sector") or raw.get("sector", "Unknown")
    exchange = _EXCHANGE_MAP.get(raw.get("exchange", ""), "NYSE")

    # FMP doesn't return market cap in constituent endpoints, so use a
    # placeholder.  The screener enriches with live data via yfinance later.
    cap_tier = "large"  # default; screener will re-classify from live data

    return {
        "ticker": ticker.upper(),
        "name": name,
        "sector": sector,
        "exchange": exchange,
        "cap_tier": cap_tier,
        "indices": [index_tag],
    }


async def _fetch_fmp_universe() -> list[StockEntry] | None:
    """Fetch S&P 500 + NASDAQ + small/mid-cap stocks from FMP and merge them.

    Returns None if the API key is missing or calls fail.
    """
    if not settings.fmp_api_key:
        return None

    sp500_raw = await _fmp_get("sp500_constituent")
    nasdaq_raw = await _fmp_get("nasdaq_constituent")

    if not sp500_raw and not nasdaq_raw:
        logger.warning("FMP: both constituent endpoints returned no data, using fallback")
        return None

    # Build a ticker → StockEntry map, merging index memberships
    stock_map: dict[str, StockEntry] = {}

    for raw in (sp500_raw or []):
        entry = _normalize_fmp_constituent(raw, "sp500")
        if entry:
            stock_map[entry["ticker"]] = entry

    for raw in (nasdaq_raw or []):
        entry = _normalize_fmp_constituent(raw, "nasdaq100")
        if entry:
            if entry["ticker"] in stock_map:
                # Merge index membership
                existing = stock_map[entry["ticker"]]
                if "nasdaq100" not in existing["indices"]:
                    existing["indices"].append("nasdaq100")
            else:
                stock_map[entry["ticker"]] = entry

    # Fetch small/mid-cap stocks via stock screener (market cap $300M–$10B)
    smid_stocks = await _fetch_fmp_smallmid_screener()
    for entry in (smid_stocks or []):
        if entry["ticker"] not in stock_map:
            stock_map[entry["ticker"]] = entry

    result = list(stock_map.values())
    if len(result) < 50:
        logger.warning("FMP returned only %d stocks, seems too few — using fallback", len(result))
        return None

    logger.info("FMP: loaded %d stocks dynamically (S&P 500 + NASDAQ + small/mid-cap)", len(result))
    return result


async def _fetch_fmp_smallmid_screener() -> list[StockEntry] | None:
    """Fetch small/mid-cap US stocks using FMP's stock screener endpoint.

    Targets market cap range $300M–$10B on NYSE/NASDAQ with reasonable
    volume to avoid illiquid penny stocks.
    """
    api_key = settings.fmp_api_key
    if not api_key:
        return None

    url = (
        f"{_FMP_BASE}/stock-screener"
        f"?marketCapMoreThan=300000000"
        f"&marketCapLowerThan=10000000000"
        f"&volumeMoreThan=200000"
        f"&exchange=NYSE,NASDAQ"
        f"&isActivelyTrading=true"
        f"&limit=500"
        f"&apikey={api_key}"
    )
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                return None

        result: list[StockEntry] = []
        for raw in data:
            ticker = raw.get("symbol")
            if not ticker or "." in ticker:  # skip class shares like BRK.B
                continue
            name = raw.get("companyName", "")
            sector = raw.get("sector") or "Unknown"
            exchange = _EXCHANGE_MAP.get(raw.get("exchangeShortName", ""), "NYSE")
            market_cap = raw.get("marketCap")
            cap_tier = _classify_cap_tier(market_cap)

            # Only keep mid and small cap stocks (large/mega already in S&P 500)
            if cap_tier not in ("mid", "small"):
                continue

            index_tag = "sp400" if cap_tier == "mid" else "sp600"
            result.append({
                "ticker": ticker.upper(),
                "name": name,
                "sector": sector,
                "exchange": exchange,
                "cap_tier": cap_tier,
                "indices": [index_tag],
            })

        logger.info("FMP screener: loaded %d small/mid-cap stocks", len(result))
        return result if result else None
    except Exception:
        logger.exception("FMP stock screener request failed")
        return None


# ── Wikipedia scraping ────────────────────────────────────────────────────────

_WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_WIKI_NASDAQ_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

# Wikipedia requires a real User-Agent or it returns 403
_WIKI_HEADERS = {
    "User-Agent": "AgenticTrader/0.1 (https://github.com/agenticTrader; educational project)",
    "Accept": "text/html,application/json",
}

# Use Wikipedia's REST API (less likely to be rate-limited than direct page scrapes)
_WIKI_API = "https://en.wikipedia.org/w/api.php"

# Wikipedia GICS sectors → Yahoo Finance sector names
_GICS_TO_YF: dict[str, str] = {
    "Information Technology": "Technology",
    "Communication Services": "Communication Services",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Health Care": "Healthcare",
    "Financials": "Financial Services",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Materials": "Basic Materials",
}


def _normalize_wiki_sector(gics_sector: str) -> str:
    """Map a GICS sector label to the Yahoo Finance equivalent."""
    return _GICS_TO_YF.get(gics_sector.strip(), gics_sector.strip())


async def _fetch_wikipedia_sp500() -> list[dict] | None:
    """Scrape the S&P 500 constituent table from Wikipedia via the parse API."""
    try:
        params = {
            "action": "parse",
            "page": "List_of_S&P_500_companies",
            "prop": "text",
            "section": "1",  # section 1 has the main constituents table
            "format": "json",
            "formatversion": "2",
        }
        async with httpx.AsyncClient(timeout=20, headers=_WIKI_HEADERS) as client:
            resp = await client.get(_WIKI_API, params=params)
            resp.raise_for_status()
            data = resp.json()

        html = data.get("parse", {}).get("text", "")
        if not html:
            # Fallback: try section 0
            params["section"] = "0"
            async with httpx.AsyncClient(timeout=20, headers=_WIKI_HEADERS) as client:
                resp = await client.get(_WIKI_API, params=params)
                resp.raise_for_status()
                data = resp.json()
            html = data.get("parse", {}).get("text", "")

        if not html:
            logger.warning("Wikipedia S&P 500 API returned no HTML")
            return None

        # pandas.read_html is sync — run in a thread to avoid blocking
        tables = await asyncio.to_thread(pd.read_html, StringIO(html))
        if not tables:
            return None

        # Pick the largest table (the constituents list)
        df = max(tables, key=len)

        # Expected columns: Symbol, Security, GICS Sector, ...
        sym_col = next((c for c in df.columns if "symbol" in str(c).lower() or "ticker" in str(c).lower()), None)
        name_col = next((c for c in df.columns if "security" in str(c).lower() or "company" in str(c).lower()), None)
        sector_col = next((c for c in df.columns if "sector" in str(c).lower()), None)

        if sym_col is None:
            logger.warning("Wikipedia S&P 500: could not find symbol column in %s", list(df.columns))
            return None

        result = []
        for _, row in df.iterrows():
            ticker = str(row[sym_col]).strip().replace(".", "-")  # BRK.B → BRK-B
            if not ticker or ticker == "nan":
                continue
            name = str(row[name_col]).strip() if name_col else ""
            sector = _normalize_wiki_sector(str(row[sector_col])) if sector_col else "Unknown"
            result.append({
                "ticker": ticker.upper(),
                "name": name,
                "sector": sector,
                "index": "sp500",
            })
        return result if result else None
    except Exception:
        logger.exception("Wikipedia S&P 500 scrape failed")
        return None


async def _fetch_wikipedia_nasdaq100() -> list[dict] | None:
    """Scrape the NASDAQ-100 constituent table from Wikipedia via the parse API."""
    try:
        # First, find the "Current components" section dynamically
        section_params = {
            "action": "parse",
            "page": "Nasdaq-100",
            "prop": "sections",
            "format": "json",
            "formatversion": "2",
        }
        async with httpx.AsyncClient(timeout=20, headers=_WIKI_HEADERS) as client:
            sec_resp = await client.get(_WIKI_API, params=section_params)
            sec_resp.raise_for_status()
            sec_data = sec_resp.json()

        sections = sec_data.get("parse", {}).get("sections", [])
        comp_section = None
        for s in sections:
            if "component" in s.get("line", "").lower():
                comp_section = s["index"]
                break

        if comp_section is None:
            logger.warning("Wikipedia NASDAQ-100: could not find 'components' section")
            comp_section = "12"  # fallback to known section

        params = {
            "action": "parse",
            "page": "Nasdaq-100",
            "prop": "text",
            "section": comp_section,
            "format": "json",
            "formatversion": "2",
        }
        async with httpx.AsyncClient(timeout=20, headers=_WIKI_HEADERS) as client:
            resp = await client.get(_WIKI_API, params=params)
            resp.raise_for_status()
            data = resp.json()

        html = data.get("parse", {}).get("text", "")

        if not html:
            logger.warning("Wikipedia NASDAQ-100 API returned no HTML")
            return None

        tables = await asyncio.to_thread(pd.read_html, StringIO(html))
        if not tables:
            return None

        # Find the table with a "Ticker" or "Symbol" column and >80 rows
        df = None
        ticker_col_name = None
        for t in tables:
            col = next((c for c in t.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()), None)
            if col and len(t) > 80:
                df = t
                ticker_col_name = col
                break

        # If no table with >80 rows, pick the largest with a ticker column
        if df is None:
            for t in sorted(tables, key=len, reverse=True):
                col = next((c for c in t.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()), None)
                if col and len(t) > 10:
                    df = t
                    ticker_col_name = col
                    break

        if df is None or ticker_col_name is None:
            logger.warning("Wikipedia NASDAQ-100: could not find constituent table")
            return None

        name_col = next((c for c in df.columns if "company" in str(c).lower() or "security" in str(c).lower()), None)
        # NASDAQ-100 page uses "ICB Industry" instead of "GICS Sector"
        sector_col = next((c for c in df.columns if "sector" in str(c).lower() or "industry" in str(c).lower()), None)

        result = []
        for _, row in df.iterrows():
            ticker = str(row[ticker_col_name]).strip().replace(".", "-")
            if not ticker or ticker == "nan":
                continue
            name = str(row[name_col]).strip() if name_col else ""
            sector = _normalize_wiki_sector(str(row[sector_col])) if sector_col else "Unknown"
            result.append({
                "ticker": ticker.upper(),
                "name": name,
                "sector": sector,
                "index": "nasdaq100",
            })
        return result if result else None
    except Exception:
        logger.exception("Wikipedia NASDAQ-100 scrape failed")
        return None


async def _fetch_wikipedia_sp400() -> list[dict] | None:
    """Scrape the S&P MidCap 400 constituent table from Wikipedia."""
    try:
        params = {
            "action": "parse",
            "page": "List_of_S&P_400_companies",
            "prop": "text",
            "section": "1",
            "format": "json",
            "formatversion": "2",
        }
        async with httpx.AsyncClient(timeout=20, headers=_WIKI_HEADERS) as client:
            resp = await client.get(_WIKI_API, params=params)
            resp.raise_for_status()
            data = resp.json()

        html = data.get("parse", {}).get("text", "")
        if not html:
            logger.warning("Wikipedia S&P 400 API returned no HTML")
            return None

        tables = await asyncio.to_thread(pd.read_html, StringIO(html))
        if not tables:
            return None

        df = max(tables, key=len)

        sym_col = next((c for c in df.columns if "symbol" in str(c).lower() or "ticker" in str(c).lower()), None)
        name_col = next((c for c in df.columns if "company" in str(c).lower() or "security" in str(c).lower()), None)
        sector_col = next((c for c in df.columns if "sector" in str(c).lower()), None)

        if sym_col is None:
            logger.warning("Wikipedia S&P 400: could not find symbol column in %s", list(df.columns))
            return None

        result = []
        for _, row in df.iterrows():
            ticker = str(row[sym_col]).strip().replace(".", "-")
            if not ticker or ticker == "nan":
                continue
            name = str(row[name_col]).strip() if name_col else ""
            sector = _normalize_wiki_sector(str(row[sector_col])) if sector_col else "Unknown"
            result.append({
                "ticker": ticker.upper(),
                "name": name,
                "sector": sector,
                "index": "sp400",
            })
        return result if result else None
    except Exception:
        logger.exception("Wikipedia S&P 400 scrape failed")
        return None


async def _fetch_wikipedia_sp600() -> list[dict] | None:
    """Scrape the S&P SmallCap 600 constituent table from Wikipedia."""
    try:
        params = {
            "action": "parse",
            "page": "List_of_S&P_600_companies",
            "prop": "text",
            "section": "1",
            "format": "json",
            "formatversion": "2",
        }
        async with httpx.AsyncClient(timeout=20, headers=_WIKI_HEADERS) as client:
            resp = await client.get(_WIKI_API, params=params)
            resp.raise_for_status()
            data = resp.json()

        html = data.get("parse", {}).get("text", "")
        if not html:
            logger.warning("Wikipedia S&P 600 API returned no HTML")
            return None

        tables = await asyncio.to_thread(pd.read_html, StringIO(html))
        if not tables:
            return None

        df = max(tables, key=len)

        sym_col = next((c for c in df.columns if "symbol" in str(c).lower() or "ticker" in str(c).lower()), None)
        name_col = next((c for c in df.columns if "company" in str(c).lower() or "security" in str(c).lower()), None)
        sector_col = next((c for c in df.columns if "sector" in str(c).lower()), None)

        if sym_col is None:
            logger.warning("Wikipedia S&P 600: could not find symbol column in %s", list(df.columns))
            return None

        result = []
        for _, row in df.iterrows():
            ticker = str(row[sym_col]).strip().replace(".", "-")
            if not ticker or ticker == "nan":
                continue
            name = str(row[name_col]).strip() if name_col else ""
            sector = _normalize_wiki_sector(str(row[sector_col])) if sector_col else "Unknown"
            result.append({
                "ticker": ticker.upper(),
                "name": name,
                "sector": sector,
                "index": "sp600",
            })
        return result if result else None
    except Exception:
        logger.exception("Wikipedia S&P 600 scrape failed")
        return None


async def _fetch_wikipedia_universe() -> list[StockEntry] | None:
    """Fetch S&P 500 + NASDAQ-100 + S&P 400 + S&P 600 from Wikipedia and merge.

    Returns None if scraping fails for all indices.
    """
    sp500_raw, nasdaq_raw, sp400_raw, sp600_raw = await asyncio.gather(
        _fetch_wikipedia_sp500(),
        _fetch_wikipedia_nasdaq100(),
        _fetch_wikipedia_sp400(),
        _fetch_wikipedia_sp600(),
    )

    if not sp500_raw and not nasdaq_raw and not sp400_raw and not sp600_raw:
        logger.warning("Wikipedia: all scrapes returned no data")
        return None

    stock_map: dict[str, StockEntry] = {}

    for raw in (sp500_raw or []):
        ticker = raw["ticker"]
        stock_map[ticker] = {
            "ticker": ticker,
            "name": raw.get("name", ""),
            "sector": raw.get("sector", "Unknown"),
            "exchange": "NASDAQ" if ticker in _KNOWN_NASDAQ else "NYSE",
            "cap_tier": "large",  # placeholder; screener enriches later
            "indices": ["sp500"],
        }

    for raw in (nasdaq_raw or []):
        ticker = raw["ticker"]
        if ticker in stock_map:
            if "nasdaq100" not in stock_map[ticker]["indices"]:
                stock_map[ticker]["indices"].append("nasdaq100")
            stock_map[ticker]["exchange"] = "NASDAQ"  # must be NASDAQ-listed
        else:
            stock_map[ticker] = {
                "ticker": ticker,
                "name": raw.get("name", ""),
                "sector": raw.get("sector", "Unknown"),
                "exchange": "NASDAQ",
                "cap_tier": "large",
                "indices": ["nasdaq100"],
            }

    for raw in (sp400_raw or []):
        ticker = raw["ticker"]
        if ticker in stock_map:
            if "sp400" not in stock_map[ticker]["indices"]:
                stock_map[ticker]["indices"].append("sp400")
        else:
            stock_map[ticker] = {
                "ticker": ticker,
                "name": raw.get("name", ""),
                "sector": raw.get("sector", "Unknown"),
                "exchange": "NASDAQ" if ticker in _KNOWN_NASDAQ else "NYSE",
                "cap_tier": "mid",  # S&P 400 = mid-cap
                "indices": ["sp400"],
            }

    for raw in (sp600_raw or []):
        ticker = raw["ticker"]
        if ticker in stock_map:
            if "sp600" not in stock_map[ticker]["indices"]:
                stock_map[ticker]["indices"].append("sp600")
        else:
            stock_map[ticker] = {
                "ticker": ticker,
                "name": raw.get("name", ""),
                "sector": raw.get("sector", "Unknown"),
                "exchange": "NASDAQ" if ticker in _KNOWN_NASDAQ else "NYSE",
                "cap_tier": "small",  # S&P 600 = small-cap
                "indices": ["sp600"],
            }

    result = list(stock_map.values())
    if len(result) < 50:
        logger.warning("Wikipedia returned only %d stocks — using fallback", len(result))
        return None

    logger.info("Wikipedia: loaded %d stocks dynamically (S&P 500 + NASDAQ-100 + S&P 400 + S&P 600)", len(result))
    return result


# Quick lookup: tickers that are NASDAQ-listed (used when Wikipedia S&P 500
# table doesn't provide exchange info).  Not exhaustive — screener re-checks.
_KNOWN_NASDAQ: set[str] = {
    "AAPL", "MSFT", "NVDA", "AVGO", "META", "GOOGL", "GOOG", "AMZN", "TSLA",
    "COST", "NFLX", "AMD", "ADBE", "CSCO", "INTC", "QCOM", "TXN", "AMAT",
    "MU", "LRCX", "KLAC", "SNPS", "CDNS", "MRVL", "PANW", "CRWD", "FTNT",
    "WDAY", "TEAM", "DDOG", "ZS", "SMCI", "MPWR", "ANSS", "LSCC", "RMBS",
    "PEP", "MDLZ", "KHC", "CMCSA", "TMUS", "CHTR", "EA", "TTWO", "WBD",
    "ROKU", "SBUX", "ABNB", "BKNG", "ROST", "ORLY", "EBAY", "LULU", "ETSY",
    "CROX", "FIVE", "AMGN", "GILD", "ISRG", "VRTX", "REGN", "DXCM", "IDXX",
    "ALGN", "FANG", "HON", "FAST", "CTAS", "CSX", "AXON", "CGNX", "NDSN",
    "EQIX", "CME", "PYPL", "AEP", "EXC", "XEL", "CEG", "LIN",
}


async def get_dynamic_universe(data_source: str = "auto") -> list[StockEntry]:
    """Return the stock universe using the specified data source.

    Args:
        data_source: Which source to fetch from.
            ``"auto"``      – Fallback chain: FMP → Wikipedia → hardcoded (default).
            ``"fmp"``       – FMP API only; falls back to hardcoded on failure.
            ``"wikipedia"`` – Wikipedia only; falls back to hardcoded on failure.
            ``"hardcoded"`` – Built-in static universe (no network calls).
    """
    # Use a source-specific cache key so different sources don't collide
    cache_key = f"universe:{data_source}"

    async def _fetch():
        if data_source == "fmp":
            fmp = await _fetch_fmp_universe()
            if fmp is not None:
                return {"source": "fmp", "stocks": fmp}
            logger.warning("FMP source requested but failed; falling back to hardcoded")
            return {"source": "hardcoded", "stocks": list(_UNIVERSE)}

        if data_source == "wikipedia":
            wiki = await _fetch_wikipedia_universe()
            if wiki is not None:
                return {"source": "wikipedia", "stocks": wiki}
            logger.warning("Wikipedia source requested but failed; falling back to hardcoded")
            return {"source": "hardcoded", "stocks": list(_UNIVERSE)}

        if data_source == "hardcoded":
            logger.info("Using hardcoded stock universe (%d stocks)", len(_UNIVERSE))
            return {"source": "hardcoded", "stocks": list(_UNIVERSE)}

        # Default: "auto" — original fallback chain
        fmp = await _fetch_fmp_universe()
        if fmp is not None:
            return {"source": "fmp", "stocks": fmp}

        wiki = await _fetch_wikipedia_universe()
        if wiki is not None:
            return {"source": "wikipedia", "stocks": wiki}

        logger.info("Using hardcoded stock universe (%d stocks)", len(_UNIVERSE))
        return {"source": "hardcoded", "stocks": list(_UNIVERSE)}

    cached = await async_get_or_set(constituents_cache, cache_key, _fetch)
    return cached["stocks"]


def get_universe_source_sync(data_source: str = "auto") -> str:
    """Check which source the cached universe came from (for diagnostics)."""
    cache_key = f"universe:{data_source}"
    cached = constituents_cache.get(cache_key)
    if cached:
        return cached["source"]
    return "not_loaded"


# ── Hardcoded Fallback Universe ──────────────────────────────────────────────
# ~260 stocks covering all GICS sectors.  Used when FMP_API_KEY is not set
# or the FMP API is unreachable.

_UNIVERSE: list[StockEntry] = [
    # ── Technology ───────────────────────────────────────────────────────────
    {"ticker": "AAPL",  "name": "Apple Inc.",                 "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "mega",  "indices": ["sp500", "nasdaq100"]},
    {"ticker": "MSFT",  "name": "Microsoft Corp.",            "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "mega",  "indices": ["sp500", "nasdaq100"]},
    {"ticker": "NVDA",  "name": "NVIDIA Corp.",               "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "mega",  "indices": ["sp500", "nasdaq100"]},
    {"ticker": "AVGO",  "name": "Broadcom Inc.",              "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "mega",  "indices": ["sp500", "nasdaq100"]},
    {"ticker": "ORCL",  "name": "Oracle Corp.",               "sector": "Technology",  "exchange": "NYSE",   "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "CRM",   "name": "Salesforce Inc.",            "sector": "Technology",  "exchange": "NYSE",   "cap_tier": "mega",  "indices": ["sp500", "nasdaq100"]},
    {"ticker": "AMD",   "name": "Advanced Micro Devices",     "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "ADBE",  "name": "Adobe Inc.",                 "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "CSCO",  "name": "Cisco Systems",              "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "INTC",  "name": "Intel Corp.",                "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "QCOM",  "name": "Qualcomm Inc.",              "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "TXN",   "name": "Texas Instruments",          "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "NOW",   "name": "ServiceNow Inc.",            "sector": "Technology",  "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "AMAT",  "name": "Applied Materials",          "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "MU",    "name": "Micron Technology",          "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "LRCX",  "name": "Lam Research",               "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "KLAC",  "name": "KLA Corp.",                  "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "SNPS",  "name": "Synopsys Inc.",              "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "CDNS",  "name": "Cadence Design Systems",    "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "MRVL",  "name": "Marvell Technology",         "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "PANW",  "name": "Palo Alto Networks",         "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "CRWD",  "name": "CrowdStrike Holdings",      "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "FTNT",  "name": "Fortinet Inc.",              "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "PLTR",  "name": "Palantir Technologies",     "sector": "Technology",  "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "WDAY",  "name": "Workday Inc.",               "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "TEAM",  "name": "Atlassian Corp.",            "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "DDOG",  "name": "Datadog Inc.",               "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "NET",   "name": "Cloudflare Inc.",            "sector": "Technology",  "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "HUBS",  "name": "HubSpot Inc.",               "sector": "Technology",  "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "ZS",    "name": "Zscaler Inc.",               "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "SNOW",  "name": "Snowflake Inc.",             "sector": "Technology",  "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "SMCI",  "name": "Super Micro Computer",      "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "MPWR",  "name": "Monolithic Power Systems",  "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "ANSS",  "name": "ANSYS Inc.",                 "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "KEYS",  "name": "Keysight Technologies",     "sector": "Technology",  "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "LSCC",  "name": "Lattice Semiconductor",     "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["russell2000"]},
    {"ticker": "RMBS",  "name": "Rambus Inc.",                "sector": "Technology",  "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["russell2000"]},
    {"ticker": "ONTO",  "name": "Onto Innovation",            "sector": "Technology",  "exchange": "NYSE",   "cap_tier": "mid",   "indices": ["russell2000"]},

    # ── Communication Services ───────────────────────────────────────────────
    {"ticker": "GOOGL", "name": "Alphabet Inc.",              "sector": "Communication Services", "exchange": "NASDAQ", "cap_tier": "mega",  "indices": ["sp500", "nasdaq100"]},
    {"ticker": "META",  "name": "Meta Platforms",             "sector": "Communication Services", "exchange": "NASDAQ", "cap_tier": "mega",  "indices": ["sp500", "nasdaq100"]},
    {"ticker": "NFLX",  "name": "Netflix Inc.",               "sector": "Communication Services", "exchange": "NASDAQ", "cap_tier": "mega",  "indices": ["sp500", "nasdaq100"]},
    {"ticker": "DIS",   "name": "Walt Disney Co.",            "sector": "Communication Services", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "CMCSA", "name": "Comcast Corp.",              "sector": "Communication Services", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "TMUS",  "name": "T-Mobile US",                "sector": "Communication Services", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "VZ",    "name": "Verizon Communications",    "sector": "Communication Services", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "T",     "name": "AT&T Inc.",                  "sector": "Communication Services", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "CHTR",  "name": "Charter Communications",    "sector": "Communication Services", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "EA",    "name": "Electronic Arts",            "sector": "Communication Services", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "TTWO",  "name": "Take-Two Interactive",       "sector": "Communication Services", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "WBD",   "name": "Warner Bros. Discovery",    "sector": "Communication Services", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "ROKU",  "name": "Roku Inc.",                  "sector": "Communication Services", "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["russell2000"]},

    # ── Consumer Cyclical ────────────────────────────────────────────────────
    {"ticker": "AMZN",  "name": "Amazon.com Inc.",            "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "mega",  "indices": ["sp500", "nasdaq100"]},
    {"ticker": "TSLA",  "name": "Tesla Inc.",                 "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "mega",  "indices": ["sp500", "nasdaq100"]},
    {"ticker": "HD",    "name": "Home Depot Inc.",            "sector": "Consumer Cyclical", "exchange": "NYSE",   "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "MCD",   "name": "McDonald's Corp.",           "sector": "Consumer Cyclical", "exchange": "NYSE",   "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "LOW",   "name": "Lowe's Cos.",                "sector": "Consumer Cyclical", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "TJX",   "name": "TJX Companies",              "sector": "Consumer Cyclical", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "BKNG",  "name": "Booking Holdings",           "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "NKE",   "name": "Nike Inc.",                  "sector": "Consumer Cyclical", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "SBUX",  "name": "Starbucks Corp.",            "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "ABNB",  "name": "Airbnb Inc.",                "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "CMG",   "name": "Chipotle Mexican Grill",    "sector": "Consumer Cyclical", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "ROST",  "name": "Ross Stores",                "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "ORLY",  "name": "O'Reilly Automotive",       "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "AZO",   "name": "AutoZone Inc.",              "sector": "Consumer Cyclical", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "DHI",   "name": "D.R. Horton Inc.",           "sector": "Consumer Cyclical", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "LEN",   "name": "Lennar Corp.",               "sector": "Consumer Cyclical", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "GM",    "name": "General Motors",             "sector": "Consumer Cyclical", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "F",     "name": "Ford Motor Co.",             "sector": "Consumer Cyclical", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "EBAY",  "name": "eBay Inc.",                  "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "DECK",  "name": "Deckers Outdoor",            "sector": "Consumer Cyclical", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "LULU",  "name": "Lululemon Athletica",        "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "ETSY",  "name": "Etsy Inc.",                  "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["russell2000"]},
    {"ticker": "CROX",  "name": "Crocs Inc.",                 "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["russell2000"]},
    {"ticker": "WSM",   "name": "Williams-Sonoma",            "sector": "Consumer Cyclical", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "FIVE",  "name": "Five Below Inc.",            "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["russell2000"]},

    # ── Consumer Defensive ───────────────────────────────────────────────────
    {"ticker": "WMT",   "name": "Walmart Inc.",               "sector": "Consumer Defensive", "exchange": "NYSE",   "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "PG",    "name": "Procter & Gamble",           "sector": "Consumer Defensive", "exchange": "NYSE",   "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "COST",  "name": "Costco Wholesale",           "sector": "Consumer Defensive", "exchange": "NASDAQ", "cap_tier": "mega",  "indices": ["sp500", "nasdaq100"]},
    {"ticker": "KO",    "name": "Coca-Cola Co.",              "sector": "Consumer Defensive", "exchange": "NYSE",   "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "PEP",   "name": "PepsiCo Inc.",               "sector": "Consumer Defensive", "exchange": "NASDAQ", "cap_tier": "mega",  "indices": ["sp500", "nasdaq100"]},
    {"ticker": "PM",    "name": "Philip Morris Intl.",        "sector": "Consumer Defensive", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "MO",    "name": "Altria Group",               "sector": "Consumer Defensive", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "CL",    "name": "Colgate-Palmolive",          "sector": "Consumer Defensive", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "MDLZ",  "name": "Mondelez International",    "sector": "Consumer Defensive", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "KMB",   "name": "Kimberly-Clark",             "sector": "Consumer Defensive", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "GIS",   "name": "General Mills",              "sector": "Consumer Defensive", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "KHC",   "name": "Kraft Heinz Co.",            "sector": "Consumer Defensive", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "K",     "name": "Kellanova",                  "sector": "Consumer Defensive", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "SYY",   "name": "Sysco Corp.",                "sector": "Consumer Defensive", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "HSY",   "name": "Hershey Co.",                "sector": "Consumer Defensive", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "EL",    "name": "Estée Lauder Cos.",          "sector": "Consumer Defensive", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},

    # ── Financial Services ───────────────────────────────────────────────────
    {"ticker": "JPM",   "name": "JPMorgan Chase",             "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "V",     "name": "Visa Inc.",                  "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "MA",    "name": "Mastercard Inc.",            "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "BAC",   "name": "Bank of America",            "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "GS",    "name": "Goldman Sachs",              "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "MS",    "name": "Morgan Stanley",             "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "BLK",   "name": "BlackRock Inc.",             "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "SCHW",  "name": "Charles Schwab",             "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "AXP",   "name": "American Express",           "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "C",     "name": "Citigroup Inc.",             "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "WFC",   "name": "Wells Fargo",                "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "USB",   "name": "U.S. Bancorp",               "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "PNC",   "name": "PNC Financial",              "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "TFC",   "name": "Truist Financial",           "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "SPGI",  "name": "S&P Global",                 "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "MCO",   "name": "Moody's Corp.",              "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "ICE",   "name": "Intercontinental Exchange", "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "CME",   "name": "CME Group",                  "sector": "Financial Services", "exchange": "NASDAQ","cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "CB",    "name": "Chubb Ltd.",                 "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "MMC",   "name": "Marsh & McLennan",           "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "AON",   "name": "Aon plc",                   "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "PYPL",  "name": "PayPal Holdings",            "sector": "Financial Services", "exchange": "NASDAQ","cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "COF",   "name": "Capital One Financial",     "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "BK",    "name": "Bank of New York Mellon",   "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},

    # ── Healthcare ───────────────────────────────────────────────────────────
    {"ticker": "UNH",   "name": "UnitedHealth Group",         "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "JNJ",   "name": "Johnson & Johnson",          "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "LLY",   "name": "Eli Lilly & Co.",            "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "ABBV",  "name": "AbbVie Inc.",                "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "MRK",   "name": "Merck & Co.",                "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "PFE",   "name": "Pfizer Inc.",                "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "TMO",   "name": "Thermo Fisher Scientific",  "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "ABT",   "name": "Abbott Laboratories",        "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "DHR",   "name": "Danaher Corp.",              "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "BMY",   "name": "Bristol-Myers Squibb",       "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "AMGN",  "name": "Amgen Inc.",                 "sector": "Healthcare", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "GILD",  "name": "Gilead Sciences",            "sector": "Healthcare", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "ISRG",  "name": "Intuitive Surgical",         "sector": "Healthcare", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "VRTX",  "name": "Vertex Pharmaceuticals",    "sector": "Healthcare", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "REGN",  "name": "Regeneron Pharmaceuticals", "sector": "Healthcare", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "MDT",   "name": "Medtronic plc",             "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "SYK",   "name": "Stryker Corp.",              "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "BSX",   "name": "Boston Scientific",          "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "EW",    "name": "Edwards Lifesciences",       "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "ZTS",   "name": "Zoetis Inc.",                "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "CI",    "name": "Cigna Group",                "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "HUM",   "name": "Humana Inc.",                "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "DXCM",  "name": "DexCom Inc.",                "sector": "Healthcare", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "IDXX",  "name": "IDEXX Laboratories",         "sector": "Healthcare", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "IQV",   "name": "IQVIA Holdings",             "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "ALGN",  "name": "Align Technology",            "sector": "Healthcare", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "VEEV",  "name": "Veeva Systems",               "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "BIO",   "name": "Bio-Rad Laboratories",       "sector": "Healthcare", "exchange": "NYSE",   "cap_tier": "mid",   "indices": ["sp500"]},
    {"ticker": "TNDM",  "name": "Tandem Diabetes Care",       "sector": "Healthcare", "exchange": "NASDAQ", "cap_tier": "small", "indices": ["russell2000"]},

    # ── Energy ───────────────────────────────────────────────────────────────
    {"ticker": "XOM",   "name": "Exxon Mobil Corp.",          "sector": "Energy", "exchange": "NYSE", "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "CVX",   "name": "Chevron Corp.",              "sector": "Energy", "exchange": "NYSE", "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "COP",   "name": "ConocoPhillips",             "sector": "Energy", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "EOG",   "name": "EOG Resources",              "sector": "Energy", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "SLB",   "name": "Schlumberger Ltd.",          "sector": "Energy", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "OXY",   "name": "Occidental Petroleum",      "sector": "Energy", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "MPC",   "name": "Marathon Petroleum",         "sector": "Energy", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "VLO",   "name": "Valero Energy",              "sector": "Energy", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "PSX",   "name": "Phillips 66",                "sector": "Energy", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "HES",   "name": "Hess Corp.",                 "sector": "Energy", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "DVN",   "name": "Devon Energy",               "sector": "Energy", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "HAL",   "name": "Halliburton Co.",            "sector": "Energy", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "FANG",  "name": "Diamondback Energy",         "sector": "Energy", "exchange": "NASDAQ","cap_tier": "large", "indices": ["sp500", "nasdaq100"]},

    # ── Industrials ──────────────────────────────────────────────────────────
    {"ticker": "GE",    "name": "GE Aerospace",               "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "CAT",   "name": "Caterpillar Inc.",           "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "mega",  "indices": ["sp500"]},
    {"ticker": "HON",   "name": "Honeywell International",   "sector": "Industrials", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "UNP",   "name": "Union Pacific",              "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "UPS",   "name": "United Parcel Service",      "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "RTX",   "name": "RTX Corp.",                  "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "BA",    "name": "Boeing Co.",                 "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "LMT",   "name": "Lockheed Martin",            "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "DE",    "name": "Deere & Co.",                "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "GD",    "name": "General Dynamics",            "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "NOC",   "name": "Northrop Grumman",           "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "WM",    "name": "Waste Management",           "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "RSG",   "name": "Republic Services",          "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "ITW",   "name": "Illinois Tool Works",        "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "EMR",   "name": "Emerson Electric",           "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "ETN",   "name": "Eaton Corp.",                "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "PH",    "name": "Parker-Hannifin",            "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "FAST",  "name": "Fastenal Co.",               "sector": "Industrials", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "CTAS",  "name": "Cintas Corp.",               "sector": "Industrials", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "FDX",   "name": "FedEx Corp.",                "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "CSX",   "name": "CSX Corp.",                  "sector": "Industrials", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "NSC",   "name": "Norfolk Southern",           "sector": "Industrials", "exchange": "NYSE",   "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "AXON",  "name": "Axon Enterprise",            "sector": "Industrials", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "CGNX",  "name": "Cognex Corp.",               "sector": "Industrials", "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["russell2000"]},
    {"ticker": "NDSN",  "name": "Nordson Corp.",              "sector": "Industrials", "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["russell2000"]},

    # ── Real Estate ──────────────────────────────────────────────────────────
    {"ticker": "PLD",   "name": "Prologis Inc.",              "sector": "Real Estate", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "AMT",   "name": "American Tower",             "sector": "Real Estate", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "CCI",   "name": "Crown Castle Intl.",        "sector": "Real Estate", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "EQIX",  "name": "Equinix Inc.",               "sector": "Real Estate", "exchange": "NASDAQ","cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "SPG",   "name": "Simon Property Group",      "sector": "Real Estate", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "O",     "name": "Realty Income",               "sector": "Real Estate", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "DLR",   "name": "Digital Realty Trust",       "sector": "Real Estate", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "PSA",   "name": "Public Storage",             "sector": "Real Estate", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "VICI",  "name": "VICI Properties",            "sector": "Real Estate", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "WELL",  "name": "Welltower Inc.",             "sector": "Real Estate", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},

    # ── Utilities ────────────────────────────────────────────────────────────
    {"ticker": "NEE",   "name": "NextEra Energy",             "sector": "Utilities", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "DUK",   "name": "Duke Energy",                "sector": "Utilities", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "SO",    "name": "Southern Co.",               "sector": "Utilities", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "D",     "name": "Dominion Energy",            "sector": "Utilities", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "AEP",   "name": "American Electric Power",   "sector": "Utilities", "exchange": "NASDAQ","cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "EXC",   "name": "Exelon Corp.",               "sector": "Utilities", "exchange": "NASDAQ","cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "SRE",   "name": "Sempra",                     "sector": "Utilities", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "XEL",   "name": "Xcel Energy",                "sector": "Utilities", "exchange": "NASDAQ","cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "CEG",   "name": "Constellation Energy",      "sector": "Utilities", "exchange": "NASDAQ","cap_tier": "large", "indices": ["sp500", "nasdaq100"]},
    {"ticker": "VST",   "name": "Vistra Corp.",               "sector": "Utilities", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},

    # ── Basic Materials ──────────────────────────────────────────────────────
    {"ticker": "LIN",   "name": "Linde plc",                 "sector": "Basic Materials", "exchange": "NASDAQ","cap_tier": "mega",  "indices": ["sp500", "nasdaq100"]},
    {"ticker": "APD",   "name": "Air Products & Chemicals",  "sector": "Basic Materials", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "SHW",   "name": "Sherwin-Williams",           "sector": "Basic Materials", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "ECL",   "name": "Ecolab Inc.",                "sector": "Basic Materials", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "DD",    "name": "DuPont de Nemours",          "sector": "Basic Materials", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "NEM",   "name": "Newmont Corp.",              "sector": "Basic Materials", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "FCX",   "name": "Freeport-McMoRan",          "sector": "Basic Materials", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "NUE",   "name": "Nucor Corp.",                "sector": "Basic Materials", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "DOW",   "name": "Dow Inc.",                   "sector": "Basic Materials", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    {"ticker": "PPG",   "name": "PPG Industries",             "sector": "Basic Materials", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},

    # ── S&P MidCap 400 (mid-cap, $2B–$10B) ──────────────────────────────────
    {"ticker": "MANH",  "name": "Manhattan Associates",       "sector": "Technology",     "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "OKTA",  "name": "Okta Inc.",                  "sector": "Technology",     "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "TWLO",  "name": "Twilio Inc.",                "sector": "Technology",     "exchange": "NYSE",   "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "DT",    "name": "Dynatrace Inc.",             "sector": "Technology",     "exchange": "NYSE",   "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "NTNX",  "name": "Nutanix Inc.",               "sector": "Technology",     "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "DBX",   "name": "Dropbox Inc.",               "sector": "Technology",     "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "PATH",  "name": "UiPath Inc.",                "sector": "Technology",     "exchange": "NYSE",   "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "SLAB",  "name": "Silicon Labs",               "sector": "Technology",     "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "CRUS",  "name": "Cirrus Logic",               "sector": "Technology",     "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "COHR",  "name": "Coherent Corp.",             "sector": "Technology",     "exchange": "NYSE",   "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "DUOL",  "name": "Duolingo Inc.",              "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "mid", "indices": ["sp400"]},
    {"ticker": "WING",  "name": "Wingstop Inc.",              "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "mid", "indices": ["sp400"]},
    {"ticker": "BURL",  "name": "Burlington Stores",          "sector": "Consumer Cyclical", "exchange": "NYSE", "cap_tier": "mid",  "indices": ["sp400"]},
    {"ticker": "TOL",   "name": "Toll Brothers",              "sector": "Consumer Cyclical", "exchange": "NYSE", "cap_tier": "mid",  "indices": ["sp400"]},
    {"ticker": "KBH",   "name": "KB Home",                    "sector": "Consumer Cyclical", "exchange": "NYSE", "cap_tier": "mid",  "indices": ["sp400"]},
    {"ticker": "TXRH",  "name": "Texas Roadhouse",            "sector": "Consumer Cyclical", "exchange": "NASDAQ","cap_tier": "mid",  "indices": ["sp400"]},
    {"ticker": "CAVA",  "name": "Cava Group",                 "sector": "Consumer Cyclical", "exchange": "NYSE", "cap_tier": "mid",  "indices": ["sp400"]},
    {"ticker": "NBIX",  "name": "Neurocrine Biosciences",     "sector": "Healthcare",    "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "HALO",  "name": "Halozyme Therapeutics",      "sector": "Healthcare",    "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "EXEL",  "name": "Exelixis Inc.",              "sector": "Healthcare",    "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "MEDP",  "name": "Medpace Holdings",           "sector": "Healthcare",    "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "CYTK",  "name": "Cytokinetics Inc.",          "sector": "Healthcare",    "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "PCTY",  "name": "Paylocity Holding",          "sector": "Industrials",   "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "SAIA",  "name": "Saia Inc.",                  "sector": "Industrials",   "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "STRL",  "name": "Sterling Infrastructure",    "sector": "Industrials",   "exchange": "NASDAQ", "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "EWBC",  "name": "East West Bancorp",          "sector": "Financial Services", "exchange": "NASDAQ", "cap_tier": "mid", "indices": ["sp400"]},
    {"ticker": "WAL",   "name": "Western Alliance Bancorp",   "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "mid",  "indices": ["sp400"]},
    {"ticker": "KNSL",  "name": "Kinsale Capital Group",      "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "mid",  "indices": ["sp400"]},
    {"ticker": "SFM",   "name": "Sprouts Farmers Market",     "sector": "Consumer Defensive", "exchange": "NASDAQ", "cap_tier": "mid","indices": ["sp400"]},
    {"ticker": "CELH",  "name": "Celsius Holdings",           "sector": "Consumer Defensive", "exchange": "NASDAQ", "cap_tier": "mid","indices": ["sp400"]},
    {"ticker": "MTDR",  "name": "Matador Resources",          "sector": "Energy",        "exchange": "NYSE",   "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "OVV",   "name": "Ovintiv Inc.",               "sector": "Energy",        "exchange": "NYSE",   "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "CLF",   "name": "Cleveland-Cliffs",           "sector": "Basic Materials","exchange": "NYSE",   "cap_tier": "mid",   "indices": ["sp400"]},
    {"ticker": "RPM",   "name": "RPM International",          "sector": "Basic Materials","exchange": "NYSE",   "cap_tier": "mid",   "indices": ["sp400"]},

    # ── S&P SmallCap 600 (small-cap, $300M–$2B) ─────────────────────────────
    {"ticker": "DOCN",  "name": "DigitalOcean Holdings",      "sector": "Technology",     "exchange": "NYSE",   "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "CALX",  "name": "Calix Inc.",                 "sector": "Technology",     "exchange": "NYSE",   "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "QTWO",  "name": "Q2 Holdings",                "sector": "Technology",     "exchange": "NYSE",   "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "POWI",  "name": "Power Integrations",         "sector": "Technology",     "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "BOX",   "name": "Box Inc.",                   "sector": "Technology",     "exchange": "NYSE",   "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "CRSR",  "name": "Corsair Gaming",             "sector": "Technology",     "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "SEDG",  "name": "SolarEdge Technologies",     "sector": "Technology",     "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "MARA",  "name": "MARA Holdings",              "sector": "Technology",     "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "CLSK",  "name": "CleanSpark Inc.",            "sector": "Technology",     "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "ENPH",  "name": "Enphase Energy",             "sector": "Technology",     "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "EAT",   "name": "Brinker International",      "sector": "Consumer Cyclical", "exchange": "NYSE", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "BOOT",  "name": "Boot Barn Holdings",         "sector": "Consumer Cyclical", "exchange": "NYSE", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "PENN",  "name": "Penn Entertainment",         "sector": "Consumer Cyclical", "exchange": "NASDAQ","cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "PLAY",  "name": "Dave & Buster's",            "sector": "Consumer Cyclical", "exchange": "NASDAQ","cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "SONO",  "name": "Sonos Inc.",                 "sector": "Consumer Cyclical", "exchange": "NASDAQ","cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "CORT",  "name": "Corcept Therapeutics",       "sector": "Healthcare",    "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "TGTX",  "name": "TG Therapeutics",            "sector": "Healthcare",    "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "INSP",  "name": "Inspire Medical Systems",    "sector": "Healthcare",    "exchange": "NYSE",   "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "KRYS",  "name": "Krystal Biotech",            "sector": "Healthcare",    "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "ADMA",  "name": "ADMA Biologics",             "sector": "Healthcare",    "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "SKYW",  "name": "SkyWest Inc.",               "sector": "Industrials",   "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "POWL",  "name": "Powell Industries",          "sector": "Industrials",   "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "AWI",   "name": "Armstrong World Industries", "sector": "Industrials",   "exchange": "NYSE",   "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "SPSC",  "name": "SPS Commerce",               "sector": "Technology",     "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "PI",    "name": "Impinj Inc.",                "sector": "Technology",     "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "PLMR",  "name": "Palomar Holdings",           "sector": "Financial Services", "exchange": "NYSE", "cap_tier": "small","indices": ["sp600"]},
    {"ticker": "GSHD",  "name": "Goosehead Insurance",        "sector": "Financial Services", "exchange": "NASDAQ","cap_tier": "small","indices": ["sp600"]},
    {"ticker": "CPRX",  "name": "Catalyst Pharmaceuticals",   "sector": "Healthcare",    "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "HRMY",  "name": "Harmony Biosciences",        "sector": "Healthcare",    "exchange": "NASDAQ", "cap_tier": "small", "indices": ["sp600"]},
    {"ticker": "BMI",   "name": "Badger Meter Inc.",          "sector": "Technology",     "exchange": "NYSE",   "cap_tier": "small", "indices": ["sp600"]},
]


def get_universe() -> list[StockEntry]:
    """Return the hardcoded stock universe (sync fallback).

    Prefer ``get_dynamic_universe()`` in async contexts — it tries FMP first.
    """
    return list(_UNIVERSE)


def get_sectors() -> list[str]:
    """Return sorted list of unique sectors in the hardcoded universe."""
    return sorted({s["sector"] for s in _UNIVERSE})


def get_tickers_by_index(index: str) -> list[str]:
    """Return tickers that belong to a given index (sp500, nasdaq100, russell2000).

    Uses the hardcoded universe.  For dynamic data, use
    ``get_dynamic_universe()`` and filter.
    """
    idx = index.lower()
    return [s["ticker"] for s in _UNIVERSE if idx in s["indices"]]


def get_tickers_by_sector(sector: str) -> list[str]:
    """Return tickers in a given sector (case-insensitive, hardcoded universe)."""
    s_lower = sector.lower()
    return [s["ticker"] for s in _UNIVERSE if s["sector"].lower() == s_lower]
