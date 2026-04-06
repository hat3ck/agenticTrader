"""Screening & Filtering Tool — Pre-filter the stock universe.

Uses FMP (Financial Modeling Prep) API for dynamic index constituents
(S&P 500, NASDAQ-100) with a hardcoded fallback, plus yfinance for live
market data enrichment.

Filters by:
  • Market cap range (small / mid / large / mega)
  • Sector (Technology, Healthcare, Financial Services, etc.)
  • Minimum average daily volume (ensures liquidity)
  • Exchange (NYSE, NASDAQ)
  • Basic valuation thresholds (P/E < threshold to exclude extreme outliers)
  • Risk tolerance (conservative / moderate / aggressive)

This is the FIRST tool the agent should call in any recommendation workflow.
It narrows the universe from hundreds of stocks to 10-30 candidates that the
other tools then evaluate deeply.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import yfinance as yf

from app.data.cache import screener_cache, async_get_or_set
from app.data.constituents import get_dynamic_universe, get_universe_source_sync

logger = logging.getLogger(__name__)

# Thread pool shared across screener calls (capped to avoid hammering Yahoo)
_executor = ThreadPoolExecutor(max_workers=8)

# ── Market-cap tier boundaries (USD) ────────────────────────────────────────

MARKET_CAP_TIERS: dict[str, tuple[float, float]] = {
    "mega":  (200e9, float("inf")),
    "large": (10e9, 200e9),
    "mid":   (2e9, 10e9),
    "small": (300e6, 2e9),
}


def _resolve_cap_range(market_cap_range: str) -> tuple[float, float]:
    """Convert a cap-range label to ``(min, max)`` bounds in USD."""
    if market_cap_range in ("all", "auto"):
        return (0, float("inf"))
    if market_cap_range in MARKET_CAP_TIERS:
        return MARKET_CAP_TIERS[market_cap_range]
    # Convenient combined ranges
    if market_cap_range == "large_and_above":
        return (10e9, float("inf"))
    if market_cap_range == "mid_and_above":
        return (2e9, float("inf"))
    if market_cap_range == "small_and_above":
        return (300e6, float("inf"))
    # Default: include mid-cap and above instead of excluding smaller stocks
    return (2e9, float("inf"))


def _cap_tiers_for_range(min_cap: float, max_cap: float) -> set[str]:
    """Return the set of tier names whose range overlaps [min_cap, max_cap]."""
    tiers: set[str] = set()
    for tier, (lo, hi) in MARKET_CAP_TIERS.items():
        if lo < max_cap and hi > min_cap:
            tiers.add(tier)
    return tiers or {"mega", "large"}  # safe fallback


# ── Static pre-filter (no network) ──────────────────────────────────────────

def _static_prefilter(
    universe: list[dict],
    excluded: set[str],
    sector_prefs: set[str] | None,
    exchange_set: set[str] | None,
    index_set: set[str] | None,
    cap_tiers: set[str],
) -> list[dict]:
    """In-memory filter on hardcoded metadata — very fast, no API calls."""
    result: list[dict] = []
    for stock in universe:
        if stock["ticker"] in excluded:
            continue
        if exchange_set and stock["exchange"] not in exchange_set:
            continue
        if index_set and not (set(stock["indices"]) & index_set):
            continue
        if cap_tiers and stock["cap_tier"] not in cap_tiers:
            continue
        result.append(stock)

    # If sector preferences given, sort preferred sectors first (but keep others)
    if sector_prefs:
        preferred = [s for s in result if s["sector"].lower() in sector_prefs]
        others = [s for s in result if s["sector"].lower() not in sector_prefs]
        result = preferred + others

    return result


# ── yfinance helpers (run in thread pool) ────────────────────────────────────

def _batch_download_volume(tickers: list[str]) -> dict[str, float]:
    """Batch-download 1 month of daily data and return avg daily volume per ticker.

    Uses a single ``yf.download`` call which is far more efficient than
    fetching each ticker individually.
    """
    if not tickers:
        return {}
    try:
        data = yf.download(
            tickers,
            period="1mo",
            progress=False,
            threads=True,
        )
        if data.empty:
            return {}

        volumes: dict[str, float] = {}
        if len(tickers) == 1:
            # Single-ticker download → flat columns
            if "Volume" in data.columns:
                avg = data["Volume"].mean()
                if avg == avg:  # not NaN
                    volumes[tickers[0]] = float(avg)
        else:
            # Multi-ticker download → MultiIndex columns
            vol_df = None
            try:
                if "Volume" in data.columns.get_level_values(0):
                    vol_df = data["Volume"]
            except AttributeError:
                pass

            if vol_df is None:
                # Try alternate column layout (Ticker, Metric)
                try:
                    vol_df = data.xs("Volume", axis=1, level=1)
                except (KeyError, TypeError):
                    pass

            if vol_df is not None:
                for t in tickers:
                    try:
                        col = vol_df[t] if t in vol_df.columns else None
                    except (KeyError, TypeError):
                        col = None
                    if col is not None and col.notna().any():
                        volumes[t] = float(col.mean())

        return volumes

    except Exception:
        logger.exception("yfinance batch volume download failed")
        return {}


def _fetch_ticker_info(ticker: str) -> dict | None:
    """Fetch key screening fields for a single ticker via ``yf.Ticker.info``."""
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get("quoteType") is None:
            return None
        trailing_pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        return {
            "ticker": ticker,
            "market_cap": info.get("marketCap"),
            "pe_ratio": trailing_pe if trailing_pe is not None else forward_pe,
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "avg_volume": info.get("averageVolume"),
            "name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        }
    except Exception:
        logger.debug("yfinance info fetch failed for %s", ticker, exc_info=True)
        return None


# ── Yahoo Finance screening (custom market-cap range) ────────────────────────

# Map Yahoo exchange codes → our canonical names
_YF_EXCHANGE_MAP: dict[str, str] = {
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ",
    "NYQ": "NYSE", "ASE": "NYSE", "PCX": "NYSE",
}

# Major US exchanges on Yahoo (excludes OTC/Pink Sheets)
_YF_MAJOR_EXCHANGES = ("NMS", "NYQ", "NGM", "NCM", "ASE", "PCX")


def _yf_screen_by_market_cap_sync(
    cap_min: float,
    cap_max: float,
    volume_min: int = 100_000,
    limit: int = 250,
) -> list[dict]:
    """Use ``yf.screen()`` + ``EquityQuery`` to find US stocks in a market-cap range.

    This is a **free** alternative to FMP's stock-screener endpoint.
    It queries the Yahoo Finance screener API via yfinance, which requires
    no API key.

    Returns a list of dicts compatible with the universe StockEntry format.
    """
    try:
        filters = [
            yf.EquityQuery("eq", ["region", "us"]),
            yf.EquityQuery("btwn", ["intradaymarketcap", cap_min, cap_max]),
            yf.EquityQuery("gt", ["avgdailyvol3m", volume_min]),
            yf.EquityQuery("is-in", ["exchange", *_YF_MAJOR_EXCHANGES]),
        ]
        query = yf.EquityQuery("and", filters)
        response = yf.screen(
            query,
            sortField="intradaymarketcap",
            sortAsc=False,
            size=min(limit, 250),
        )
        quotes = response.get("quotes", [])
        if not quotes:
            return []

        from app.data.constituents import _classify_cap_tier  # noqa: avoid circular at module level

        results: list[dict] = []
        for q in quotes:
            symbol = q.get("symbol")
            if not symbol or "." in symbol:
                continue
            exchange_raw = q.get("exchange", "")
            exchange = _YF_EXCHANGE_MAP.get(exchange_raw, "NYSE")
            market_cap = q.get("marketCap")
            results.append({
                "ticker": symbol.upper(),
                "name": q.get("shortName") or q.get("longName") or "",
                "sector": q.get("sector", "Unknown"),
                "exchange": exchange,
                "cap_tier": _classify_cap_tier(market_cap),
                "indices": [],
                # Pre-attach live data so the enrichment step can use it
                "market_cap": market_cap,
                "avg_daily_volume": q.get("averageDailyVolume3Month"),
                "price": q.get("regularMarketPrice"),
                "pe_ratio": q.get("trailingPE") or q.get("forwardPE"),
            })

        logger.info(
            "yfinance screener: found %d stocks for cap range $%s–$%s",
            len(results), f"{cap_min:,.0f}", f"{cap_max:,.0f}",
        )
        return results
    except Exception:
        logger.exception("yfinance screen by market cap failed")
        return []


async def _yf_screen_by_market_cap(
    cap_min: float,
    cap_max: float,
    volume_min: int = 100_000,
    limit: int = 250,
) -> list[dict]:
    """Async wrapper around the synchronous yfinance screener call."""
    return await asyncio.to_thread(
        _yf_screen_by_market_cap_sync, cap_min, cap_max, volume_min, limit,
    )


# ── Public API ───────────────────────────────────────────────────────────────

async def screen_stocks(
    risk_tolerance: str = "moderate",
    sector_preferences: list[str] | None = None,
    excluded_tickers: list[str] | None = None,
    market_cap_range: str = "large_and_above",
    min_avg_daily_volume: int = 500_000,
    exchanges: list[str] | None = None,
    max_pe_ratio: float = 50.0,
    indices: list[str] | None = None,
    max_results: int = 25,
    data_source: str = "auto",
    market_cap_min_override: float | None = None,
    market_cap_max_override: float | None = None,
) -> dict:
    """Screen the stock universe and return a filtered list of candidates.

    Args:
        risk_tolerance: ``'conservative'``, ``'moderate'``, or ``'aggressive'``
        sector_preferences: Optional list of preferred sectors
            (e.g. ``["Technology", "Healthcare"]``)
        excluded_tickers: Tickers to exclude from results
        market_cap_range: Cap-size filter — one of ``'mega'``, ``'large'``,
            ``'mid'``, ``'small'``, ``'large_and_above'``, ``'mid_and_above'``,
            or ``'all'``
        min_avg_daily_volume: Minimum 30-day average daily volume
            (default 500 000 shares — ensures liquidity)
        exchanges: Optional exchange filter, e.g. ``["NYSE"]`` or
            ``["NASDAQ"]``.  ``None`` means all exchanges.
        max_pe_ratio: Exclude stocks with trailing P/E above this value
            (default 50 — filters extreme outlier valuations)
        indices: Optional filter by index membership, e.g.
            ``["sp500"]``, ``["nasdaq100"]``, ``["russell2000"]``
        max_results: Maximum number of candidates to return (default 25)

    Returns a dict with:
        - **candidates**: List of dicts, each containing ``ticker``, ``name``,
          ``sector``, ``market_cap``, ``exchange``, ``pe_ratio``,
          ``avg_daily_volume``, ``price``
        - **total_screened**: How many stocks were in the universe
        - **candidates_returned**: Length of the candidates list
        - **filters_applied**: Human-readable summary of active filters

    This is the **first** tool the agent should call in any recommendation
    workflow.  It narrows the universe from hundreds of stocks to a
    manageable 10-30 candidates that the other tools then evaluate deeply.
    """

    async def _fetch() -> dict:
        # ── 1. Build filter sets ─────────────────────────────────────────
        excluded = {t.upper() for t in (excluded_tickers or [])}
        sector_prefs = (
            {s.lower() for s in sector_preferences} if sector_preferences else None
        )
        exchange_set = (
            {e.upper() for e in exchanges} if exchanges else None
        )
        index_set = (
            {i.lower() for i in indices} if indices else None
        )

        min_cap, max_cap = _resolve_cap_range(market_cap_range)

        # Apply user-specified market cap overrides (in USD).
        # These REPLACE the risk-based bounds — the user's explicit
        # request must take priority, otherwise the risk-based floor
        # (e.g. 2 B for "mid_and_above") can invert the range when the
        # user asks for a lower cap (e.g. 0–1 B → min=2B, max=1B → ∅).
        if market_cap_min_override is not None:
            min_cap = market_cap_min_override
        if market_cap_max_override is not None:
            max_cap = market_cap_max_override

        cap_tiers = _cap_tiers_for_range(min_cap, max_cap)

        # ── 2. Static pre-filter (instant) ───────────────────────────────
        universe = await get_dynamic_universe(data_source=data_source)
        total_screened = len(universe)

        # Dynamic sources (FMP/Wikipedia) use placeholder cap_tiers, so skip
        # cap-tier prefiltering — the yfinance enrichment phase will apply the
        # actual market-cap filter from live data instead.
        source = get_universe_source_sync(data_source=data_source)
        pre_cap_tiers = cap_tiers if source == "hardcoded" else set()

        prefiltered = _static_prefilter(
            universe, excluded, sector_prefs, exchange_set, index_set, pre_cap_tiers,
        )

        # ── 2b. Targeted screener for custom market-cap ranges ────────────
        # The standard universe (S&P 500, NASDAQ-100, etc.) mostly covers
        # stocks above ~$1B.  When the user requests a cap range that falls
        # partially or entirely below that, supplement the universe with
        # stocks found via the Yahoo Finance screener API (free, no key).
        if market_cap_min_override is not None or market_cap_max_override is not None:
            yf_extra = await _yf_screen_by_market_cap(
                cap_min=max(min_cap, 1.0),
                cap_max=max_cap if max_cap < float("inf") else 1e15,
                volume_min=min_avg_daily_volume,
            )
            if yf_extra:
                existing_tickers = {s["ticker"] for s in prefiltered}
                new_stocks = []
                for stock in yf_extra:
                    if stock["ticker"] not in existing_tickers and stock["ticker"] not in excluded:
                        new_stocks.append(stock)
                        existing_tickers.add(stock["ticker"])
                # Prepend screener stocks — they're confirmed in-range and
                # must not be truncated by _BATCH_LIMIT in favour of the
                # standard universe stocks (which are mostly out of range).
                prefiltered = new_stocks + prefiltered
                total_screened += len(yf_extra)
                logger.info(
                    "Supplemented universe with %d stocks from yfinance screener "
                    "(cap range $%s–$%s)",
                    len(yf_extra), f"{min_cap:,.0f}", f"{max_cap:,.0f}",
                )

        if not prefiltered:
            return _empty_result(
                total_screened, market_cap_range, risk_tolerance,
                sector_preferences, exchanges, indices,
                excluded_tickers, min_avg_daily_volume, max_pe_ratio,
            )

        # Limit how many tickers we send to yfinance (avoid timeouts).
        # Pre-enriched stocks (from yfinance screener) skip the expensive
        # individual .info calls, so they're cheap to keep.
        _BATCH_LIMIT = 250
        prefiltered = prefiltered[:_BATCH_LIMIT]
        tickers = [s["ticker"] for s in prefiltered]

        # ── 3. Batch volume screening (single yf.download call) ──────────
        volume_data = await asyncio.to_thread(_batch_download_volume, tickers)

        volume_filtered: list[dict] = []
        for stock in prefiltered:
            avg_vol = volume_data.get(stock["ticker"])
            if avg_vol is not None:
                if avg_vol < min_avg_daily_volume:
                    continue  # not liquid enough
                stock = {**stock, "avg_daily_volume": int(avg_vol)}
            # If yfinance didn't return volume, keep the stock (benefit of doubt)
            volume_filtered.append(stock)

        if not volume_filtered:
            return _empty_result(
                total_screened, market_cap_range, risk_tolerance,
                sector_preferences, exchanges, indices,
                excluded_tickers, min_avg_daily_volume, max_pe_ratio,
            )

        # ── 4. Detailed info fetch (P/E, market cap, price) ─────────────
        # Run individual .info calls in the thread pool concurrently.
        # Skip tickers that already have enriched data (e.g. from yfinance
        # screener in step 2b) — they already carry market_cap, price, etc.
        _INFO_LIMIT = 50
        tickers_needing_info = [
            s["ticker"] for s in volume_filtered[:_INFO_LIMIT]
            if s.get("market_cap") is None
        ]

        loop = asyncio.get_running_loop()
        info_futures = [
            loop.run_in_executor(_executor, _fetch_ticker_info, t)
            for t in tickers_needing_info
        ]
        info_results = await asyncio.gather(*info_futures, return_exceptions=True)

        info_map: dict[str, dict] = {}
        for res in info_results:
            if isinstance(res, dict) and res is not None:
                info_map[res["ticker"]] = res

        # ── 5. Apply P/E and market-cap filters, enrich data ────────────
        candidates: list[dict] = []
        for stock in volume_filtered:
            info = info_map.get(stock["ticker"])
            enriched = dict(stock)  # shallow copy

            if info:
                if info.get("market_cap") is not None:
                    enriched["market_cap"] = info["market_cap"]
                if info.get("pe_ratio") is not None:
                    enriched["pe_ratio"] = round(info["pe_ratio"], 2)
                if info.get("price") is not None:
                    enriched["price"] = round(info["price"], 2)
                if info.get("name"):
                    enriched["name"] = info["name"]
                if info.get("avg_volume") is not None and "avg_daily_volume" not in enriched:
                    enriched["avg_daily_volume"] = info["avg_volume"]

                # P/E filter — skip negative earnings (< 0) or extreme (> max)
                pe = info.get("pe_ratio")
                if pe is not None and (pe < 0 or pe > max_pe_ratio):
                    continue

                # Exact market-cap filter from live data
                mc = info.get("market_cap")
                if mc is not None and (mc < min_cap or mc > max_cap):
                    continue

            # When user specified explicit market cap bounds, enforce them even if yfincance doesn't return a market cap
            if market_cap_min_override is not None or market_cap_max_override is not None:
                mc = enriched.get("market_cap")
                if mc is None or mc < min_cap or mc > max_cap:
                    continue

            # Build clean candidate dict (drop internal fields like indices)
            candidates.append({
                "ticker": enriched["ticker"],
                "name": enriched.get("name", ""),
                "sector": enriched.get("sector", ""),
                "market_cap": enriched.get("market_cap"),
                "exchange": enriched.get("exchange", ""),
                "pe_ratio": enriched.get("pe_ratio"),
                "avg_daily_volume": enriched.get("avg_daily_volume"),
                "price": enriched.get("price"),
            })

        # ── 6. Risk-based ranking ────────────────────────────────────────
        if risk_tolerance == "conservative":
            # Favour mega/large cap, sorted by market cap descending
            candidates.sort(
                key=lambda x: x.get("market_cap") or 0, reverse=True,
            )
        elif risk_tolerance == "aggressive":
            # Favour smaller caps (more upside potential), cap ascending
            candidates.sort(
                key=lambda x: x.get("market_cap") or float("inf"),
            )
        # "moderate" keeps the sector-preference ordering from step 2

        candidates = candidates[:max_results]

        logger.info(
            "Screener returning %d candidates: %s",
            len(candidates),
            ", ".join(c["ticker"] for c in candidates[:15]),
        )

        # ── 7. Build response ────────────────────────────────────────────
        filters_desc = (
            f"risk={risk_tolerance}, cap_range={market_cap_range}, "
            f"min_volume={min_avg_daily_volume:,}, max_pe={max_pe_ratio}"
        )
        if sector_preferences:
            filters_desc += f", sectors={sector_preferences}"
        if exchanges:
            filters_desc += f", exchanges={exchanges}"
        if indices:
            filters_desc += f", indices={indices}"
        if excluded_tickers:
            filters_desc += f", excluded={excluded_tickers}"

        return {
            "candidates": candidates,
            "total_screened": total_screened,
            "candidates_returned": len(candidates),
            "filters_applied": filters_desc,
        }

    # Cache key incorporates all filter params
    cache_key = (
        f"screen:{risk_tolerance}:{market_cap_range}:{min_avg_daily_volume}:"
        f"{max_pe_ratio}:{sorted(sector_preferences or [])}:"
        f"{sorted(excluded_tickers or [])}:{sorted(exchanges or [])}:"
        f"{sorted(indices or [])}:{market_cap_min_override}:{market_cap_max_override}"
    )
    return await async_get_or_set(screener_cache, cache_key, _fetch)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _empty_result(
    total_screened: int,
    market_cap_range: str,
    risk_tolerance: str,
    sector_preferences: list[str] | None,
    exchanges: list[str] | None,
    indices: list[str] | None,
    excluded_tickers: list[str] | None,
    min_avg_daily_volume: int,
    max_pe_ratio: float,
) -> dict:
    filters_desc = (
        f"risk={risk_tolerance}, cap_range={market_cap_range}, "
        f"min_volume={min_avg_daily_volume:,}, max_pe={max_pe_ratio}"
    )
    if sector_preferences:
        filters_desc += f", sectors={sector_preferences}"
    if exchanges:
        filters_desc += f", exchanges={exchanges}"
    if indices:
        filters_desc += f", indices={indices}"
    if excluded_tickers:
        filters_desc += f", excluded={excluded_tickers}"
    return {
        "candidates": [],
        "total_screened": total_screened,
        "candidates_returned": 0,
        "filters_applied": filters_desc,
    }
