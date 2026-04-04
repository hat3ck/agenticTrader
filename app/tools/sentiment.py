"""Sentiment & Macro Analysis Tool.

Data sources:
  • News API  (newsapi.org, free tier)  — recent headlines about stocks/sectors
  • FRED API  (Federal Reserve Economic Data, free) — macroeconomic indicators:
    interest rates (Fed Funds Rate), inflation (CPI), GDP growth, unemployment,
    yield-curve spread
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import httpx

from app.config import settings
from app.data.cache import sentiment_cache, macro_cache, async_get_or_set

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_NEWS_API_BASE = "https://newsapi.org/v2"
_FRED_API_BASE = "https://api.stlouisfed.org/fred"

# FRED series IDs
_FRED_SERIES = {
    "fed_rate": "FEDFUNDS",          # Federal Funds Effective Rate
    "cpi": "CPIAUCSL",               # Consumer Price Index (All Urban)
    "gdp_growth": "A191RL1Q225SBEA", # Real GDP growth rate (quarterly, annualised)
    "unemployment": "UNRATE",        # Unemployment Rate
    "treasury_10y": "DGS10",         # 10-Year Treasury Constant Maturity Rate
    "treasury_2y": "DGS2",           # 2-Year Treasury Constant Maturity Rate
    "treasury_3mo": "DTB3",          # 3-Month Treasury Bill Rate
}

# Sector rotation heuristics based on macro conditions
_SECTOR_RATE_SENSITIVITY: dict[str, float] = {
    "Technology": -0.8,          # hurt by rising rates
    "Real Estate": -0.9,        # very rate-sensitive
    "Consumer Cyclical": -0.5,
    "Consumer Defensive": 0.3,  # defensive, benefits mildly
    "Utilities": 0.4,           # bond-proxy, inversely correlated
    "Healthcare": 0.2,          # relatively insensitive
    "Financial Services": 0.6,  # benefits from higher rates
    "Energy": 0.1,              # mostly commodity-driven
    "Industrials": -0.2,
    "Communication Services": -0.4,
    "Basic Materials": 0.0,
}


# ── News API helpers ─────────────────────────────────────────────────────────

async def _fetch_news_headlines(ticker: str) -> list[dict] | None:
    """Fetch recent headlines from News API for *ticker*. Returns None on failure."""
    api_key = settings.news_api_key
    if not api_key:
        return None

    # Search for both the ticker symbol and broader stock-related query
    query = f'"{ticker.upper()}" OR "{ticker.upper()} stock"'
    from_date = (datetime.now(tz=timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    url = f"{_NEWS_API_BASE}/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "relevancy",
        "pageSize": 10,
        "language": "en",
        "apiKey": api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = data.get("articles", [])
        if not articles:
            logger.info("News API returned no articles for %s", ticker)
            return []

        headlines: list[dict] = []
        for art in articles:
            title = art.get("title", "")
            # Skip removed/empty articles
            if not title or title == "[Removed]":
                continue
            headlines.append({
                "title": title,
                "description": (art.get("description") or "")[:200],
                "source": art.get("source", {}).get("name", "Unknown"),
                "published_at": art.get("publishedAt", ""),
                "url": art.get("url", ""),
            })

        return headlines

    except httpx.HTTPStatusError as exc:
        logger.warning("News API HTTP error for %s: %s", ticker, exc.response.status_code)
        return None
    except Exception:
        logger.exception("News API request failed for %s", ticker)
        return None


def _simple_headline_sentiment(headlines: list[dict]) -> tuple[float, list[dict]]:
    """Lightweight keyword-based sentiment scoring for headlines.

    Returns (aggregate_score, annotated_headlines) where score is in [-1, 1].
    The LLM can interpret the raw headlines more deeply at reasoning time —
    this score is just a quick directional signal.
    """
    positive_words = {
        "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "rallies",
        "gain", "gains", "beat", "beats", "exceeds", "record", "upgrade", "upgraded",
        "strong", "bullish", "optimistic", "growth", "profit", "boom", "breakout",
        "outperform", "buy", "positive", "accelerate", "boost", "innovation",
    }
    negative_words = {
        "fall", "falls", "drop", "drops", "plunge", "plunges", "crash", "crashes",
        "decline", "declines", "miss", "misses", "downgrade", "downgraded", "weak",
        "bearish", "pessimistic", "loss", "losses", "concern", "risk", "warning",
        "sell", "negative", "slowdown", "recession", "layoff", "layoffs", "cut",
        "debt", "lawsuit", "investigation", "probe", "fraud", "scandal",
    }

    scores: list[float] = []
    annotated: list[dict] = []

    for hl in headlines:
        title_lower = (hl.get("title", "") + " " + hl.get("description", "")).lower()
        words = set(title_lower.split())

        pos_hits = len(words & positive_words)
        neg_hits = len(words & negative_words)
        total = pos_hits + neg_hits

        if total == 0:
            score = 0.0
            label = "neutral"
        else:
            score = (pos_hits - neg_hits) / total
            label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"

        scores.append(score)
        annotated.append({**hl, "sentiment": label, "sentiment_score": round(score, 2)})

    agg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    return agg_score, annotated


# ── FRED API helpers ─────────────────────────────────────────────────────────

async def _fetch_fred_series(series_id: str, limit: int = 6) -> list[dict] | None:
    """Fetch the latest *limit* observations for a FRED series."""
    api_key = settings.fred_api_key
    if not api_key:
        return None

    url = f"{_FRED_API_BASE}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        return data.get("observations", [])
    except httpx.HTTPStatusError as exc:
        logger.warning("FRED API HTTP error for %s: %s", series_id, exc.response.status_code)
        return None
    except Exception:
        logger.exception("FRED API request failed for %s", series_id)
        return None


def _latest_fred_value(observations: list[dict] | None) -> float | None:
    """Extract the most recent numeric value from FRED observations."""
    if not observations:
        return None
    for obs in observations:
        val = obs.get("value", ".")
        if val != ".":
            try:
                return round(float(val), 2)
            except (ValueError, TypeError):
                continue
    return None


def _fred_trend(observations: list[dict] | None, periods: int = 3) -> str:
    """Determine trend direction from the latest *periods* FRED observations.

    Returns 'rising', 'falling', or 'stable'.
    """
    if not observations or len(observations) < 2:
        return "unknown"

    values: list[float] = []
    for obs in observations[:periods]:
        val = obs.get("value", ".")
        if val != ".":
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                continue

    if len(values) < 2:
        return "unknown"

    # Observations are sorted desc (newest first) — reverse for chronological
    values.reverse()
    diff = values[-1] - values[0]
    threshold = 0.15  # small moves are "stable"

    if diff > threshold:
        return "rising"
    elif diff < -threshold:
        return "falling"
    return "stable"


def _infer_fed_direction(rate: float | None, trend: str) -> str:
    """Infer whether the Fed is hiking, holding, or cutting."""
    if trend == "rising":
        return "hiking"
    elif trend == "falling":
        return "cutting"
    return "holding"


def _infer_yield_curve(t10y: float | None, t2y: float | None, t3mo: float | None) -> str:
    """Classify yield curve shape from Treasury rates."""
    if t10y is None:
        return "unknown"

    # Prefer 10Y-2Y spread, fall back to 10Y-3M
    short_rate = t2y if t2y is not None else t3mo
    if short_rate is None:
        return "unknown"

    spread = t10y - short_rate
    if spread > 0.5:
        return "normal"
    elif spread < -0.1:
        return "inverted"
    return "flat"


def _compute_sector_signals(
    fed_direction: str,
    inflation_trend: str,
    yield_curve: str,
) -> dict[str, str]:
    """Derive sector headwind/tailwind signals from macro conditions."""
    signals: dict[str, str] = {}

    rate_factor = {"hiking": -1, "holding": 0, "cutting": 1}.get(fed_direction, 0)
    inflation_factor = {"rising": -0.5, "stable": 0, "falling": 0.5}.get(inflation_trend, 0)
    curve_factor = {"normal": 0.5, "flat": 0, "inverted": -0.5}.get(yield_curve, 0)

    for sector, sensitivity in _SECTOR_RATE_SENSITIVITY.items():
        # Positive sensitivity means sector benefits from higher rates
        score = (
            sensitivity * rate_factor
            + inflation_factor * 0.3
            + curve_factor * 0.3
        )
        if score > 0.2:
            signals[sector] = "tailwind"
        elif score < -0.2:
            signals[sector] = "headwind"
        else:
            signals[sector] = "neutral"

    return signals


def _infer_market_regime(
    gdp: float | None,
    unemployment: float | None,
    inflation_trend: str,
    yield_curve: str,
) -> str:
    """Simple market regime classification."""
    if yield_curve == "inverted":
        return "late_cycle_caution"
    if gdp is not None and gdp < 0:
        return "recession"
    if gdp is not None and gdp > 3.0:
        if inflation_trend == "rising":
            return "overheating"
        return "strong_bull"
    if gdp is not None and gdp > 1.5:
        return "moderate_bull"
    return "slow_growth"


# ── Public API ───────────────────────────────────────────────────────────────

async def get_news_sentiment(ticker: str) -> dict:
    """Get recent news headlines and sentiment for a ticker.

    Returns a dict with:
    - ticker: The stock symbol
    - data_source: 'newsapi'
    - overall_sentiment: 'positive', 'negative', or 'neutral'
    - sentiment_score: Float from -1 (very negative) to +1 (very positive)
    - headlines: List of recent headline dicts with title, sentiment, source, date
    - summary: Natural-language summary of current sentiment

    Use this to understand market perception and news-driven momentum.
    Positive sentiment with strong fundamentals = higher conviction.
    Negative sentiment on strong fundamentals = potential contrarian opportunity.
    """

    async def _fetch():
        # Try real News API first
        headlines = await _fetch_news_headlines(ticker)

        if headlines is None:
            # API key not set or request failed
            logger.warning(
                "Cannot fetch sentiment for %s (NEWS_API_KEY not set or request failed)",
                ticker,
            )
            return {
                "ticker": ticker.upper(),
                "data_source": "unavailable",
                "overall_sentiment": "unknown",
                "sentiment_score": 0.0,
                "headlines": [],
                "error": "NEWS_API_KEY is not configured or the request failed.",
                "summary": (
                    f"Sentiment data unavailable for {ticker.upper()}. "
                    "Configure NEWS_API_KEY to enable live news sentiment."
                ),
            }

        if not headlines:
            # API returned no results
            return {
                "ticker": ticker.upper(),
                "data_source": "newsapi",
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "headlines": [],
                "summary": f"No recent news articles found for {ticker.upper()}.",
            }

        # Score headlines
        agg_score, annotated_headlines = _simple_headline_sentiment(headlines)
        sentiment_label = (
            "positive" if agg_score > 0.15
            else "negative" if agg_score < -0.15
            else "neutral"
        )

        return {
            "ticker": ticker.upper(),
            "data_source": "newsapi",
            "overall_sentiment": sentiment_label,
            "sentiment_score": agg_score,
            "headlines": annotated_headlines,
            "summary": (
                f"Overall sentiment for {ticker.upper()} is {sentiment_label} "
                f"(score: {agg_score}) based on {len(annotated_headlines)} recent "
                f"articles. News coverage is "
                f"{'favorable' if agg_score > 0 else 'mixed to negative'}."
            ),
        }

    return await async_get_or_set(sentiment_cache, f"sent:{ticker.upper()}", _fetch)


async def get_macro_environment() -> dict:
    """Get current macroeconomic environment snapshot.

    Returns a dict with:
    - data_source: 'fred'
    - fed_rate: Current federal funds rate (percentage)
    - fed_direction: 'hiking', 'holding', or 'cutting'
    - inflation_rate: Current CPI inflation (percentage, year-over-year approx)
    - inflation_trend: 'rising', 'stable', or 'falling'
    - yield_curve: 'normal', 'flat', or 'inverted'
    - gdp_growth: GDP growth rate (percentage, annualised quarterly)
    - unemployment: Unemployment rate (percentage)
    - treasury_10y: 10-Year Treasury rate
    - treasury_2y: 2-Year Treasury rate
    - treasury_3mo: 3-Month Treasury Bill rate
    - yield_spread_10y_2y: 10Y minus 2Y spread
    - market_regime: Classified regime (e.g. 'moderate_bull', 'recession')
    - sector_signals: Dict mapping sectors to 'headwind', 'tailwind', or 'neutral'
    - summary: Natural-language macro summary

    Use this tool for macro context when making investment decisions.
    Rising rates = headwind for growth/tech, tailwind for financials.
    Inverted yield curve = recession signal.
    Low unemployment + high inflation => Fed likely to stay hawkish.
    """

    async def _fetch():
        if not settings.fred_api_key:
            logger.warning("FRED_API_KEY is not configured — macro data unavailable")
            return {
                "data_source": "unavailable",
                "error": "FRED_API_KEY is not configured.",
                "summary": "Macro environment data unavailable. Configure FRED_API_KEY to enable live macro data.",
            }

        # Fetch all FRED series in parallel
        gathered = await asyncio.gather(
            *[_fetch_fred_series(sid, limit=6) for sid in _FRED_SERIES.values()],
            return_exceptions=True,
        )
        results: dict[str, list[dict] | None] = {}
        for (name, _sid), result in zip(_FRED_SERIES.items(), gathered):
            if isinstance(result, Exception):
                logger.warning("FRED fetch failed for %s: %s", name, result)
                results[name] = None
            else:
                results[name] = result

        # Extract latest values
        fed_rate = _latest_fred_value(results.get("fed_rate"))
        gdp_growth = _latest_fred_value(results.get("gdp_growth"))
        unemployment = _latest_fred_value(results.get("unemployment"))
        t10y = _latest_fred_value(results.get("treasury_10y"))
        t2y = _latest_fred_value(results.get("treasury_2y"))
        t3mo = _latest_fred_value(results.get("treasury_3mo"))

        # CPI -> approximate annualised inflation rate from recent observations
        cpi_obs = results.get("cpi")
        inflation_rate: float | None = None
        if cpi_obs and len(cpi_obs) >= 2:
            vals: list[float] = []
            for obs in cpi_obs:
                v = obs.get("value", ".")
                if v != ".":
                    try:
                        vals.append(float(v))
                    except (ValueError, TypeError):
                        continue
            if len(vals) >= 2:
                recent, older = vals[0], vals[-1]
                if older > 0:
                    months_span = len(vals) - 1
                    monthly_rate = (recent / older) ** (1 / months_span) - 1
                    inflation_rate = round(monthly_rate * 12 * 100, 1)

        # Trends
        fed_trend = _fred_trend(results.get("fed_rate"))
        inflation_trend = _fred_trend(results.get("cpi"))
        fed_direction = _infer_fed_direction(fed_rate, fed_trend)

        # Yield curve
        yield_curve = _infer_yield_curve(t10y, t2y, t3mo)
        yield_spread = round(t10y - t2y, 2) if t10y is not None and t2y is not None else None

        # Sector signals
        sector_signals = _compute_sector_signals(fed_direction, inflation_trend, yield_curve)

        # Market regime
        market_regime = _infer_market_regime(
            gdp_growth, unemployment, inflation_trend, yield_curve,
        )

        # Build summary
        parts: list[str] = []
        if fed_rate is not None:
            parts.append(f"The Fed Funds Rate is at {fed_rate}% ({fed_direction}).")
        if inflation_rate is not None:
            parts.append(
                f"Inflation is approximately {inflation_rate}% annualised ({inflation_trend})."
            )
        if yield_curve != "unknown":
            spread_str = f" ({yield_spread:+.2f}%)" if yield_spread is not None else ""
            parts.append(f"The yield curve is {yield_curve}{spread_str}.")
        if gdp_growth is not None:
            parts.append(f"GDP growth is {gdp_growth}%.")
        if unemployment is not None:
            parts.append(f"Unemployment is {unemployment}%.")
        parts.append(f"Market regime: {market_regime.replace('_', ' ')}.")

        summary = " ".join(parts) if parts else "Macro data partially unavailable."

        # If all key values are None, something went wrong
        if all(v is None for v in [fed_rate, inflation_rate, gdp_growth, unemployment]):
            logger.warning("All FRED series returned None — macro data unavailable")
            return {
                "data_source": "fred",
                "error": "All FRED series returned no data.",
                "summary": "Macro environment data unavailable — all FRED API requests returned no data.",
            }

        return {
            "data_source": "fred",
            "fed_rate": fed_rate,
            "fed_direction": fed_direction,
            "inflation_rate": inflation_rate,
            "inflation_trend": inflation_trend,
            "yield_curve": yield_curve,
            "gdp_growth": gdp_growth,
            "unemployment": unemployment,
            "treasury_10y": t10y,
            "treasury_2y": t2y,
            "treasury_3mo": t3mo,
            "yield_spread_10y_2y": yield_spread,
            "market_regime": market_regime,
            "sector_signals": sector_signals,
            "summary": summary,
        }

    return await async_get_or_set(macro_cache, "macro:current", _fetch)
