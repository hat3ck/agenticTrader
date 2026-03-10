"""Unit tests for market data, fundamentals, technicals, sentiment, screener tools."""

import pytest
from unittest.mock import AsyncMock, patch
from app.tools.market_data import get_stock_price, get_historical_data, get_company_info
from app.tools.fundamentals import get_fundamental_metrics
from app.tools.technicals import get_technical_indicators
from app.tools.sentiment import get_news_sentiment, get_macro_environment
from app.tools.screener import screen_stocks
from app.tools.portfolio import kelly_fraction, optimize_portfolio
from app.data.cache import clear_all_caches
from app.data.constituents import get_dynamic_universe, get_universe


@pytest.fixture(autouse=True)
def _clear_caches():
    clear_all_caches()
    yield
    clear_all_caches()


# ── Market Data ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_stock_price():
    result = await get_stock_price("AAPL")
    assert result["ticker"] == "AAPL"
    assert result["price"] > 0
    assert "market_cap" in result


@pytest.mark.asyncio
async def test_get_historical_data():
    result = await get_historical_data("MSFT", "3mo")
    assert result["ticker"] == "MSFT"
    assert result["data_points"] > 0
    assert len(result["sample_data"]) > 0


@pytest.mark.asyncio
async def test_get_company_info():
    result = await get_company_info("GOOGL")
    assert result["ticker"] == "GOOGL"
    assert result["name"] == "Alphabet Inc."
    assert result["sector"]  # non-empty string from yfinance


# ── Fundamentals ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_fundamentals():
    result = await get_fundamental_metrics("NVDA")
    assert result["ticker"] == "NVDA"
    # Core fields must be present (may be None if data unavailable)
    for key in ("pe_ratio", "pb_ratio", "peg_ratio", "debt_to_equity",
                "free_cash_flow", "roe", "dividend_yield", "revenue_growth",
                "earnings_per_share", "profit_margin", "payout_ratio",
                "market_cap", "valuation_summary", "health_summary",
                "quality_summary"):
        assert key in result, f"Missing key: {key}"
    # At least some metrics should be populated for a major stock
    populated = sum(1 for k in ("pe_ratio", "roe", "market_cap") if result[k] is not None)
    assert populated >= 1, "Expected at least one core metric to be non-None for NVDA"


@pytest.mark.asyncio
async def test_get_fundamentals_summaries():
    result = await get_fundamental_metrics("AAPL")
    assert isinstance(result["valuation_summary"], str)
    assert isinstance(result["health_summary"], str)
    assert isinstance(result["quality_summary"], str)


# ── Technicals ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_technicals():
    result = await get_technical_indicators("AAPL")
    assert result["ticker"] == "AAPL"
    assert "rsi" in result
    assert result["rsi_signal"] in ("overbought", "oversold", "neutral")
    assert result["macd_signal"] in ("bullish", "bearish", "neutral")
    assert result["sma_trend"] in ("golden_cross", "death_cross", "neutral")


# ── Sentiment ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_news_sentiment():
    result = await get_news_sentiment("AAPL")
    assert result["ticker"] == "AAPL"
    assert result["overall_sentiment"] in ("positive", "negative", "neutral")
    assert -1 <= result["sentiment_score"] <= 1


@pytest.mark.asyncio
async def test_get_macro_environment():
    result = await get_macro_environment()
    assert "fed_rate" in result
    assert "summary" in result
    assert "sector_signals" in result


# ── Screener ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_screen_stocks():
    result = await screen_stocks(risk_tolerance="moderate", max_results=10)
    assert result["candidates_returned"] <= 10
    assert len(result["candidates"]) > 0
    assert "ticker" in result["candidates"][0]
    # New fields from yfinance enrichment
    candidate = result["candidates"][0]
    assert "sector" in candidate
    assert "exchange" in candidate


@pytest.mark.asyncio
async def test_screen_stocks_with_exclusions():
    result = await screen_stocks(excluded_tickers=["AAPL", "MSFT"])
    tickers = [c["ticker"] for c in result["candidates"]]
    assert "AAPL" not in tickers
    assert "MSFT" not in tickers


@pytest.mark.asyncio
async def test_screen_stocks_sector_filter():
    result = await screen_stocks(
        sector_preferences=["Healthcare"],
        max_results=15,
    )
    assert result["candidates_returned"] > 0
    # Preferred sector should appear in top results
    top_sectors = [c["sector"] for c in result["candidates"][:5]]
    assert "Healthcare" in top_sectors


@pytest.mark.asyncio
async def test_screen_stocks_market_cap_range():
    result = await screen_stocks(market_cap_range="mega", max_results=10)
    assert result["candidates_returned"] > 0
    assert "cap_range=mega" in result["filters_applied"]


@pytest.mark.asyncio
async def test_screen_stocks_exchange_filter():
    result = await screen_stocks(exchanges=["NASDAQ"], max_results=10)
    assert result["candidates_returned"] > 0
    for c in result["candidates"]:
        assert c["exchange"] == "NASDAQ"


@pytest.mark.asyncio
async def test_screen_stocks_index_filter():
    result = await screen_stocks(indices=["nasdaq100"], max_results=10)
    assert result["candidates_returned"] > 0
    assert "indices=['nasdaq100']" in result["filters_applied"]


@pytest.mark.asyncio
async def test_screen_stocks_conservative():
    result = await screen_stocks(risk_tolerance="conservative", max_results=10)
    assert result["candidates_returned"] > 0
    # Conservative should sort by market cap descending
    caps = [c["market_cap"] for c in result["candidates"] if c.get("market_cap")]
    if len(caps) >= 2:
        assert caps[0] >= caps[1], "Conservative should favour largest caps first"


# ── Dynamic Universe (FMP + fallback) ────────────────────

@pytest.mark.asyncio
async def test_dynamic_universe_fallback():
    """Without FMP_API_KEY, get_dynamic_universe falls back to hardcoded list."""
    clear_all_caches()
    universe = await get_dynamic_universe()
    # Should return the hardcoded list (since no FMP key is set in tests)
    assert len(universe) > 50
    tickers = {s["ticker"] for s in universe}
    assert "AAPL" in tickers
    assert "MSFT" in tickers


@pytest.mark.asyncio
async def test_dynamic_universe_fmp_success():
    """When FMP returns data, get_dynamic_universe uses it instead of hardcoded."""
    clear_all_caches()
    fake_universe = [
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "exchange": "NASDAQ", "cap_tier": "mega", "indices": ["sp500", "nasdaq100"]},
        {"ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Technology", "exchange": "NASDAQ", "cap_tier": "mega", "indices": ["sp500", "nasdaq100"]},
        {"ticker": "NEWSTOCK", "name": "New Stock Co.", "sector": "Healthcare", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
        {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Cyclical", "exchange": "NASDAQ", "cap_tier": "mega", "indices": ["sp500", "nasdaq100"]},
    ]

    with patch("app.data.constituents._fetch_fmp_universe", return_value=fake_universe):
        universe = await get_dynamic_universe()

    tickers = {s["ticker"] for s in universe}
    assert "NEWSTOCK" in tickers  # FMP-only stock present
    assert "TSLA" in tickers
    # AAPL should be present
    aapl = next(s for s in universe if s["ticker"] == "AAPL")
    assert "sp500" in aapl["indices"]
    assert "nasdaq100" in aapl["indices"]


@pytest.mark.asyncio
async def test_dynamic_universe_fmp_failure_falls_back():
    """When FMP API fails and Wikipedia also fails, falls back to hardcoded list."""
    clear_all_caches()
    with patch("app.data.constituents._fetch_fmp_universe", return_value=None):
        with patch("app.data.constituents._fetch_wikipedia_universe", return_value=None):
            universe = await get_dynamic_universe()

    # Should have fallen back to hardcoded
    assert len(universe) > 50
    tickers = {s["ticker"] for s in universe}
    assert "AAPL" in tickers


@pytest.mark.asyncio
async def test_dynamic_universe_wikipedia_fallback():
    """When FMP fails, Wikipedia is used as the next fallback."""
    clear_all_caches()
    fake_wiki_universe = [
        {"ticker": "WIKI1", "name": "Wiki Stock 1", "sector": "Technology", "exchange": "NASDAQ", "cap_tier": "large", "indices": ["sp500"]},
        {"ticker": "WIKI2", "name": "Wiki Stock 2", "sector": "Healthcare", "exchange": "NYSE", "cap_tier": "large", "indices": ["sp500"]},
    ]

    with patch("app.data.constituents._fetch_fmp_universe", return_value=None):
        with patch("app.data.constituents._fetch_wikipedia_universe", return_value=fake_wiki_universe):
            universe = await get_dynamic_universe()

    tickers = {s["ticker"] for s in universe}
    assert "WIKI1" in tickers
    assert "WIKI2" in tickers


def test_get_universe_sync():
    """get_universe() always returns the hardcoded list (sync)."""
    universe = get_universe()
    assert len(universe) > 50
    assert all("ticker" in s for s in universe)


# ── Portfolio / Kelly ────────────────────────────────────

def test_kelly_fraction_positive():
    f = kelly_fraction(win_prob=0.6, win_loss_ratio=2.0, mode="full")
    assert f > 0
    assert f <= 1.0


def test_kelly_fraction_negative_edge():
    f = kelly_fraction(win_prob=0.2, win_loss_ratio=0.5, mode="full")
    assert f == 0.0  # negative edge → no bet


def test_kelly_half():
    full = kelly_fraction(0.6, 2.0, "full")
    half = kelly_fraction(0.6, 2.0, "half")
    assert half == pytest.approx(full / 2, abs=0.001)


@pytest.mark.asyncio
async def test_optimize_portfolio():
    candidates = [
        {"ticker": "AAPL", "confidence": 0.7, "expected_return": 0.12, "expected_loss": 0.06, "sector": "Technology"},
        {"ticker": "JPM", "confidence": 0.6, "expected_return": 0.08, "expected_loss": 0.05, "sector": "Financial"},
        {"ticker": "JNJ", "confidence": 0.65, "expected_return": 0.07, "expected_loss": 0.04, "sector": "Healthcare"},
    ]
    result = await optimize_portfolio(candidates, total_funds=10000, kelly_mode="half", cash_reserve_pct=10)
    assert result["cash_reserve_usd"] == 1000.0
    assert result["total_invested"] > 0
    assert len(result["allocations"]) == 3
    total = sum(a["allocation_usd"] for a in result["allocations"]) + result["cash_reserve_usd"]
    assert total == pytest.approx(10000, abs=1.0)
