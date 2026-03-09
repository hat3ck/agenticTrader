"""Unit tests for market data, fundamentals, technicals, sentiment, screener tools."""

import pytest
from app.tools.market_data import get_stock_price, get_historical_data, get_company_info
from app.tools.fundamentals import get_fundamental_metrics
from app.tools.technicals import get_technical_indicators
from app.tools.sentiment import get_news_sentiment, get_macro_environment
from app.tools.screener import screen_stocks
from app.tools.portfolio import kelly_fraction, optimize_portfolio
from app.data.cache import clear_all_caches


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
    assert "pe_ratio" in result
    assert "roe" in result
    assert "valuation_summary" in result


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


@pytest.mark.asyncio
async def test_screen_stocks_with_exclusions():
    result = await screen_stocks(excluded_tickers=["AAPL", "MSFT"])
    tickers = [c["ticker"] for c in result["candidates"]]
    assert "AAPL" not in tickers
    assert "MSFT" not in tickers


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
