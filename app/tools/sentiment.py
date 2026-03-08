"""Sentiment & Macro Analysis Tool.

Currently returns MOCK data.  Will be replaced with News API + FRED later.
"""

from __future__ import annotations

import random

from app.data.cache import sentiment_cache, macro_cache, async_get_or_set


async def get_news_sentiment(ticker: str) -> dict:
    """Get recent news headlines and sentiment for a ticker.

    Returns a dict with:
    - ticker: The stock symbol
    - overall_sentiment: 'positive', 'negative', or 'neutral'
    - sentiment_score: Float from -1 (very negative) to +1 (very positive)
    - headlines: List of recent headline dicts with title, sentiment, source, date
    - summary: Natural-language summary of current sentiment

    Use this to understand market perception and news-driven momentum.
    Positive sentiment with strong fundamentals = higher conviction.
    Negative sentiment on strong fundamentals = potential contrarian opportunity.
    """

    async def _fetch():
        # ── MOCK DATA — replace with News API later ──
        sentiments = {
            "AAPL": {"score": 0.6, "headlines": [
                {"title": "Apple Vision Pro sales exceed expectations", "sentiment": "positive", "source": "Reuters"},
                {"title": "Apple AI strategy gains traction with developers", "sentiment": "positive", "source": "Bloomberg"},
                {"title": "iPhone sales face headwinds in China", "sentiment": "negative", "source": "CNBC"},
            ]},
            "NVDA": {"score": 0.8, "headlines": [
                {"title": "NVIDIA reports record AI chip demand", "sentiment": "positive", "source": "Reuters"},
                {"title": "Data center revenue surges 200% YoY", "sentiment": "positive", "source": "Bloomberg"},
                {"title": "NVIDIA valuation concerns grow among analysts", "sentiment": "negative", "source": "WSJ"},
            ]},
            "MSFT": {"score": 0.5, "headlines": [
                {"title": "Microsoft Azure AI services drive cloud growth", "sentiment": "positive", "source": "CNBC"},
                {"title": "Copilot adoption accelerates across enterprise", "sentiment": "positive", "source": "Reuters"},
                {"title": "Antitrust scrutiny on Microsoft-OpenAI deal", "sentiment": "negative", "source": "FT"},
            ]},
        }

        if ticker.upper() in sentiments:
            data = sentiments[ticker.upper()]
        else:
            score = round(random.uniform(-0.5, 0.8), 2)
            data = {
                "score": score,
                "headlines": [
                    {"title": f"{ticker.upper()} reports quarterly earnings", "sentiment": "neutral", "source": "Reuters"},
                    {"title": f"Analysts upgrade {ticker.upper()} price target", "sentiment": "positive", "source": "Bloomberg"},
                ],
            }

        score = data["score"]
        sentiment = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"

        return {
            "ticker": ticker.upper(),
            "overall_sentiment": sentiment,
            "sentiment_score": score,
            "headlines": data["headlines"],
            "summary": f"Overall sentiment for {ticker.upper()} is {sentiment} "
                       f"(score: {score}). Recent news coverage is "
                       f"{'favorable' if score > 0 else 'mixed to negative'}.",
        }

    return await async_get_or_set(sentiment_cache, f"sent:{ticker.upper()}", _fetch)


async def get_macro_environment() -> dict:
    """Get current macroeconomic environment snapshot.

    Returns a dict with:
    - fed_rate: Current federal funds rate (percentage)
    - fed_direction: 'hiking', 'holding', or 'cutting'
    - inflation_rate: Current CPI inflation (percentage)
    - inflation_trend: 'rising', 'stable', or 'falling'
    - yield_curve: 'normal', 'flat', or 'inverted'
    - gdp_growth: GDP growth rate (percentage)
    - unemployment: Unemployment rate (percentage)
    - sector_signals: Dict mapping sectors to 'headwind' or 'tailwind'
    - summary: Natural-language macro summary

    Use this tool for macro context when making investment decisions.
    Rising rates = headwind for growth/tech, tailwind for financials.
    Inverted yield curve = recession signal.
    Low unemployment + high inflation = Fed likely to stay hawkish.
    """

    async def _fetch():
        # ── MOCK DATA — replace with FRED API later ──
        return {
            "fed_rate": 5.25,
            "fed_direction": "holding",
            "inflation_rate": 3.1,
            "inflation_trend": "falling",
            "yield_curve": "flat",
            "gdp_growth": 2.5,
            "unemployment": 3.8,
            "vix": 16.5,
            "market_regime": "moderate_bull",
            "sector_signals": {
                "Technology": "neutral",
                "Healthcare": "tailwind",
                "Financial Services": "tailwind",
                "Energy": "neutral",
                "Consumer Cyclical": "headwind",
                "Consumer Defensive": "tailwind",
                "Utilities": "tailwind",
                "Real Estate": "headwind",
            },
            "summary": (
                "The Fed is holding rates at 5.25% with inflation trending down to 3.1%. "
                "The yield curve is flat, suggesting economic uncertainty. GDP growth remains "
                "positive at 2.5% with low unemployment at 3.8%. Overall a moderate bullish "
                "environment favoring quality stocks and defensive sectors."
            ),
        }

    return await async_get_or_set(macro_cache, "macro:current", _fetch)
