"""PydanticAI Agent definition — the central orchestrator.

Registers all tools, wires up dependencies, and exposes `run_suggest`
and `run_analyze` entry-points consumed by the FastAPI routes.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic_ai import Agent, RunContext

from app.agent.deps import TraderDeps
from app.agent.prompts import ANALYSIS_SYSTEM_PROMPT, build_system_prompt
from app.config import settings
from app.models.requests import InvestmentHorizon, RiskTolerance
from app.models.responses import (
    AnalysisResponse,
    KeyMetrics,
    StockRecommendation,
    TradeRecommendationResponse,
)
from app.strategies.registry import (
    get_cash_reserve_pct,
    get_kelly_mode,
    select_strategies,
)
from app.tools.fundamentals import get_fundamental_metrics
from app.tools.market_data import get_company_info, get_historical_data, get_stock_price
from app.tools.portfolio import optimize_portfolio
from app.tools.screener import screen_stocks
from app.tools.sentiment import get_macro_environment, get_news_sentiment
from app.tools.technicals import get_technical_indicators

_model_name = settings.llm_model

# Auto-prefix bare model names with "openai:" (needed for local endpoints)
if ":" not in _model_name:
    _model_name = f"openai:{_model_name}"

if settings.llm_base_url:
    # Local compatible endpoint
    os.environ.setdefault("OPENAI_BASE_URL", settings.llm_base_url)

trader_agent = Agent(
    model=_model_name,
    deps_type=TraderDeps,
    output_type=TradeRecommendationResponse,
    system_prompt="You are an expert financial analyst.",  # overridden at run time
    retries=3,
)

analysis_agent = Agent(
    model=_model_name,
    deps_type=TraderDeps,
    output_type=AnalysisResponse,
    system_prompt=ANALYSIS_SYSTEM_PROMPT,
    retries=3,
)

@trader_agent.tool
@analysis_agent.tool
async def tool_get_stock_price(ctx: RunContext[TraderDeps], ticker: str) -> dict:
    """Get current price, volume, market cap, and 52-week range for a stock."""
    return await get_stock_price(ticker)


@trader_agent.tool
@analysis_agent.tool
async def tool_get_historical_data(ctx: RunContext[TraderDeps], ticker: str, period: str = "6mo") -> dict:
    """Get historical OHLCV data. period: '1mo','3mo','6mo','1y','5y'."""
    return await get_historical_data(ticker, period)


@trader_agent.tool
@analysis_agent.tool
async def tool_get_company_info(ctx: RunContext[TraderDeps], ticker: str) -> dict:
    """Get company name, sector, industry, and description."""
    return await get_company_info(ticker)


@trader_agent.tool
@analysis_agent.tool
async def tool_get_fundamentals(ctx: RunContext[TraderDeps], ticker: str) -> dict:
    """Get fundamental metrics: P/E, P/B, PEG, D/E, FCF, ROE, dividend yield, revenue growth."""
    return await get_fundamental_metrics(ticker)


@trader_agent.tool
@analysis_agent.tool
async def tool_get_technicals(ctx: RunContext[TraderDeps], ticker: str) -> dict:
    """Get technical indicators: RSI, MACD, Bollinger Bands, SMA/EMA, ATR, volume profile."""
    return await get_technical_indicators(ticker)


@trader_agent.tool
@analysis_agent.tool
async def tool_get_news_sentiment(ctx: RunContext[TraderDeps], ticker: str) -> dict:
    """Get news sentiment summary: headlines, overall sentiment, sentiment score."""
    return await get_news_sentiment(ticker)


@trader_agent.tool
@analysis_agent.tool
async def tool_get_macro_environment(ctx: RunContext[TraderDeps]) -> dict:
    """Get macroeconomic snapshot: Fed rate, inflation, yield curve, sector signals."""
    return await get_macro_environment()


@trader_agent.tool
async def tool_screen_stocks(
    ctx: RunContext[TraderDeps],
    risk_tolerance: str = "moderate",
    sector_preferences: list[str] | None = None,
    excluded_tickers: list[str] | None = None,
) -> dict:
    """Screen stock universe to find 10-30 candidates matching criteria. Call this FIRST."""
    deps = ctx.deps
    return await screen_stocks(
        risk_tolerance=risk_tolerance or deps.risk_tolerance.value,
        sector_preferences=sector_preferences or deps.sector_preferences or None,
        excluded_tickers=excluded_tickers or deps.excluded_tickers or None,
    )


@trader_agent.tool
async def tool_optimize_portfolio(
    ctx: RunContext[TraderDeps],
    candidates: list[dict[str, Any]],
) -> dict:
    """Allocate capital across selected stocks using Kelly Criterion.

    Each candidate dict needs: ticker (str), confidence (float 0-1),
    expected_return (float), expected_loss (float), sector (str).
    """
    deps = ctx.deps
    return await optimize_portfolio(
        candidates=candidates,
        total_funds=deps.funds,
        kelly_mode=deps.kelly_mode,
        cash_reserve_pct=deps.cash_reserve_pct,
    )


@trader_agent.tool
@analysis_agent.tool
async def tool_get_strategy_context(ctx: RunContext[TraderDeps]) -> dict:
    """Get the selected trading strategies and their rules for this request."""
    deps = ctx.deps
    strategies = select_strategies(deps.horizon, deps.risk_tolerance)
    return {
        "selected_strategies": [s.name for s in strategies],
        "horizon": deps.horizon.value,
        "risk_tolerance": deps.risk_tolerance.value,
        "kelly_mode": deps.kelly_mode,
        "cash_reserve_pct": deps.cash_reserve_pct,
        "funds": deps.funds,
        "strategy_details": [
            {
                "name": s.name,
                "description": s.description,
                "fundamental_weight": s.fundamental_weight,
                "technical_weight": s.technical_weight,
                "sentiment_weight": s.sentiment_weight,
                "key_metrics": s.key_metrics,
                "thresholds": [
                    {"metric": t.metric, "operator": t.favorable_operator,
                     "value": t.favorable_value, "description": t.description}
                    for t in s.thresholds
                ],
            }
            for s in strategies
        ],
    }

async def run_suggest(
    funds: float,
    horizon: InvestmentHorizon,
    risk_tolerance: RiskTolerance,
    sector_preferences: list[str] | None = None,
    excluded_tickers: list[str] | None = None,
) -> TradeRecommendationResponse:
    """Run the full suggestion pipeline through the PydanticAI agent."""
    # Build dependencies
    strategies = select_strategies(horizon, risk_tolerance)
    deps = TraderDeps(
        funds=funds,
        horizon=horizon,
        risk_tolerance=risk_tolerance,
        sector_preferences=sector_preferences or [],
        excluded_tickers=excluded_tickers or [],
        selected_strategies=[s.name for s in strategies],
        kelly_mode=get_kelly_mode(horizon, risk_tolerance),
        cash_reserve_pct=get_cash_reserve_pct(horizon, risk_tolerance),
    )

    # Build system prompt with strategy context
    strategy_descs = [f"{s.name}: {s.description}" for s in strategies]
    system_prompt = build_system_prompt(strategy_descs)

    user_prompt = (
        f"I have ${funds:,.0f} to invest with a {horizon.value.replace('_', ' ')} horizon "
        f"and {risk_tolerance.value} risk tolerance. "
    )
    if sector_preferences:
        user_prompt += f"I prefer these sectors: {', '.join(sector_preferences)}. "
    if excluded_tickers:
        user_prompt += f"Exclude these tickers: {', '.join(excluded_tickers)}. "
    user_prompt += (
        "Please screen for candidates, analyse them, check macro conditions, "
        "and provide specific stock recommendations with dollar allocations."
    )

    result = await trader_agent.run(
        user_prompt,
        deps=deps,
        instructions=system_prompt,
    )
    return result.output


async def run_analyze(
    ticker: str,
    horizon: InvestmentHorizon = InvestmentHorizon.THREE_MONTHS,
) -> AnalysisResponse:
    """Run a deep-dive analysis on a single ticker."""
    deps = TraderDeps(
        funds=0,
        horizon=horizon,
        risk_tolerance=RiskTolerance.MODERATE,
        selected_strategies=[s.name for s in select_strategies(horizon)],
        kelly_mode=get_kelly_mode(horizon),
        cash_reserve_pct=get_cash_reserve_pct(horizon),
    )

    user_prompt = (
        f"Perform a comprehensive analysis of {ticker.upper()} for a "
        f"{horizon.value.replace('_', ' ')} investment horizon. "
        "Include fundamental analysis, technical analysis, and sentiment analysis. "
        "Provide an overall rating and key risk warnings."
    )

    result = await analysis_agent.run(user_prompt, deps=deps)
    return result.output
