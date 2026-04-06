"""PydanticAI Agent definition — the central orchestrator.

Registers all tools, wires up dependencies, and exposes `run_suggest`
and `run_analyze` entry-points consumed by the FastAPI routes.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
    compute_recommendation_range,
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

logger = logging.getLogger(__name__)

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
    market_cap_range: str = "auto",
    min_avg_daily_volume: int = 500_000,
    exchanges: list[str] | None = None,
    max_pe_ratio: float = 50.0,
    indices: list[str] | None = None,
    max_results: int = 25,
) -> dict:
    """Screen the stock universe and return 10-30 filtered candidates. Call this FIRST.

    Filters by market cap range (mega/large/mid/small/large_and_above/mid_and_above/all/auto),
    sector, exchange (NYSE/NASDAQ), minimum average daily volume (liquidity),
    and maximum P/E ratio (exclude extreme valuations).
    indices can be ["sp500"], ["nasdaq100"], ["sp400"], ["sp600"], or ["russell2000"].

    When market_cap_range is "auto" (default), it is determined by the user's
    risk tolerance:
      • conservative  → large_and_above (≥$10B)
      • moderate      → mid_and_above   (≥$2B)
      • aggressive    → all             (≥$300M, includes small-cap)
    """
    deps = ctx.deps

    # ── Auto-adjust market_cap_range based on risk tolerance ─────────
    _RISK_TO_CAP: dict[str, str] = {
        "conservative": "large_and_above",
        "moderate":     "mid_and_above",
        "aggressive":   "all",
    }
    effective_risk = risk_tolerance or deps.risk_tolerance.value
    if market_cap_range == "auto":
        market_cap_range = _RISK_TO_CAP.get(effective_risk, "mid_and_above")

    result = await screen_stocks(
        risk_tolerance=effective_risk,
        sector_preferences=sector_preferences or deps.sector_preferences or None,
        excluded_tickers=excluded_tickers or deps.excluded_tickers or None,
        market_cap_range=market_cap_range,
        min_avg_daily_volume=min_avg_daily_volume,
        exchanges=exchanges,
        max_pe_ratio=max_pe_ratio,
        indices=indices,
        max_results=max_results,
        data_source=deps.data_source,
        market_cap_min_override=deps.market_cap_min,
        market_cap_max_override=deps.market_cap_max,
    )

    # Track which tickers the screener returned so post-processing can
    # detect hallucinated symbols the LLM might invent.
    for candidate in result.get("candidates", []):
        deps.screened_tickers.add(candidate["ticker"])

    return result


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
        "min_recommendations": deps.min_recommendations,
        "max_recommendations": deps.max_recommendations,
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

async def _fetch_real_key_metrics(ticker: str) -> KeyMetrics:
    """Fetch real key metrics for a ticker from deterministic tools.

    This is called AFTER the LLM response to overwrite any hallucinated
    metric values with actual data from yfinance / SEC EDGAR.
    """
    fundamentals: dict = {}
    technicals: dict = {}

    # Fetch fundamentals and technicals in parallel
    fund_task = asyncio.create_task(_safe_fetch(get_fundamental_metrics, ticker))
    tech_task = asyncio.create_task(_safe_fetch(get_technical_indicators, ticker))
    fundamentals, technicals = await asyncio.gather(fund_task, tech_task)

    return KeyMetrics(
        pe_ratio=fundamentals.get("pe_ratio"),
        trailing_pe=fundamentals.get("trailing_pe"),
        forward_pe=fundamentals.get("forward_pe"),
        pb_ratio=fundamentals.get("pb_ratio"),
        peg_ratio=fundamentals.get("peg_ratio"),
        roe=fundamentals.get("roe"),
        debt_to_equity=fundamentals.get("debt_to_equity"),
        free_cash_flow=fundamentals.get("free_cash_flow"),
        earnings_per_share=fundamentals.get("earnings_per_share"),
        trailing_eps=fundamentals.get("trailing_eps"),
        forward_eps=fundamentals.get("forward_eps"),
        rsi=technicals.get("rsi"),
        macd_signal=technicals.get("macd_signal"),
        bollinger_position=technicals.get("bollinger_position"),
        sma_trend=technicals.get("sma_trend"),
    )


async def _safe_fetch(fn, ticker: str) -> dict:
    """Call an async tool function, returning {} on failure."""
    try:
        return await fn(ticker)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Post-processing fetch failed for %s: %s", ticker, exc)
        return {}


async def _backfill_suggest_metrics(
    response: TradeRecommendationResponse,
) -> TradeRecommendationResponse:
    """Replace LLM-generated key_metrics with real tool data for every recommendation."""
    if not response.recommendations:
        return response

    # Fetch real metrics for all tickers in parallel
    tasks = [
        _fetch_real_key_metrics(rec.ticker)
        for rec in response.recommendations
    ]
    real_metrics_list = await asyncio.gather(*tasks)

    # Overwrite each recommendation's key_metrics
    for rec, real_metrics in zip(response.recommendations, real_metrics_list):
        rec.key_metrics = real_metrics

    return response


async def _backfill_analysis_metrics(
    response: AnalysisResponse,
) -> AnalysisResponse:
    """Replace LLM-generated key_metrics with real tool data for an analysis."""
    response.key_metrics = await _fetch_real_key_metrics(response.ticker)

    # Overwrite company_name with real data to prevent hallucination
    try:
        company_info = await get_company_info(response.ticker)
        real_name = company_info.get("name")
        if real_name:
            response.company_name = real_name
    except Exception:
        logger.warning("Could not backfill company name for %s", response.ticker)

    return response


async def _enforce_market_cap_bounds(
    response: TradeRecommendationResponse,
    market_cap_min: float | None,
    market_cap_max: float | None,
) -> TradeRecommendationResponse:
    """Remove recommendations whose real market cap falls outside the user's bounds.

    Fetches live market cap for each ticker via get_stock_price and drops
    any that violate the constraints.  Redistributes freed capital
    proportionally among the remaining recommendations.
    """
    if not response.recommendations:
        return response

    # Fetch live market cap for each recommendation in parallel
    tasks = [
        _safe_fetch(get_stock_price, rec.ticker)
        for rec in response.recommendations
    ]
    price_data_list = await asyncio.gather(*tasks)

    kept: list[StockRecommendation] = []
    removed_tickers: list[str] = []
    for rec, price_data in zip(response.recommendations, price_data_list):
        mc = price_data.get("market_cap")
        if mc is None:
            # Cannot verify — drop to be safe when user set explicit bounds
            removed_tickers.append(rec.ticker)
            continue
        if market_cap_min is not None and mc < market_cap_min:
            removed_tickers.append(rec.ticker)
            continue
        if market_cap_max is not None and mc > market_cap_max:
            removed_tickers.append(rec.ticker)
            continue
        kept.append(rec)

    if removed_tickers:
        logger.info(
            "Market cap filter removed %d recommendation(s): %s",
            len(removed_tickers),
            ", ".join(removed_tickers),
        )

    if not kept:
        # All recommendations removed — return empty with full cash reserve
        response.recommendations = []
        response.cash_reserve_usd = response.cash_reserve_usd + sum(
            r.allocation_usd for r in response.recommendations
        )
        response.cash_reserve_pct = 100.0
        response.risk_warnings.append(
            "No stocks matched the requested market cap range. "
            "All funds are held as cash."
        )
        return response

    # Redistribute allocations proportionally among kept recommendations
    if len(kept) < len(response.recommendations):
        total_available = sum(r.allocation_usd for r in kept) + sum(
            r.allocation_usd
            for r, _ in zip(response.recommendations, price_data_list)
            if r not in kept
        )
        total_funds = total_available + response.cash_reserve_usd
        weight_sum = sum(r.confidence for r in kept)
        for rec in kept:
            rec.allocation_usd = round(
                total_available * (rec.confidence / weight_sum), 2
            )
            rec.allocation_pct = round(rec.allocation_usd / total_funds * 100, 2)

    response.recommendations = kept
    return response


async def run_suggest(
    funds: float,
    horizon: InvestmentHorizon,
    risk_tolerance: RiskTolerance,
    sector_preferences: list[str] | None = None,
    excluded_tickers: list[str] | None = None,
    dividend_investing: bool = True,
    data_source: str = "auto",
    market_cap_min_billions: float | None = None,
    market_cap_max_billions: float | None = None,
) -> TradeRecommendationResponse:
    """Run the full suggestion pipeline through the PydanticAI agent."""
    # Compute dynamic recommendation count
    min_recs, max_recs = compute_recommendation_range(funds, horizon, risk_tolerance)

    # Build dependencies
    strategies = select_strategies(horizon, risk_tolerance, dividend_investing=dividend_investing)
    deps = TraderDeps(
        funds=funds,
        horizon=horizon,
        risk_tolerance=risk_tolerance,
        sector_preferences=sector_preferences or [],
        excluded_tickers=excluded_tickers or [],
        selected_strategies=[s.name for s in strategies],
        kelly_mode=get_kelly_mode(horizon, risk_tolerance),
        cash_reserve_pct=get_cash_reserve_pct(horizon, risk_tolerance),
        data_source=data_source,
        dividend_investing=dividend_investing,
        min_recommendations=min_recs,
        max_recommendations=max_recs,
        market_cap_min=market_cap_min_billions * 1e9 if market_cap_min_billions is not None else None,
        market_cap_max=market_cap_max_billions * 1e9 if market_cap_max_billions is not None else None,
    )

    # Build system prompt with strategy context and dynamic sizing
    strategy_descs = [f"{s.name}: {s.description}" for s in strategies]
    horizon_label = horizon.value.replace('_', ' ')
    system_prompt = build_system_prompt(
        strategy_descs,
        min_recs=min_recs,
        max_recs=max_recs,
        funds=funds,
        horizon=horizon_label,
        risk_tolerance=risk_tolerance.value,
        dividend_investing=dividend_investing,
    )

    user_prompt = (
        f"I have ${funds:,.0f} to invest with a {horizon_label} horizon "
        f"and {risk_tolerance.value} risk tolerance. "
    )
    if sector_preferences:
        user_prompt += f"I prefer these sectors: {', '.join(sector_preferences)}. "
    if excluded_tickers:
        user_prompt += f"Exclude these tickers: {', '.join(excluded_tickers)}. "
    if not dividend_investing:
        user_prompt += (
            "I do NOT want dividend-focused stocks. Avoid recommending stocks "
            "primarily because of their dividend yield. Prefer high-growth "
            "companies over dividend payers. "
        )
    if market_cap_min_billions is not None or market_cap_max_billions is not None:
        cap_parts: list[str] = []
        if market_cap_min_billions is not None:
            cap_parts.append(f"at least ${market_cap_min_billions:.1f}B")
        if market_cap_max_billions is not None:
            cap_parts.append(f"at most ${market_cap_max_billions:.1f}B")
        user_prompt += (
            f"IMPORTANT: Only recommend stocks with a market capitalisation "
            f"{' and '.join(cap_parts)}. "
            f"Do NOT recommend any stock outside this market cap range. "
        )
    user_prompt += (
        f"Please screen for candidates, analyse them, check macro conditions, "
        f"and provide between {min_recs} and {max_recs} specific stock "
        f"recommendations with dollar allocations."
    )

    result = await trader_agent.run(
        user_prompt,
        deps=deps,
        instructions=system_prompt,
    )
    # Post-process: drop any tickers the LLM hallucinated (not from screener)
    output = result.output
    if deps.screened_tickers and output.recommendations:
        valid = []
        hallucinated = []
        for rec in output.recommendations:
            if rec.ticker.upper() in deps.screened_tickers:
                valid.append(rec)
            else:
                hallucinated.append(rec.ticker)
        if hallucinated:
            logger.warning(
                "Removed %d hallucinated ticker(s) not from screener: %s",
                len(hallucinated), ", ".join(hallucinated),
            )
            output.recommendations = valid

    # Post-process: replace LLM-generated key_metrics with real tool data
    output = await _backfill_suggest_metrics(output)

    # Post-process: enforce market cap bounds using real data
    if market_cap_min_billions is not None or market_cap_max_billions is not None:
        output = await _enforce_market_cap_bounds(
            output,
            market_cap_min=market_cap_min_billions * 1e9 if market_cap_min_billions is not None else None,
            market_cap_max=market_cap_max_billions * 1e9 if market_cap_max_billions is not None else None,
        )

    return output


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

    # Pre-fetch company info so the LLM cannot hallucinate the company identity
    try:
        company_info = await get_company_info(ticker)
        company_context = (
            f"Verified company information for {ticker.upper()}:\n"
            f"  Company Name: {company_info.get('name', ticker.upper())}\n"
            f"  Sector: {company_info.get('sector', 'Unknown')}\n"
            f"  Industry: {company_info.get('industry', 'Unknown')}\n"
            f"  Description: {company_info.get('description', 'N/A')}\n\n"
        )
    except Exception:
        logger.warning("Could not pre-fetch company info for %s", ticker)
        company_context = ""

    user_prompt = (
        f"{company_context}"
        f"Perform a comprehensive analysis of {ticker.upper()} for a "
        f"{horizon.value.replace('_', ' ')} investment horizon. "
        "Include fundamental analysis, technical analysis, and sentiment analysis. "
        "Provide an overall rating and key risk warnings."
    )

    result = await analysis_agent.run(user_prompt, deps=deps)
    # Post-process: replace LLM-generated key_metrics with real tool data
    return await _backfill_analysis_metrics(result.output)
