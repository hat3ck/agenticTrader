"""FastAPI application — REST + WebSocket API for Agentic Trader."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.agent.trader import run_analyze, run_suggest
from app.config import settings
from app.data.storage import init_db, save_recommendations
from app.models.requests import AnalyzeRequest, SuggestRequest
from app.models.responses import (
    AnalysisResponse,
    HealthResponse,
    StrategyInfo,
    TradeRecommendationResponse,
)
from app.strategies.registry import list_all_strategies

logger = logging.getLogger("agentic_trader")


# ── Lifespan ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic."""
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
    logger.info("Initialising database…")
    await init_db()
    logger.info("Agentic Trader started  ✔")
    yield
    logger.info("Agentic Trader shutting down…")


# ── App ──────────────────────────────────────────────────

app = FastAPI(
    title="Agentic Trader",
    description="AI-powered stock analysis and portfolio recommendation engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────

@app.post("/api/v1/suggest", response_model=TradeRecommendationResponse)
async def suggest(req: SuggestRequest):
    """Main endpoint: recommend stocks with dollar-denominated allocations.

    Provide available funds, investment horizon, and risk tolerance.
    The agent will screen stocks, analyse candidates, check macro
    conditions, and return optimised portfolio recommendations.
    """
    result = await run_suggest(
        funds=req.funds,
        horizon=req.horizon,
        risk_tolerance=req.risk_tolerance,
        sector_preferences=req.sector_preferences,
        excluded_tickers=req.excluded_tickers,
        data_source=req.data_source.value,
    )

    # Persist recommendations asynchronously (best-effort)
    try:
        records = [
            {
                "ticker": r.ticker,
                "allocation_usd": r.allocation_usd,
                "confidence": r.confidence,
                "strategy": result.strategy_applied,
                "rationale": r.rationale,
                "horizon": req.horizon.value,
            }
            for r in result.recommendations
        ]
        await save_recommendations(records)
    except Exception:
        logger.exception("Failed to persist recommendations")

    return result


@app.post("/api/v1/analyze/{ticker}", response_model=AnalysisResponse)
async def analyze(ticker: str, req: AnalyzeRequest | None = None):
    """Deep-dive analysis of a single stock.

    Returns fundamentals, technicals, sentiment, overall rating.
    """
    horizon = req.horizon if req else None
    from app.models.requests import InvestmentHorizon
    return await run_analyze(
        ticker=ticker,
        horizon=horizon or InvestmentHorizon.THREE_MONTHS,
    )


@app.get("/api/v1/strategies", response_model=list[StrategyInfo])
async def strategies():
    """List all available trading strategies with descriptions."""
    raw = list_all_strategies()
    return [
        StrategyInfo(
            name=s["name"],
            best_horizon=", ".join(s["best_horizons"]),
            key_metrics=s["key_metrics"],
            description=s["description"],
        )
        for s in raw
    ]


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    """Health check — LLM connectivity, cache, database status."""
    llm_ok = False
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            if settings.llm_base_url:
                resp = await client.get(f"{settings.llm_base_url}/models")
                llm_ok = resp.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        llm_connected=llm_ok,
        cache_status="in-memory (TTLCache)",
        db_status="sqlite",
    )


# ──────────────────────────────────────────────────────────
# WebSocket — Streaming Suggest
# ──────────────────────────────────────────────────────────

@app.websocket("/ws/suggest")
async def ws_suggest(ws: WebSocket):
    """Streaming version of /suggest — sends incremental updates.

    Client sends a JSON message matching SuggestRequest schema.
    Server streams status updates and the final recommendation.
    """
    await ws.accept()
    try:
        raw = await ws.receive_text()
        req = SuggestRequest.model_validate_json(raw)

        # Send progress updates
        await ws.send_json({"type": "status", "message": "Selecting strategies…"})

        await ws.send_json({"type": "status", "message": "Running agent analysis…"})

        result = await run_suggest(
            funds=req.funds,
            horizon=req.horizon,
            risk_tolerance=req.risk_tolerance,
            sector_preferences=req.sector_preferences,
            excluded_tickers=req.excluded_tickers,
            data_source=req.data_source.value,
        )

        await ws.send_json({"type": "status", "message": "Analysis complete."})
        await ws.send_json({
            "type": "result",
            "data": result.model_dump(),
        })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.exception("WebSocket error")
        await ws.send_json({"type": "error", "message": str(exc)})
    finally:
        try:
            await ws.close()
        except Exception:
            pass
