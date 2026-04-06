"""Integration tests for the FastAPI endpoints (no LLM required)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.requests import SuggestRequest, InvestmentHorizon

client = TestClient(app)


def test_health():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_strategies():
    resp = client.get("/api/v1/strategies")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 6
    assert all("name" in s for s in data)
    assert all("description" in s for s in data)


def test_suggest_request_market_cap_fields_optional():
    """market_cap_min_billions and market_cap_max_billions default to None."""
    req = SuggestRequest(funds=10000, horizon=InvestmentHorizon.THREE_MONTHS)
    assert req.market_cap_min_billions is None
    assert req.market_cap_max_billions is None


def test_suggest_request_market_cap_fields_set():
    """market_cap_min_billions and market_cap_max_billions can be set."""
    req = SuggestRequest(
        funds=10000,
        horizon=InvestmentHorizon.THREE_MONTHS,
        market_cap_min_billions=10.0,
        market_cap_max_billions=200.0,
    )
    assert req.market_cap_min_billions == 10.0
    assert req.market_cap_max_billions == 200.0


def test_suggest_request_market_cap_negative_rejected():
    """Negative market cap values should be rejected by the ge=0 validator."""
    with pytest.raises(Exception):
        SuggestRequest(
            funds=10000,
            horizon=InvestmentHorizon.THREE_MONTHS,
            market_cap_min_billions=-5.0,
        )


# Note: POST /api/v1/suggest and /api/v1/analyze/{ticker} require a running
# LLM backend, so they are skipped in unit tests.  Use the WebSocket or
# manual curl testing against a running server for integration tests.
