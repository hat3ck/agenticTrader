"""Integration tests for the FastAPI endpoints (no LLM required)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

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


# Note: POST /api/v1/suggest and /api/v1/analyze/{ticker} require a running
# LLM backend, so they are skipped in unit tests.  Use the WebSocket or
# manual curl testing against a running server for integration tests.
