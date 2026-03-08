# Agentic Trader

AI-powered stock analysis and portfolio recommendation engine built with **PydanticAI** + **FastAPI**.

## Architecture

```
FastAPI Gateway  →  PydanticAI Agent (LLM Orchestrator)
                        │
        ┌───────┬───────┼───────┬──────────┐
        ▼       ▼       ▼       ▼          ▼
    Market   Fundmntl  Tech  Sentiment  Portfolio
    Data     Analysis  Anlys  Analysis  Optimizer
                        │
                Strategy Engine
                        │
                  Data Layer
              (Cache + SQLite)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Configure your LLM (edit .env)
#    Default: LM Studio at localhost:1234
#    Or set OPENAI_API_KEY / ANTHROPIC_API_KEY

# 3. Run the server
uvicorn app.main:app --reload --port 8000

# 4. Open API docs
open http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/suggest` | Get stock recommendations with $ allocations |
| `POST` | `/api/v1/analyze/{ticker}` | Deep-dive single stock analysis |
| `GET`  | `/api/v1/strategies` | List available trading strategies |
| `GET`  | `/api/v1/health` | Health check |
| `WS`   | `/ws/suggest` | Streaming recommendations |

## Example Request

```bash
curl -X POST http://localhost:8000/api/v1/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "funds": 10000,
    "horizon": "3_months",
    "risk_tolerance": "moderate"
  }'
```

## Example Response

```json
{
  "strategy_applied": "Momentum + Growth",
  "macro_context": "Fed holding rates, moderate inflation",
  "recommendations": [
    {
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "allocation_usd": 3000,
      "allocation_pct": 30.0,
      "confidence": 0.78,
      "rationale": "Strong momentum with RSI at 55...",
      "key_metrics": { "pe_ratio": 28.5, "rsi": 55.0 }
    }
  ],
  "cash_reserve_usd": 1500,
  "cash_reserve_pct": 15.0,
  "risk_warnings": ["Market volatility elevated"],
  "disclaimer": "This is AI-generated analysis..."
}
```

## Project Structure

```
app/
├── main.py                 # FastAPI app + routes
├── config.py               # Settings (Pydantic BaseSettings)
├── models/
│   ├── requests.py         # Input schemas
│   └── responses.py        # Output schemas
├── agent/
│   ├── trader.py           # PydanticAI Agent + tool registrations
│   ├── deps.py             # Agent dependencies
│   └── prompts.py          # System prompt templates
├── tools/
│   ├── market_data.py      # Prices, OHLCV (mock → yfinance)
│   ├── fundamentals.py     # P/E, ROE, FCF, etc. (mock → yfinance)
│   ├── technicals.py       # RSI, MACD, Bollinger (pandas-ta)
│   ├── sentiment.py        # News + macro (mock → News API + FRED)
│   ├── screener.py         # Stock universe filtering (mock)
│   └── portfolio.py        # Kelly Criterion + allocation optimizer
├── strategies/
│   ├── models.py           # Strategy definitions (Pydantic models)
│   └── registry.py         # Strategy selection logic
└── data/
    ├── cache.py            # TTLCache (→ Redis)
    └── storage.py          # SQLite via SQLAlchemy (→ Postgres)
```

## Key Design Decisions

- **PydanticAI** as both LLM abstraction and agent framework — tools are registered with `@agent.tool`, the agent decides which to call
- **Kelly Criterion** for mathematically optimal position sizing (defaults to Half-Kelly)
- **Codified strategies** as Pydantic models, not LLM prompts — deterministic and auditable
- **The LLM reasons, the tools compute** — all financial math is in deterministic Python code
- **Mock data** for market data, fundamentals, sentiment, and screener — swap in real APIs later without changing agent logic

## Running Tests

```bash
pytest -v
```

## Switching LLM Providers

Edit `.env`:

```bash
# Local (LM Studio)
LLM_MODEL=openai:gemma-3
LLM_BASE_URL=http://localhost:1234/v1

# OpenAI
LLM_MODEL=openai:gpt-4o
OPENAI_API_KEY=sk-...

# Anthropic
LLM_MODEL=anthropic:claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-...
```

Zero code changes required.

## License

See [LICENSE](LICENSE).
