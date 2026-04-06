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

### Market Cap Filtering

Narrow recommendations to a specific company-size range:

```bash
curl -X POST http://localhost:8000/api/v1/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "funds": 200000,
    "horizon": "1_year",
    "risk_tolerance": "aggressive",
    "market_cap_min_billions": 1.0,
    "market_cap_max_billions": 10.0
  }'
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `market_cap_min_billions` | `float \| null` | Minimum market cap (billions USD). `null` = no floor. |
| `market_cap_max_billions` | `float \| null` | Maximum market cap (billions USD). `null` = no ceiling. |

When set, these bounds override the risk-tolerance-based defaults
(e.g. moderate → ≥$2B). The screener supplements its standard universe
(S&P 500, NASDAQ-100, etc.) with a live Yahoo Finance equity screen to find
stocks in the requested range — no paid API key required.

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
│   ├── market_data.py      # Prices, OHLCV via yfinance
│   ├── fundamentals.py     # P/E, ROE, FCF, etc. via yfinance + SEC EDGAR
│   ├── technicals.py       # RSI, MACD, Bollinger (pandas-ta)
│   ├── sentiment.py        # News + macro (News API + FRED)
│   ├── screener.py         # Stock universe filtering + yfinance equity screener
│   └── portfolio.py        # Kelly Criterion + allocation optimizer
├── strategies/
│   ├── models.py           # Strategy definitions (Pydantic models)
│   └── registry.py         # Strategy selection logic
└── data/
    ├── cache.py            # TTLCache (→ Redis)
    ├── constituents.py     # Dynamic universe (FMP → Wikipedia → hardcoded)
    └── storage.py          # SQLite via SQLAlchemy (→ Postgres)
```

## Key Design Decisions

- **PydanticAI** as both LLM abstraction and agent framework — tools are registered with `@agent.tool`, the agent decides which to call
- **Kelly Criterion** for mathematically optimal position sizing (defaults to Half-Kelly)
- **Codified strategies** as Pydantic models, not LLM prompts — deterministic and auditable
- **The LLM reasons, the tools compute** — all financial math is in deterministic Python code
- **Multi-layer data sourcing** — FMP API → Wikipedia scraping → hardcoded fallback for the stock universe
- **Free Yahoo Finance equity screener** (`yf.screen()` + `EquityQuery`) for arbitrary market-cap ranges — no API key required
- **Anti-hallucination guardrails** — the agent can only recommend tickers returned by the screener tool; fabricated symbols are caught and removed in post-processing
- **Market cap enforcement pipeline** — user bounds flow through the system prompt (LLM guidance), screener filters (universe narrowing), and post-processing verification (live market-cap fetch via yfinance)

## Running Tests

```bash
pytest -v
```

## Switching LLM Providers

Edit `.env`:

```bash
# Local LM Studio (must use "openai:" prefix for OpenAI-compatible APIs)
LLM_MODEL=openai:google/gemma-3-27b
LLM_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio

# OpenAI
# LLM_MODEL=openai:gpt-4o
# OPENAI_API_KEY=sk-...

# Anthropic
# LLM_MODEL=anthropic:claude-sonnet-4-20250514
# ANTHROPIC_API_KEY=sk-ant-...
```

Zero code changes required.

## Environment Variables

Full `.env` reference:

```bash
# App settings
APP_ENV=development
LOG_LEVEL=INFO

# Cache TTLs (seconds)
MARKET_DATA_CACHE_TTL=900
FUNDAMENTAL_DATA_CACHE_TTL=86400
SENTIMENT_CACHE_TTL=3600
MACRO_CACHE_TTL=86400

# Sentiment & Macro APIs
NEWS_API_KEY=your-newsapi-key        # https://newsapi.org (free, 100 req/day)
FRED_API_KEY=your-fred-api-key       # https://fred.stlouisfed.org/docs/api/api_key.html (free)
FMP_API_KEY=your-fmp-api-key         # https://financialmodelingprep.com/developer/docs/pricing/ (optional, free tier)
```

> **Note:** The FMP API key is optional. When unavailable (or on the free tier
> where constituent endpoints return 403), the system automatically falls back
> to Wikipedia scraping and a hardcoded universe. Market cap screening uses
> Yahoo Finance's free equity screener API and does not require any API key.

## License

See [LICENSE](LICENSE).
