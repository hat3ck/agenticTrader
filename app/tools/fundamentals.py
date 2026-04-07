"""Fundamental Analysis Tool — Financial metrics calculator.

Data sources:
  • Primary  — yfinance (balance sheet, income statement, cash flow, .info)
  • Fallback — SEC EDGAR XBRL company-facts API (free, official filings)

Metrics: Market Cap, P/E, P/B, PEG, D/E, FCF, ROE, Dividend Yield,
Revenue Growth Rate, EPS, Profit Margin, Payout Ratio.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import yfinance as yf

from app.config import settings
from app.data.cache import fundamental_cache, async_get_or_set
from app.tools.market_data import _YF_SEMAPHORE

logger = logging.getLogger(__name__)

# ── SEC EDGAR helpers ────────────────────────────────────────────────────────

_CIK_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# In-memory ticker→CIK map (populated lazily)
_ticker_to_cik: dict[str, str] = {}


def _safe(val: Any, fallback: Any = None) -> Any:
    """Return *val* unless it is None / NaN, in which case return *fallback*."""
    if val is None:
        return fallback
    try:
        if val != val:  # noqa: PLR0124  (NaN ≠ itself)
            return fallback
    except (TypeError, ValueError):
        pass
    return val


def _safe_round(val: Any, decimals: int = 2, fallback: Any = None) -> Any:
    """Round *val* if numeric, else return *fallback*."""
    val = _safe(val, fallback)
    if val is None:
        return fallback
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return fallback


async def _ensure_cik_map() -> None:
    """Populate _ticker_to_cik from SEC EDGAR company tickers JSON (once)."""
    global _ticker_to_cik  # noqa: PLW0603
    if _ticker_to_cik:
        return
    try:
        headers = {"User-Agent": settings.sec_edgar_user_agent}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(_CIK_LOOKUP_URL, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        _ticker_to_cik = {
            entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
            for entry in data.values()
        }
        logger.debug("Loaded %d ticker→CIK mappings from SEC", len(_ticker_to_cik))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load SEC CIK map: %s", exc)


async def _fetch_edgar_facts(ticker: str) -> dict | None:
    """Fetch XBRL company facts from SEC EDGAR.  Returns None on failure."""
    await _ensure_cik_map()
    cik = _ticker_to_cik.get(ticker.upper())
    if not cik:
        logger.debug("No CIK found for %s", ticker)
        return None
    url = _COMPANY_FACTS_URL.format(cik=cik)
    try:
        headers = {"User-Agent": settings.sec_edgar_user_agent}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("SEC EDGAR fetch failed for %s (CIK %s): %s", ticker, cik, exc)
        return None


def _latest_fact(facts: dict, taxonomy: str, concept: str) -> float | None:
    """Extract the latest filed value for a given XBRL concept."""
    try:
        units = facts["facts"][taxonomy][concept]["units"]
        # Prefer USD, fall back to first available unit
        values = units.get("USD") or units.get("USD/shares") or next(iter(units.values()), None)
        if not values:
            return None
        # Sort by end date descending; take most recent 10-K/10-Q
        annual = [v for v in values if v.get("form") in ("10-K", "10-Q")]
        if not annual:
            annual = values
        annual.sort(key=lambda v: v.get("end", ""), reverse=True)
        return float(annual[0]["val"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _two_latest_annual_facts(facts: dict, taxonomy: str, concept: str) -> tuple[float | None, float | None]:
    """Return (latest, previous) annual filed values for YoY growth calculation."""
    try:
        units = facts["facts"][taxonomy][concept]["units"]
        values = units.get("USD") or next(iter(units.values()), None)
        if not values:
            return None, None
        annual = [v for v in values if v.get("form") == "10-K"]
        if len(annual) < 2:
            return None, None
        annual.sort(key=lambda v: v.get("end", ""), reverse=True)
        return float(annual[0]["val"]), float(annual[1]["val"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None, None


# ── yfinance helpers ─────────────────────────────────────────────────────────

def _yf_ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker.upper())


def _extract_yf_fundamentals(t: yf.Ticker) -> dict[str, Any]:
    """Pull all available fundamental data from a yfinance Ticker (sync).

    Runs in a thread via asyncio.to_thread.
    """
    info: dict = t.info or {}

    # ── Direct from .info ────────────────────────────────────────────────
    market_cap = _safe(info.get("marketCap"))
    # Keep trailing and forward PE separate — never silently swap
    trailing_pe = _safe(info.get("trailingPE"))
    forward_pe = _safe(info.get("forwardPE"))
    pb = _safe(info.get("priceToBook"))
    peg = _safe(info.get("pegRatio"))
    de = _safe(info.get("debtToEquity"))
    # yfinance reports debtToEquity as a percentage (e.g. 180 = 1.8x)
    if de is not None:
        de = round(de / 100, 2)
    fcf = _safe(info.get("freeCashflow"))
    roe = _safe(info.get("returnOnEquity"))
    # yfinance ROE is a decimal (0.25 = 25%)
    if roe is not None:
        roe = round(roe * 100, 2)
    div_yield = _safe(info.get("dividendYield"))
    if div_yield is not None:
        div_yield = round(div_yield * 100, 2)
    # Keep trailing and forward EPS separate
    trailing_eps = _safe(info.get("trailingEps"))
    forward_eps = _safe(info.get("forwardEps"))
    margin = _safe(info.get("profitMargins"))
    if margin is not None:
        margin = round(margin * 100, 2)
    payout = _safe(info.get("payoutRatio"))
    if payout is not None:
        payout = round(payout * 100, 2)

    # ── Revenue growth from financials ───────────────────────────────────
    rev_growth = _safe(info.get("revenueGrowth"))
    if rev_growth is not None:
        rev_growth = round(rev_growth * 100, 2)
    else:
        # Calculate from income statement if .info doesn't have it
        try:
            inc = t.financials  # columns = fiscal year ends
            if inc is not None and "Total Revenue" in inc.index and inc.shape[1] >= 2:
                rev_latest = float(inc.loc["Total Revenue"].iloc[0])
                rev_prev = float(inc.loc["Total Revenue"].iloc[1])
                if rev_prev and rev_prev != 0:
                    rev_growth = round((rev_latest - rev_prev) / abs(rev_prev) * 100, 2)
        except Exception:  # noqa: BLE001
            pass

    # ── FCF fallback: operating cash flow − capex from cashflow stmt ─────
    if fcf is None:
        try:
            cf = t.cashflow
            if cf is not None:
                ocf = cf.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cf.index else None
                capex = cf.loc["Capital Expenditure"].iloc[0] if "Capital Expenditure" in cf.index else None
                if ocf is not None and capex is not None:
                    fcf = float(ocf) + float(capex)  # capex is typically negative
        except Exception:  # noqa: BLE001
            pass

    # ── D/E fallback from balance sheet ──────────────────────────────────
    if de is None:
        try:
            bs = t.balance_sheet
            if bs is not None:
                total_debt = None
                for label in ("Total Debt", "Long Term Debt"):
                    if label in bs.index:
                        total_debt = float(bs.loc[label].iloc[0])
                        break
                equity = None
                for label in ("Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"):
                    if label in bs.index:
                        equity = float(bs.loc[label].iloc[0])
                        break
                if total_debt is not None and equity and equity != 0:
                    de = round(total_debt / equity, 2)
        except Exception:  # noqa: BLE001
            pass

    # ── ROE fallback: net income / equity ────────────────────────────────
    if roe is None:
        try:
            inc = t.financials
            bs = t.balance_sheet
            if inc is not None and bs is not None:
                ni = None
                for label in ("Net Income", "Net Income Common Stockholders"):
                    if label in inc.index:
                        ni = float(inc.loc[label].iloc[0])
                        break
                equity = None
                for label in ("Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"):
                    if label in bs.index:
                        equity = float(bs.loc[label].iloc[0])
                        break
                if ni is not None and equity and equity != 0:
                    roe = round(ni / equity * 100, 2)
        except Exception:  # noqa: BLE001
            pass

    return {
        "market_cap": market_cap,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "pb": pb,
        "peg": peg,
        "de": de,
        "fcf": fcf,
        "roe": roe,
        "div_yield": div_yield,
        "rev_growth": rev_growth,
        "trailing_eps": trailing_eps,
        "forward_eps": forward_eps,
        "margin": margin,
        "payout": payout,
    }


# ── SEC EDGAR enrichment ────────────────────────────────────────────────────

async def _enrich_from_edgar(ticker: str, yf_data: dict[str, Any]) -> dict[str, Any]:
    """Fill gaps in *yf_data* using SEC EDGAR XBRL facts.

    Only overwrites None values — yfinance is preferred when available.
    """
    facts = await _fetch_edgar_facts(ticker)
    if facts is None:
        return yf_data

    data = dict(yf_data)  # shallow copy

    # Revenue (for growth calc)
    if data["rev_growth"] is None:
        latest, prev = _two_latest_annual_facts(facts, "us-gaap", "Revenues")
        if latest is None or prev is None:
            latest, prev = _two_latest_annual_facts(facts, "us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax")
        if latest is not None and prev is not None and prev != 0:
            data["rev_growth"] = round((latest - prev) / abs(prev) * 100, 2)

    # EPS (fill trailing EPS from EDGAR if missing)
    if data["trailing_eps"] is None:
        eps_val = _latest_fact(facts, "us-gaap", "EarningsPerShareDiluted")
        if eps_val is None:
            eps_val = _latest_fact(facts, "us-gaap", "EarningsPerShareBasic")
        data["trailing_eps"] = _safe_round(eps_val)

    # Net income for ROE / margin
    net_income = _latest_fact(facts, "us-gaap", "NetIncomeLoss")

    # ROE
    if data["roe"] is None and net_income is not None:
        equity = _latest_fact(facts, "us-gaap", "StockholdersEquity")
        if equity and equity != 0:
            data["roe"] = round(net_income / equity * 100, 2)

    # D/E
    if data["de"] is None:
        debt = _latest_fact(facts, "us-gaap", "LongTermDebt")
        if debt is None:
            debt = _latest_fact(facts, "us-gaap", "LongTermDebtNoncurrent")
        equity = _latest_fact(facts, "us-gaap", "StockholdersEquity")
        if debt is not None and equity and equity != 0:
            data["de"] = round(debt / equity, 2)

    # FCF
    if data["fcf"] is None:
        ocf = _latest_fact(facts, "us-gaap", "NetCashProvidedByOperatingActivities")
        capex = _latest_fact(facts, "us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment")
        if ocf is not None and capex is not None:
            data["fcf"] = round(ocf - capex, 0)
        elif ocf is not None:
            data["fcf"] = round(ocf, 0)

    # Profit margin
    if data["margin"] is None and net_income is not None:
        revenue = _latest_fact(facts, "us-gaap", "Revenues")
        if revenue is None:
            revenue = _latest_fact(facts, "us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax")
        if revenue and revenue != 0:
            data["margin"] = round(net_income / revenue * 100, 2)

    return data


# ── public API ───────────────────────────────────────────────────────────────

async def get_fundamental_metrics(ticker: str) -> dict:
    """Calculate key fundamental metrics for a stock.

    Returns a dict containing:
    - market_cap: Total market value in USD
    - pe_ratio: Price-to-Earnings ratio
    - pb_ratio: Price-to-Book ratio
    - peg_ratio: P/E adjusted for earnings growth
    - debt_to_equity: Financial leverage ratio
    - free_cash_flow: FCF in USD
    - roe: Return on Equity (percentage)
    - dividend_yield: Annual dividend yield (percentage)
    - revenue_growth: Year-over-year revenue growth (percentage)
    - earnings_per_share: EPS in USD
    - profit_margin: Net profit margin (percentage)
    - payout_ratio: Dividend payout ratio (percentage)

    Data sourced from yfinance (primary) with SEC EDGAR XBRL fallback.

    Use this to evaluate a stock's valuation, financial health, and quality.
    P/E < 20 is generally reasonable; PEG < 1 suggests undervaluation relative
    to growth; ROE > 15% signals quality; D/E > 2 is risky.
    """

    async def _fetch():
        try:
            async with _YF_SEMAPHORE:
                t = _yf_ticker(ticker)
                yf_data = await asyncio.to_thread(_extract_yf_fundamentals, t)
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance fundamentals failed for %s: %s — trying SEC EDGAR only", ticker, exc)
            yf_data = {
                "market_cap": None, "trailing_pe": None, "forward_pe": None,
                "pb": None, "peg": None,
                "de": None, "fcf": None, "roe": None, "div_yield": None,
                "rev_growth": None, "trailing_eps": None, "forward_eps": None,
                "margin": None, "payout": None,
            }

        # Enrich missing fields from SEC EDGAR
        data = await _enrich_from_edgar(ticker, yf_data)

        trailing_pe = _safe_round(data["trailing_pe"], 2)
        forward_pe = _safe_round(data["forward_pe"], 2)
        # Use trailing PE as primary; only fall back to forward if trailing is unavailable
        pe_display = trailing_pe if trailing_pe is not None else forward_pe
        peg = _safe_round(data["peg"], 2)
        pb = _safe_round(data["pb"], 2)
        de = _safe_round(data["de"], 2)
        fcf = _safe_round(data["fcf"], 0)
        roe = _safe_round(data["roe"], 2)
        trailing_eps = _safe_round(data["trailing_eps"], 2)
        forward_eps = _safe_round(data["forward_eps"], 2)

        return {
            "ticker": ticker.upper(),
            "market_cap": _safe_round(data["market_cap"], 0),
            "pe_ratio": pe_display,
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "pb_ratio": pb,
            "peg_ratio": peg,
            "debt_to_equity": de,
            "free_cash_flow": fcf,
            "roe": roe,
            "dividend_yield": _safe_round(data["div_yield"], 2),
            "revenue_growth": _safe_round(data["rev_growth"], 2),
            "earnings_per_share": trailing_eps,
            "trailing_eps": trailing_eps,
            "forward_eps": forward_eps,
            "profit_margin": _safe_round(data["margin"], 2),
            "payout_ratio": _safe_round(data["payout"], 2),
            "valuation_summary": _valuation_summary(pe_display, peg, pb),
            "health_summary": _health_summary(de, fcf),
            "quality_summary": _quality_summary(roe, data.get("rev_growth"), data.get("div_yield")),
        }

    return await async_get_or_set(fundamental_cache, f"fund:{ticker.upper()}", _fetch)


def _valuation_summary(pe: float | None, peg: float | None, pb: float | None) -> str:
    parts = []
    if pe is not None:
        if pe < 15:
            parts.append("attractively valued (low P/E)")
        elif pe < 25:
            parts.append("reasonably valued")
        else:
            parts.append("premium valuation (high P/E)")
    else:
        parts.append("P/E data unavailable")

    if peg is not None:
        if peg < 1:
            parts.append("undervalued relative to growth (PEG < 1)")
        elif peg < 1.5:
            parts.append("fairly priced for growth")
        else:
            parts.append("growth may be priced in")

    return "; ".join(parts)


def _health_summary(de: float | None, fcf: float | None) -> str:
    parts = []
    if de is not None:
        if de < 0.5:
            parts.append("very low leverage")
        elif de < 1.5:
            parts.append("manageable debt levels")
        else:
            parts.append("high leverage — elevated financial risk")
    else:
        parts.append("debt data unavailable")

    if fcf is not None:
        if fcf > 10e9:
            parts.append("strong cash generation")
        elif fcf > 0:
            parts.append("positive free cash flow")
        else:
            parts.append("negative FCF — cash burn risk")
    else:
        parts.append("FCF data unavailable")

    return "; ".join(parts)


def _quality_summary(roe: float | None, rev_g: float | None, div_yield: float | None) -> str:
    parts = []
    if roe is not None:
        if roe > 20:
            parts.append("high-quality business (strong ROE)")
        elif roe > 10:
            parts.append("decent returns on equity")
        else:
            parts.append("low ROE — management effectiveness questionable")
    else:
        parts.append("ROE data unavailable")

    if rev_g is not None:
        if rev_g > 15:
            parts.append("strong revenue growth")
        elif rev_g > 0:
            parts.append("moderate growth")
        else:
            parts.append("declining revenue")

    if div_yield is not None and div_yield > 2:
        parts.append(f"attractive {div_yield}% dividend yield")

    return "; ".join(parts)
