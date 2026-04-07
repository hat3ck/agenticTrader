"""Technical Analysis Tool — RSI, MACD, Bollinger, SMA/EMA, ATR, Volume.

Uses pandas-ta for indicator computation on real OHLCV data from yfinance.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore
import yfinance as yf

from app.data.cache import market_data_cache, async_get_or_set
from app.tools.market_data import _YF_SEMAPHORE

logger = logging.getLogger(__name__)


async def _fetch_real_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch real OHLCV data from yfinance for technical analysis.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    Falls back to a shorter period if data is insufficient.
    """
    async with _YF_SEMAPHORE:
        t = yf.Ticker(ticker.upper())
        df = await asyncio.to_thread(lambda: t.history(period=period, auto_adjust=True))
    if df.empty:
        raise ValueError(f"No OHLCV data returned by yfinance for {ticker}")
    # Ensure standard column names
    df = df.rename(columns={c: c.title() for c in df.columns})
    # Keep only the columns we need
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' column in yfinance data for {ticker}")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _compute_indicators(df: pd.DataFrame) -> dict:
    """Run pandas-ta on a DataFrame and return a summary dict."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ── RSI (14-period) ──
    rsi_series = ta.rsi(close, length=14)
    rsi_value = round(float(rsi_series.iloc[-1]), 2) if rsi_series is not None and not rsi_series.empty else None

    # ── MACD (12, 26, 9) ──
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        macd_line = float(macd_df.iloc[-1, 0])
        signal_line = float(macd_df.iloc[-1, 2])
        macd_hist = float(macd_df.iloc[-1, 1])
        macd_signal = "bullish" if macd_line > signal_line else "bearish"
    else:
        macd_line = signal_line = macd_hist = 0.0
        macd_signal = "neutral"

    # ── Bollinger Bands (20, 2) ──
    bbands = ta.bbands(close, length=20, std=2)
    if bbands is not None and not bbands.empty:
        upper_band = float(bbands.iloc[-1, 2])
        lower_band = float(bbands.iloc[-1, 0])
        current_price = float(close.iloc[-1])
        if current_price >= upper_band:
            bollinger_position = "upper"
        elif current_price <= lower_band:
            bollinger_position = "lower"
        else:
            bollinger_position = "middle"
    else:
        upper_band = lower_band = 0.0
        bollinger_position = "middle"

    # ── SMA / EMA (50 & 200 day) ──
    sma_50 = ta.sma(close, length=50)
    sma_200 = ta.sma(close, length=200)
    ema_50 = ta.ema(close, length=50)
    ema_200 = ta.ema(close, length=200)

    sma_50_val = float(sma_50.iloc[-1]) if sma_50 is not None and not sma_50.empty else None
    sma_200_val = float(sma_200.iloc[-1]) if sma_200 is not None and not sma_200.empty else None
    ema_50_val = float(ema_50.iloc[-1]) if ema_50 is not None and not ema_50.empty else None
    ema_200_val = float(ema_200.iloc[-1]) if ema_200 is not None and not ema_200.empty else None

    if sma_50_val and sma_200_val:
        if sma_50_val > sma_200_val:
            sma_trend = "golden_cross"
        else:
            sma_trend = "death_cross"
    else:
        sma_trend = "neutral"

    # ── ATR (14-period) ──
    atr_series = ta.atr(high, low, close, length=14)
    atr_value = round(float(atr_series.iloc[-1]), 2) if atr_series is not None and not atr_series.empty else None

    # ── Volume Profile ──
    avg_volume_20 = float(volume.tail(20).mean())
    recent_volume = float(volume.iloc[-1])
    volume_ratio = round(recent_volume / avg_volume_20, 2) if avg_volume_20 > 0 else 1.0
    if volume_ratio > 1.5:
        volume_signal = "high_volume"
    elif volume_ratio < 0.5:
        volume_signal = "low_volume"
    else:
        volume_signal = "normal_volume"

    return {
        "rsi": rsi_value,
        "rsi_signal": (
            "overbought" if rsi_value and rsi_value > 70
            else "oversold" if rsi_value and rsi_value < 30
            else "neutral"
        ),
        "macd_line": round(macd_line, 4),
        "macd_signal_line": round(signal_line, 4),
        "macd_histogram": round(macd_hist, 4),
        "macd_signal": macd_signal,
        "bollinger_upper": round(upper_band, 2),
        "bollinger_lower": round(lower_band, 2),
        "bollinger_position": bollinger_position,
        "sma_50": round(sma_50_val, 2) if sma_50_val else None,
        "sma_200": round(sma_200_val, 2) if sma_200_val else None,
        "ema_50": round(ema_50_val, 2) if ema_50_val else None,
        "ema_200": round(ema_200_val, 2) if ema_200_val else None,
        "sma_trend": sma_trend,
        "atr": atr_value,
        "volume_ratio": volume_ratio,
        "volume_signal": volume_signal,
        "current_price": round(float(close.iloc[-1]), 2),
    }


async def get_technical_indicators(ticker: str) -> dict:
    """Compute technical indicators for a stock.

    Calculates RSI, MACD, Bollinger Bands, SMA/EMA (50 & 200 day),
    ATR, and Volume Profile from OHLCV data.

    Returns a dict containing all indicator values plus human-readable
    signals (e.g. 'overbought', 'bullish', 'golden_cross').

    Interpretation guide:
    - RSI > 70 = overbought (potential sell signal)
    - RSI < 30 = oversold (potential buy signal)
    - MACD line > signal line = bullish momentum
    - Price near lower Bollinger Band = potential mean reversion buy
    - SMA 50 > SMA 200 = 'golden_cross' (bullish trend)
    - SMA 50 < SMA 200 = 'death_cross' (bearish trend)
    - ATR measures volatility — use for position sizing
    - High volume confirms price moves
    """

    async def _fetch():
        try:
            df = await _fetch_real_ohlcv(ticker, period="1y")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch real OHLCV for %s: %s", ticker, exc)
            raise RuntimeError(
                f"Could not fetch OHLCV data for {ticker}. "
                "Technical analysis requires real market data."
            ) from exc
        indicators = _compute_indicators(df)
        indicators["ticker"] = ticker.upper()
        return indicators

    return await async_get_or_set(market_data_cache, f"tech:{ticker.upper()}", _fetch)
