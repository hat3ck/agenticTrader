"""Technical Analysis Tool — RSI, MACD, Bollinger, SMA/EMA, ATR, Volume.

Uses pandas-ta for indicator computation on mock OHLCV DataFrames.
When yfinance is integrated, this module will consume real DataFrames.
"""

from __future__ import annotations

import random
import datetime as dt

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore

from app.data.cache import market_data_cache, async_get_or_set


def _generate_mock_ohlcv(ticker: str, days: int = 252) -> pd.DataFrame:
    """Generate a realistic-looking OHLCV DataFrame for testing."""
    np.random.seed(hash(ticker) % 2**31)
    dates = pd.bdate_range(end=dt.date.today(), periods=days)
    n = len(dates)  # actual count — may differ from `days` on weekends/holidays
    base_price = {"AAPL": 185, "MSFT": 420, "NVDA": 890, "GOOGL": 175, "AMZN": 195}.get(
        ticker.upper(), random.randint(50, 400)
    )
    # Geometric Brownian Motion
    returns = np.random.normal(0.0005, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            "High": prices * (1 + np.random.uniform(0.001, 0.025, n)),
            "Low": prices * (1 - np.random.uniform(0.001, 0.025, n)),
            "Close": prices,
            "Volume": np.random.randint(5_000_000, 80_000_000, n),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df.round(2)


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
        df = _generate_mock_ohlcv(ticker)
        indicators = _compute_indicators(df)
        indicators["ticker"] = ticker.upper()
        return indicators

    return await async_get_or_set(market_data_cache, f"tech:{ticker.upper()}", _fetch)
