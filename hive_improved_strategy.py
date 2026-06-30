#!/usr/bin/env python3
"""Improved HIVE BTC + AI/datacenter signal used by the options experiment."""

from __future__ import annotations

import math

import pandas as pd

from hive_strategy import (
    BTC_MA_PERIOD,
    BTC_SYMBOL,
    HIVE_MA_FAST,
    HIVE_MA_SLOW,
    HIVE_RSI_PERIOD,
    HIVE_SYMBOL,
    QQQ_MA_PERIOD,
    QQQ_SYMBOL,
    VIX_SYMBOL,
    rsi,
)


AI_SYMBOLS = ["NVDA", "AVGO", "AMD", "MSFT", "GOOGL", "META", "TSM"]
TRADING_DAYS = 252


def add_improved_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy().sort_index()
    ai_returns = df[AI_SYMBOLS].pct_change().mean(axis=1).fillna(0.0)
    df["AI_INDEX"] = 100.0 * (1.0 + ai_returns).cumprod()

    df["HIVE_MA5"] = df["HIVE"].rolling(HIVE_MA_FAST).mean()
    df["HIVE_MA20"] = df["HIVE"].rolling(HIVE_MA_SLOW).mean()
    df["HIVE_MA50"] = df["HIVE"].rolling(50).mean()
    df["HIVE_MA100"] = df["HIVE"].rolling(100).mean()
    df["HIVE_RSI14"] = rsi(df["HIVE"], HIVE_RSI_PERIOD)
    df["HIVE_VOL20"] = df["HIVE"].pct_change().rolling(20).std() * math.sqrt(TRADING_DAYS)

    df["BTC_MA20"] = df["BTC"].rolling(BTC_MA_PERIOD).mean()
    df["BTC_MA50"] = df["BTC"].rolling(50).mean()
    df["BTC_MA100"] = df["BTC"].rolling(100).mean()
    df["QQQ_MA50"] = df["QQQ"].rolling(QQQ_MA_PERIOD).mean()
    df["QQQ_MA100"] = df["QQQ"].rolling(100).mean()
    df["AI_MA20"] = df["AI_INDEX"].rolling(20).mean()
    df["AI_MA50"] = df["AI_INDEX"].rolling(50).mean()
    df["AI_MA100"] = df["AI_INDEX"].rolling(100).mean()
    return df.dropna()


def classify_improved_signal(row: pd.Series) -> dict:
    btc_bull = row["BTC"] > row["BTC_MA50"] and row["BTC_MA20"] > row["BTC_MA50"]
    btc_bear = row["BTC"] < row["BTC_MA50"] and row["BTC_MA20"] < row["BTC_MA50"]
    ai_bull = row["AI_INDEX"] > row["AI_MA50"] and row["AI_MA20"] > row["AI_MA50"]
    ai_bear = row["AI_INDEX"] < row["AI_MA50"] and row["AI_MA20"] < row["AI_MA50"]
    qqq_bull = row["QQQ"] > row["QQQ_MA100"]
    qqq_bear = row["QQQ"] < row["QQQ_MA100"]
    hive_uptrend = row["HIVE"] > row["HIVE_MA50"] and row["HIVE_MA20"] > row["HIVE_MA50"]
    hive_tradeable = row["HIVE"] > row["HIVE_MA20"]
    hive_breakdown = row["HIVE"] < row["HIVE_MA50"] and row["HIVE_MA20"] < row["HIVE_MA50"]
    overbought = row["HIVE_RSI14"] >= 75.0
    oversold = row["HIVE_RSI14"] <= 30.0

    theme_score = int(btc_bull) + int(ai_bull) + int(qqq_bull)

    if row["VIX"] >= 45.0:
        alloc, regime = 0.0, "risk_off_vix_crisis"
    elif hive_uptrend and btc_bull and ai_bull and qqq_bull:
        alloc, regime = 0.95, "hive_btc_ai_full_long"
    elif hive_uptrend and theme_score >= 2:
        alloc, regime = 0.70, "theme_confirmed_long"
    elif hive_tradeable and theme_score >= 2:
        alloc, regime = 0.45, "theme_tradeable_long"
    elif hive_tradeable and (btc_bull or ai_bull) and qqq_bull:
        alloc, regime = 0.25, "pilot_theme_long"
    elif hive_breakdown and btc_bear and ai_bear and qqq_bear and not oversold and row["VIX"] < 35.0:
        alloc, regime = -0.25, "confirmed_short"
    else:
        alloc, regime = 0.0, "flat_no_edge"

    if overbought and alloc > 0.45:
        alloc *= 0.65
        regime += "_rsi_trim"

    if row["VIX"] >= 30.0 and alloc > 0:
        alloc *= 0.50
        regime += "_vix_trim"

    if row["HIVE_VOL20"] > 0:
        vol_scale = min(1.0, 0.70 / row["HIVE_VOL20"])
        alloc *= vol_scale
        if vol_scale < 0.999:
            regime += "_vol_scaled"

    return {
        "regime": regime,
        "equity_target_alloc": float(alloc),
        "option_call_overlay": float(alloc) >= 0.45,
        "hive": float(row["HIVE"]),
        "hive_rsi14": float(row["HIVE_RSI14"]),
        "hive_vol20": float(row["HIVE_VOL20"]),
        "btc_theme_bull": bool(btc_bull),
        "ai_theme_bull": bool(ai_bull),
        "qqq_risk_on": bool(qqq_bull),
        "theme_score": int(theme_score),
    }
