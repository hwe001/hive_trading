#!/usr/bin/env python3
"""
HIVE-only equity trading strategy: long and short positions.

QQQ and Bitcoin are used as signals only — never traded.
No options overlay in this version (see tradier-options branch for options).

Signals:
  BTC 20-day MA  →  primary trend driver (HIVE is a Bitcoin miner)
  QQQ 50-day MA  →  tech / risk-on sentiment filter
  VIX            →  risk scaling
  HIVE RSI-14    →  entry timing within the regime

Equity positions (fraction of account equity):
  > 0  →  LONG HIVE
  < 0  →  SHORT HIVE
  = 0  →  FLAT (cash)
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

# Traded asset
HIVE_SYMBOL = "HIVE"

# Signal-only assets (never traded)
QQQ_SYMBOL = "QQQ"
BTC_SYMBOL = "BTC-USD"
VIX_SYMBOL = "^VIX"

# Lookback periods (trading days for equities, calendar days for BTC)
HIVE_RSI_PERIOD = 14
HIVE_MA_FAST = 5
HIVE_MA_SLOW = 20
BTC_MA_PERIOD = 20
QQQ_MA_PERIOD = 50

# Equity allocation targets (fraction of account equity; negative = short)
LONG_FULL     =  1.00
LONG_MODERATE =  0.70
LONG_LIGHT    =  0.35
LONG_PILOT    =  0.20
FLAT          =  0.00
SHORT_LIGHT   = -0.25
SHORT_MODERATE= -0.40
SHORT_FULL    = -0.55

# VIX thresholds
VIX_CALM      = 18.0
VIX_CAUTION   = 28.0
VIX_ELEVATED  = 38.0
VIX_DANGER    = 50.0

# RSI extremes
RSI_OVERBOUGHT = 72.0
RSI_OVERSOLD   = 28.0


def rsi(series: pd.Series, period: int = HIVE_RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)


# ---------------------------------------------------------------------------
# Signal output
# ---------------------------------------------------------------------------

def _signal(
    regime: str,
    regime_desc: str,
    equity_action: str,
    equity_alloc: float,
    confidence: str,
    reason: str,
    row: pd.Series,
) -> dict:
    return {
        "timestamp_ny": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S"),
        "regime": regime,
        "regime_description": regime_desc,
        "equity_action": equity_action,
        "equity_target_alloc": equity_alloc,
        "confidence": confidence,
        "reason": reason,
        # HIVE indicators
        "hive": float(row["HIVE"]),
        "hive_ma5": float(row["HIVE_MA5"]),
        "hive_ma20": float(row["HIVE_MA20"]),
        "hive_rsi14": float(row["HIVE_RSI14"]),
        "hive_above_ma20": float(row["HIVE"]) >= float(row["HIVE_MA20"]),
        "hive_rsi_overbought": float(row["HIVE_RSI14"]) >= RSI_OVERBOUGHT,
        "hive_rsi_oversold": float(row["HIVE_RSI14"]) <= RSI_OVERSOLD,
        # Signal inputs (not traded)
        "btc": float(row["BTC"]),
        "btc_ma20": float(row["BTC_MA20"]),
        "btc_above_ma20": float(row["BTC"]) >= float(row["BTC_MA20"]),
        "qqq": float(row["QQQ"]),
        "qqq_ma50": float(row["QQQ_MA50"]),
        "qqq_above_ma50": float(row["QQQ"]) >= float(row["QQQ_MA50"]),
        "vix": float(row["VIX"]),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def classify_signal(row: pd.Series) -> dict:
    """
    Classify the current regime and return a HIVE-only equity trading signal.

    Expected columns in `row`:
        HIVE, HIVE_MA5, HIVE_MA20, HIVE_RSI14
        BTC, BTC_MA20
        QQQ, QQQ_MA50
        VIX
    """
    vix            = float(row["VIX"])
    btc_above_ma20 = float(row["BTC"]) >= float(row["BTC_MA20"])
    qqq_above_ma50 = float(row["QQQ"]) >= float(row["QQQ_MA50"])
    hive_rsi       = float(row["HIVE_RSI14"])
    overbought     = hive_rsi >= RSI_OVERBOUGHT
    oversold       = hive_rsi <= RSI_OVERSOLD

    # ── VIX crisis ────────────────────────────────────────────────────────
    if vix >= VIX_DANGER:
        return _signal(
            "danger", "VIX Crisis — All Flat",
            "FLAT", FLAT, "high",
            f"VIX {vix:.1f} ≥ {VIX_DANGER}: extreme fear. Close all HIVE positions.",
            row,
        )

    # ── Both signals bullish ───────────────────────────────────────────────
    if btc_above_ma20 and qqq_above_ma50:
        if overbought:
            return _signal(
                "bull_cautious", "Bull but HIVE Overbought — Light Long",
                "LONG", LONG_LIGHT, "medium",
                f"Bull macro intact but HIVE RSI {hive_rsi:.0f} is overbought. Trim to light long.",
                row,
            )
        if vix >= VIX_ELEVATED:
            return _signal(
                "bull_cautious", "Bull but VIX Elevated — Light Long",
                "LONG", LONG_LIGHT, "medium",
                f"BTC and QQQ bullish but VIX {vix:.1f} is elevated. Light long only.",
                row,
            )
        if vix < VIX_CALM:
            return _signal(
                "bull_strong", "Strong Bull — Full Long",
                "LONG", LONG_FULL, "high",
                f"BTC above MA20, QQQ above MA50, VIX {vix:.1f} calm, RSI {hive_rsi:.0f}. Full long.",
                row,
            )
        return _signal(
            "bull", "Bull — Moderate Long",
            "LONG", LONG_MODERATE, "medium",
            f"BTC above MA20, QQQ above MA50, VIX {vix:.1f}. Moderate long.",
            row,
        )

    # ── BTC bullish but QQQ soft ───────────────────────────────────────────
    if btc_above_ma20 and not qqq_above_ma50:
        return _signal(
            "btc_bull_qqq_soft", "BTC Up, QQQ Soft — Pilot Long",
            "LONG", LONG_PILOT, "medium",
            "BTC trend positive but QQQ below MA50. Pilot long only.",
            row,
        )

    # ── BTC bearish but QQQ firm ───────────────────────────────────────────
    if not btc_above_ma20 and qqq_above_ma50:
        return _signal(
            "btc_bear_qqq_firm", "BTC Weak, QQQ Firm — Flat",
            "FLAT", FLAT, "medium",
            "Conflicting signals: BTC (HIVE's primary driver) is below MA20 but QQQ holds. Stay flat.",
            row,
        )

    # ── Both signals bearish ───────────────────────────────────────────────
    if vix >= VIX_ELEVATED or oversold:
        regime = "bear_panic" if vix >= VIX_ELEVATED else "bear_oversold"
        label  = "Bear + High VIX — Flat" if vix >= VIX_ELEVATED else "Bear but Oversold — Light Short"
        equity_alloc = SHORT_LIGHT if vix < VIX_ELEVATED else FLAT
        return _signal(
            regime, label,
            "SHORT" if equity_alloc < 0 else "FLAT", equity_alloc, "medium",
            (
                f"BTC below MA20, QQQ below MA50. "
                f"{'VIX ' + str(round(vix, 1)) + ' elevated: squeeze risk too high for full short. ' if vix >= VIX_ELEVATED else ''}"
                f"{'HIVE RSI ' + str(round(hive_rsi, 0)) + ' oversold: bounce risk. ' if oversold else ''}"
            ),
            row,
        )

    equity_alloc = SHORT_FULL if vix < VIX_CAUTION else SHORT_MODERATE
    return _signal(
        "bear", "Bear — Short Equity",
        "SHORT", equity_alloc, "medium",
        f"BTC below MA20, QQQ below MA50, VIX {vix:.1f}. Short HIVE {abs(equity_alloc):.0%}.",
        row,
    )
