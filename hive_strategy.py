#!/usr/bin/env python3
"""
HIVE-only trading strategy: long equity, short equity, and options overlay.

QQQ and Bitcoin are used as signals only -- never traded.

Signals:
  BTC 20-day MA  ->  primary trend driver (HIVE is a Bitcoin miner)
  QQQ 50-day MA  ->  tech / risk-on sentiment filter
  VIX            ->  risk scaling and option strategy selector
  HIVE RSI-14    ->  entry timing within the regime

Equity positions (fraction of account equity):
  > 0  ->  LONG HIVE
  < 0  ->  SHORT HIVE
  = 0  ->  FLAT (cash)

Options overlay is advisory only; sized per 100-share blocks.
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
# Options advisory
# ---------------------------------------------------------------------------

def _options_advisory(regime: str, equity_alloc: float) -> list[dict]:
    """Return ordered advisory options plays for the current regime."""

    if regime == "danger":
        return [{
            "priority": 1,
            "strategy": "STAND ASIDE -- NO POSITIONS",
            "detail": (
                "VIX is in crisis. Close or hedge all open options. "
                "Do not open new trades until volatility subsides."
            ),
        }]

    if regime == "bull_strong":
        return [{
            "priority": 1,
            "strategy": "SELL COVERED CALL",
            "detail": (
                "Sell 1 call per 100 shares long, 20-25% OTM, 30-45 DTE. "
                "Target premium >= 4-6% of stock price. "
                "Roll up-and-out if HIVE rallies through the strike."
            ),
        }]

    if regime == "bull":
        return [
            {
                "priority": 1,
                "strategy": "SELL COVERED CALL",
                "detail": (
                    "Sell 1 call per 100 shares, 15-20% OTM, 30 DTE. "
                    "Tighter than bull_strong to collect more premium in a moderately cautious market."
                ),
            },
            {
                "priority": 2,
                "strategy": "BUY PROTECTIVE PUT (OPTIONAL)",
                "detail": (
                    "Buy 1 put per 100 shares, 10-15% OTM, 30 DTE, partially funded by call premium. "
                    "Skip if net collar cost exceeds 1% of stock price."
                ),
            },
        ]

    if regime == "bull_cautious":
        return [
            {
                "priority": 1,
                "strategy": "SELL COVERED CALL (TIGHT COLLAR)",
                "detail": (
                    "Sell 1 call per 100 shares, 10% OTM, 30 DTE. "
                    "Tighter strike maximises premium in a cautious environment."
                ),
            },
            {
                "priority": 2,
                "strategy": "BUY PROTECTIVE PUT",
                "detail": (
                    "Buy 1 put per 100 shares, 8-10% OTM, 30 DTE. "
                    "Collar caps both upside and downside -- appropriate when conviction is low."
                ),
            },
        ]

    if regime == "btc_bull_qqq_soft":
        return [
            {
                "priority": 1,
                "strategy": "SELL CASH-SECURED PUT",
                "detail": (
                    "Sell 1 put per 100 target shares, 10-15% OTM, 30 DTE. "
                    "Collect premium while waiting for QQQ to recover. "
                    "If assigned, you own HIVE at an effective discount."
                ),
            },
            {
                "priority": 2,
                "strategy": "BULL CALL SPREAD",
                "detail": (
                    "Buy 5% OTM call, sell 20% OTM call, 30 DTE. "
                    "Defined-risk upside participation if BTC breaks higher while QQQ lags."
                ),
            },
        ]

    if regime == "btc_bear_qqq_firm":
        return [
            {
                "priority": 1,
                "strategy": "IRON CONDOR (HIGH IV ONLY)",
                "detail": (
                    "Sell 20% OTM call + 20% OTM put; buy 30% OTM call + 30% OTM put, 30-45 DTE. "
                    "Profits if HIVE stays range-bound. Only when HIVE IV rank > 50%."
                ),
            },
            {
                "priority": 2,
                "strategy": "STAND ASIDE",
                "detail": (
                    "If IV rank < 50%, premium is insufficient. Stay in cash and wait for clearer direction."
                ),
            },
        ]

    if regime == "bear":
        return [
            {
                "priority": 1,
                "strategy": "SELL COVERED PUT",
                "detail": (
                    "Sell 1 put per 100 shares short, 15-20% OTM below entry, 30-45 DTE. "
                    "Premium offsets short borrow cost. If assigned, you buy back at a discount."
                ),
            },
            {
                "priority": 2,
                "strategy": "BUY CALL SPREAD (SQUEEZE STOP)",
                "detail": (
                    "Buy 20% OTM call, sell 35% OTM call, 30 DTE. "
                    "Defines loss on a sharp upside spike. Cost < 30% of put premium collected."
                ),
            },
        ]

    if regime in ("bear_oversold", "bear_panic"):
        return [
            {
                "priority": 1,
                "strategy": "BUY PUT SPREAD (DEFINED RISK)",
                "detail": (
                    "Buy ATM or 5% OTM put, sell 25% OTM put, 30-45 DTE. "
                    "Prefer over naked short equity: RSI oversold or VIX spike "
                    "both raise short-squeeze risk. Bounded loss, bounded gain."
                ),
            },
        ]

    return [{
        "priority": 1,
        "strategy": "NO OPTION OVERLAY",
        "detail": "Position too small to efficiently pair with options. Scale up first or stay flat.",
    }]


# ---------------------------------------------------------------------------
# Signal output
# ---------------------------------------------------------------------------

def _signal(
    regime: str,
    regime_desc: str,
    equity_action: str,
    equity_alloc: float,
    options_plays: list[dict],
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
        "options_plays": options_plays,
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
    Classify the current regime and return a HIVE-only trading signal.

    Expected columns in `row`:
        HIVE, HIVE_MA5, HIVE_MA20, HIVE_RSI14
        BTC, BTC_MA20
        QQQ, QQQ_MA50
        VIX
    """
    vix              = float(row["VIX"])
    btc_above_ma20   = float(row["BTC"])  >= float(row["BTC_MA20"])
    qqq_above_ma50   = float(row["QQQ"])  >= float(row["QQQ_MA50"])
    hive_rsi         = float(row["HIVE_RSI14"])
    hive_above_ma20  = float(row["HIVE"]) >= float(row["HIVE_MA20"])
    overbought       = hive_rsi >= RSI_OVERBOUGHT
    oversold         = hive_rsi <= RSI_OVERSOLD

    # VIX crisis
    if vix >= VIX_DANGER:
        plays = _options_advisory("danger", FLAT)
        return _signal(
            "danger", "VIX Crisis -- All Flat",
            "FLAT", FLAT, plays, "high",
            f"VIX {vix:.1f} >= {VIX_DANGER}: extreme fear. "
            "Close all HIVE positions and new options.",
            row,
        )

    # Both signals bullish
    if btc_above_ma20 and qqq_above_ma50:
        if overbought:
            plays = _options_advisory("bull_cautious", LONG_LIGHT)
            return _signal(
                "bull_cautious", "Bull but HIVE Overbought -- Light Long",
                "LONG", LONG_LIGHT, plays, "medium",
                f"Bull macro intact but HIVE RSI {hive_rsi:.0f} is overbought. "
                "Trim to light long; tight collar guards the position.",
                row,
            )
        if vix >= VIX_ELEVATED:
            plays = _options_advisory("bull_cautious", LONG_LIGHT)
            return _signal(
                "bull_cautious", "Bull but VIX Elevated -- Light Long",
                "LONG", LONG_LIGHT, plays, "medium",
                f"BTC and QQQ bullish but VIX {vix:.1f} is elevated. "
                "Light long with collar protection.",
                row,
            )
        if vix < VIX_CALM:
            plays = _options_advisory("bull_strong", LONG_FULL)
            return _signal(
                "bull_strong", "Strong Bull -- Full Long",
                "LONG", LONG_FULL, plays, "high",
                f"BTC above MA20, QQQ above MA50, VIX {vix:.1f} calm, RSI {hive_rsi:.0f}. "
                "Full long; sell covered calls to harvest IV.",
                row,
            )
        plays = _options_advisory("bull", LONG_MODERATE)
        return _signal(
            "bull", "Bull -- Moderate Long",
            "LONG", LONG_MODERATE, plays, "medium",
            f"BTC above MA20, QQQ above MA50, VIX {vix:.1f}. "
            "Moderate long with covered call overlay.",
            row,
        )

    # BTC bullish but QQQ soft
    if btc_above_ma20 and not qqq_above_ma50:
        plays = _options_advisory("btc_bull_qqq_soft", LONG_PILOT)
        return _signal(
            "btc_bull_qqq_soft", "BTC Up, QQQ Soft -- Pilot Long",
            "LONG", LONG_PILOT, plays, "medium",
            "BTC trend positive but QQQ below MA50. "
            "Pilot long only; sell cash-secured puts for more shares on a dip.",
            row,
        )

    # BTC bearish but QQQ firm
    if not btc_above_ma20 and qqq_above_ma50:
        plays = _options_advisory("btc_bear_qqq_firm", FLAT)
        return _signal(
            "btc_bear_qqq_firm", "BTC Weak, QQQ Firm -- Flat",
            "FLAT", FLAT, plays, "medium",
            "Conflicting signals: BTC (HIVE's primary driver) is below MA20 but QQQ holds. "
            "Stay flat; run iron condor if IV is high.",
            row,
        )

    # Both signals bearish
    if vix >= VIX_ELEVATED or oversold:
        regime = "bear_panic" if vix >= VIX_ELEVATED else "bear_oversold"
        label  = "Bear + High VIX -- Put Spread Only" if vix >= VIX_ELEVATED else "Bear but Oversold -- Reduce Short"
        equity_alloc = SHORT_LIGHT if vix < VIX_ELEVATED else FLAT
        plays = _options_advisory(regime, equity_alloc)
        return _signal(
            regime, label,
            "SHORT" if equity_alloc < 0 else "FLAT", equity_alloc, plays, "medium",
            (
                f"BTC below MA20, QQQ below MA50. "
                f"{'VIX '+str(round(vix,1))+' elevated: squeeze risk too high for full short. ' if vix >= VIX_ELEVATED else ''}"
                f"{'HIVE RSI '+str(round(hive_rsi,0))+' oversold: bounce risk. ' if oversold else ''}"
                "Use put spread for defined-risk bearish exposure."
            ),
            row,
        )

    equity_alloc = SHORT_FULL if vix < VIX_CAUTION else SHORT_MODERATE
    plays = _options_advisory("bear", equity_alloc)
    return _signal(
        "bear", "Bear -- Short Equity",
        "SHORT", equity_alloc, plays, "medium",
        f"BTC below MA20, QQQ below MA50, VIX {vix:.1f}. "
        f"Short HIVE {abs(equity_alloc):.0%} with covered put + call-spread stop.",
        row,
    )
