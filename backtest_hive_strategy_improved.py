#!/usr/bin/env python3
"""Research backtest for an improved HIVE BTC + AI/datacenter strategy."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

os.environ.setdefault("YFINANCE_CACHE_DIR", str(Path(".yfinance_cache").resolve()))

import yfinance as yf

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
    classify_signal,
    rsi,
)


INITIAL_EQUITY = 10_000.0
TRADING_DAYS = 252
TURNOVER_COST = 0.001
AI_SYMBOLS = ["NVDA", "AVGO", "AMD", "MSFT", "GOOGL", "META", "TSM"]

yf.set_tz_cache_location(str(Path(".yfinance_cache").resolve()))


@dataclass
class Metrics:
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    final_equity: float


def fetch_history(symbol: str, period: str = "5y") -> pd.Series:
    hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if hist.empty:
        raise RuntimeError(f"No price data returned for {symbol}")
    close = hist["Close"].rename(symbol.replace("^", "").replace("-", "_"))
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close


def build_data() -> pd.DataFrame:
    series = {
        "HIVE": fetch_history(HIVE_SYMBOL),
        "BTC": fetch_history(BTC_SYMBOL),
        "QQQ": fetch_history(QQQ_SYMBOL),
        "VIX": fetch_history(VIX_SYMBOL),
    }
    for symbol in AI_SYMBOLS:
        series[symbol] = fetch_history(symbol)

    df = pd.DataFrame(series).sort_index()
    df = df[df["QQQ"].notna()].ffill().dropna()

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


def original_alloc(row: pd.Series) -> tuple[float, str]:
    sig = classify_signal(row)
    return float(sig["equity_target_alloc"]), sig["regime"]


def improved_alloc(row: pd.Series) -> tuple[float, str]:
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

    return float(alloc), regime


def simulate(df: pd.DataFrame, signal_fn, name: str) -> pd.DataFrame:
    out = df.copy()
    decisions = [signal_fn(row) for _, row in out.iterrows()]
    out[f"{name}_target_alloc"] = [x[0] for x in decisions]
    out[f"{name}_regime"] = [x[1] for x in decisions]
    out["hive_return"] = out["HIVE"].pct_change().fillna(0.0)
    out[f"{name}_position"] = out[f"{name}_target_alloc"].shift(1).fillna(0.0)
    out[f"{name}_turnover"] = out[f"{name}_position"].diff().abs().fillna(0.0)
    out[f"{name}_return"] = out[f"{name}_position"] * out["hive_return"] - out[f"{name}_turnover"] * TURNOVER_COST
    out[f"{name}_equity"] = INITIAL_EQUITY * (1.0 + out[f"{name}_return"]).cumprod()
    return out


def buy_hold_equity(price: pd.Series) -> pd.Series:
    returns = price.pct_change().fillna(0.0)
    return INITIAL_EQUITY * (1.0 + returns).cumprod()


def metrics(returns: pd.Series, equity: pd.Series) -> Metrics:
    years = len(returns) / TRADING_DAYS
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
    volatility = returns.std() * math.sqrt(TRADING_DAYS)
    sharpe = (returns.mean() / returns.std() * math.sqrt(TRADING_DAYS)) if returns.std() else 0.0
    max_drawdown = (equity / equity.cummax() - 1.0).min()
    return Metrics(total_return, cagr, volatility, sharpe, max_drawdown, equity.iloc[-1])


def fmt_pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def metric_row(label: str, m: Metrics) -> str:
    return (
        f"| {label} | ${m.final_equity:,.2f} | {fmt_pct(m.total_return)} | "
        f"{fmt_pct(m.cagr)} | {fmt_pct(m.volatility)} | {m.sharpe:.2f} | {fmt_pct(m.max_drawdown)} |"
    )


def main() -> None:
    df = build_data()
    original = simulate(df, original_alloc, "original")
    improved = simulate(original, improved_alloc, "improved")

    improved["hive_hold_equity"] = buy_hold_equity(improved["HIVE"])
    improved["qqq_hold_equity"] = buy_hold_equity(improved["QQQ"])

    original_metrics = metrics(improved["original_return"], improved["original_equity"])
    improved_metrics = metrics(improved["improved_return"], improved["improved_equity"])
    hive_metrics = metrics(improved["HIVE"].pct_change().fillna(0.0), improved["hive_hold_equity"])
    qqq_metrics = metrics(improved["QQQ"].pct_change().fillna(0.0), improved["qqq_hold_equity"])

    out_dir = Path("..") / ".." / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hive_strategy_improved_5y_daily.csv"
    md_path = out_dir / "hive_strategy_improved_5y_report.md"
    improved.to_csv(csv_path, index_label="date")

    improved_trades = int((improved["improved_position"].diff().abs().fillna(0.0) > 0).sum())
    original_trades = int((improved["original_position"].diff().abs().fillna(0.0) > 0).sum())
    improved_counts = improved["improved_regime"].value_counts()

    regime_lines = ["| Regime | Days |", "|---|---:|"]
    for regime, days in improved_counts.items():
        regime_lines.append(f"| {regime} | {int(days)} |")

    report = f"""# Improved HIVE Strategy 5-Year Backtest

Period: {improved.index[0].date()} to {improved.index[-1].date()}

Improvement tested:

- HIVE remains the only traded asset.
- BTC is a separate crypto-mining theme vote.
- Equal-weight AI/datacenter basket is a separate theme vote: {", ".join(AI_SYMBOLS)}.
- QQQ remains the broad risk-on filter.
- HIVE's own MA20/MA50 trend gates position size.
- VIX and HIVE 20-day realized volatility scale exposure down.
- A small short is allowed only when HIVE, BTC, AI, and QQQ all confirm bearish.

Assumptions:

- Daily close-to-close simulation.
- Signal generated from today's completed row is applied to the next day's HIVE return.
- Transaction/slippage cost: {TURNOVER_COST * 10000:.0f} bps per 100% notional allocation changed.
- Initial equity: ${INITIAL_EQUITY:,.0f}

## Results

| Strategy | Final Equity | Total Return | CAGR | Ann. Vol | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|---:|---:|
{metric_row("Improved HIVE BTC + AI", improved_metrics)}
{metric_row("Original HIVE Strategy", original_metrics)}
{metric_row("Buy & Hold QQQ", qqq_metrics)}
{metric_row("Buy & Hold HIVE", hive_metrics)}

## Behavior

- Improved average net exposure: {fmt_pct(improved["improved_position"].mean())}
- Improved average gross exposure: {fmt_pct(improved["improved_position"].abs().mean())}
- Improved allocation changes: {improved_trades}
- Original allocation changes: {original_trades}

## Improved Regime Counts

{chr(10).join(regime_lines)}
"""
    md_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
