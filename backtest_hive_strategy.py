#!/usr/bin/env python3
"""Five-year close-to-close backtest for the current HIVE strategy rules."""

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
YEARS = 5
TRADING_DAYS = 252
TURNOVER_COST = 0.001  # 10 bps per 100% notional allocation changed.

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
    hive = fetch_history(HIVE_SYMBOL)
    btc = fetch_history(BTC_SYMBOL)
    qqq = fetch_history(QQQ_SYMBOL)
    vix = fetch_history(VIX_SYMBOL)

    # Same alignment as the runner/dashboard: anchor to QQQ trading days and
    # forward-fill BTC/VIX/HIVE where needed.
    df = pd.DataFrame({"HIVE": hive, "BTC": btc, "QQQ": qqq, "VIX": vix}).sort_index()
    df = df[df["QQQ"].notna()].ffill().dropna()

    df["HIVE_MA5"] = df["HIVE"].rolling(HIVE_MA_FAST).mean()
    df["HIVE_MA20"] = df["HIVE"].rolling(HIVE_MA_SLOW).mean()
    df["HIVE_RSI14"] = rsi(df["HIVE"], HIVE_RSI_PERIOD)
    df["BTC_MA20"] = df["BTC"].rolling(BTC_MA_PERIOD).mean()
    df["QQQ_MA50"] = df["QQQ"].rolling(QQQ_MA_PERIOD).mean()
    return df.dropna()


def simulate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    signals = [classify_signal(row) for _, row in out.iterrows()]
    out["target_alloc"] = [sig["equity_target_alloc"] for sig in signals]
    out["regime"] = [sig["regime"] for sig in signals]

    # Signal is known at the close and applied to the next close-to-close HIVE
    # return. This avoids using the same close both for signal generation and PnL.
    out["hive_return"] = out["HIVE"].pct_change()
    out["position"] = out["target_alloc"].shift(1).fillna(0.0)
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["cost"] = out["turnover"] * TURNOVER_COST
    out["strategy_return"] = out["position"] * out["hive_return"] - out["cost"]
    out["strategy_return"] = out["strategy_return"].fillna(0.0)
    out["equity"] = INITIAL_EQUITY * (1.0 + out["strategy_return"]).cumprod()
    out["buy_hold_equity"] = INITIAL_EQUITY * (1.0 + out["hive_return"].fillna(0.0)).cumprod()
    return out


def metrics(returns: pd.Series, equity: pd.Series) -> Metrics:
    returns = returns.fillna(0.0)
    years = len(returns) / TRADING_DAYS
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
    volatility = returns.std() * math.sqrt(TRADING_DAYS)
    sharpe = (returns.mean() / returns.std() * math.sqrt(TRADING_DAYS)) if returns.std() else 0.0
    drawdown = equity / equity.cummax() - 1.0
    return Metrics(total_return, cagr, volatility, sharpe, drawdown.min(), equity.iloc[-1])


def fmt_pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def regime_table(counts: pd.Series) -> str:
    lines = ["| Regime | Days |", "|---|---:|"]
    for regime, days in counts.items():
        lines.append(f"| {regime} | {int(days)} |")
    return "\n".join(lines)


def main() -> None:
    df = simulate(build_data())
    strategy = metrics(df["strategy_return"], df["equity"])
    buy_hold_returns = df["HIVE"].pct_change().fillna(0.0)
    buy_hold = metrics(buy_hold_returns, df["buy_hold_equity"])

    counts = df["regime"].value_counts()
    exposure = df["position"].mean()
    gross_exposure = df["position"].abs().mean()
    trades = int((df["position"].diff().abs().fillna(0.0) > 0).sum())
    win_rate = (df.loc[df["strategy_return"] != 0, "strategy_return"] > 0).mean()

    out_dir = Path("..") / ".." / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hive_strategy_5y_backtest_daily.csv"
    md_path = out_dir / "hive_strategy_5y_backtest.md"

    df.to_csv(csv_path, index_label="date")

    report = f"""# HIVE Strategy 5-Year Backtest

Period: {df.index[0].date()} to {df.index[-1].date()}

Assumptions:

- Uses the current `alpaca-equity` branch strategy rules as-is.
- Daily close-to-close simulation.
- Signal generated from today's completed row is applied to the next day's HIVE return.
- HIVE is the only traded asset; BTC, QQQ, and VIX are signals only.
- Transaction/slippage cost: {TURNOVER_COST * 10000:.0f} bps per 100% notional allocation changed.
- Initial equity: ${INITIAL_EQUITY:,.0f}

## Results

| Metric | Strategy | Buy & Hold HIVE |
|---|---:|---:|
| Final equity | ${strategy.final_equity:,.2f} | ${buy_hold.final_equity:,.2f} |
| Total return | {fmt_pct(strategy.total_return)} | {fmt_pct(buy_hold.total_return)} |
| CAGR | {fmt_pct(strategy.cagr)} | {fmt_pct(buy_hold.cagr)} |
| Annual volatility | {fmt_pct(strategy.volatility)} | {fmt_pct(buy_hold.volatility)} |
| Sharpe, rf=0 | {strategy.sharpe:.2f} | {buy_hold.sharpe:.2f} |
| Max drawdown | {fmt_pct(strategy.max_drawdown)} | {fmt_pct(buy_hold.max_drawdown)} |

## Strategy Behavior

- Average net exposure: {fmt_pct(exposure)}
- Average gross exposure: {fmt_pct(gross_exposure)}
- Allocation changes: {trades}
- Non-zero daily win rate: {fmt_pct(float(win_rate)) if not math.isnan(float(win_rate)) else "n/a"}

## Regime Counts

{regime_table(counts)}
"""
    md_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
