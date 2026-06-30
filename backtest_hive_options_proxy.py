#!/usr/bin/env python3
"""Proxy options-overlay backtest for the improved HIVE strategy.

This uses synthetic Black-Scholes marks from HIVE price and realized volatility.
It is not a replacement for a true historical option-chain backtest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


INITIAL_EQUITY = 10_000.0
TRADING_DAYS = 252
RISK_FREE_RATE = 0.04
OPTION_DTE = 45
EXIT_DTE = 5
OPTION_BUDGET = 0.05
OPTION_SPREAD_COST = 0.02  # 2% one-way option mark friction.
MIN_IV = 0.80
MAX_IV = 2.50


@dataclass
class OptionPosition:
    kind: str
    strike: float
    short_strike: float | None
    units: float
    dte: int
    price: float


@dataclass
class Metrics:
    final_equity: float
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(s: float, k: float, t: float, sigma: float, r: float = RISK_FREE_RATE) -> float:
    if t <= 0:
        return max(s - k, 0.0)
    sigma = max(sigma, 0.01)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return s * norm_cdf(d1) - k * math.exp(-r * t) * norm_cdf(d2)


def option_mark(kind: str, s: float, strike: float, short_strike: float | None, dte: int, iv: float) -> float:
    t = max(dte, 0) / TRADING_DAYS
    if kind == "call":
        return bs_call(s, strike, t, iv)
    if kind == "call_spread":
        assert short_strike is not None
        return max(bs_call(s, strike, t, iv) - bs_call(s, short_strike, t, iv), 0.0)
    raise ValueError(kind)


def metrics(equity: pd.Series, returns: pd.Series) -> Metrics:
    years = len(returns) / TRADING_DAYS
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
    vol = returns.std() * math.sqrt(TRADING_DAYS)
    sharpe = returns.mean() / returns.std() * math.sqrt(TRADING_DAYS) if returns.std() else 0.0
    dd = (equity / equity.cummax() - 1.0).min()
    return Metrics(equity.iloc[-1], total, cagr, vol, sharpe, dd)


def fmt_pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def run_overlay(df: pd.DataFrame, kind: str, iv_multiplier: float) -> pd.DataFrame:
    nav = INITIAL_EQUITY
    pos: OptionPosition | None = None
    rows = []

    for _, row in df.iterrows():
        start_nav = nav
        nav *= 1.0 + float(row["improved_return"])

        iv = max(MIN_IV, min(MAX_IV, float(row["HIVE_VOL20"]) * iv_multiplier))
        s = float(row["HIVE"])
        long_signal = float(row["improved_target_alloc"]) >= 0.45

        option_pnl = 0.0
        trade_cost = 0.0

        if pos is not None:
            pos.dte -= 1
            new_price = option_mark(pos.kind, s, pos.strike, pos.short_strike, pos.dte, iv)
            option_pnl = pos.units * (new_price - pos.price)
            nav += option_pnl
            pos.price = new_price

            if (not long_signal) or pos.dte <= EXIT_DTE:
                trade_cost = abs(pos.units * pos.price) * OPTION_SPREAD_COST
                nav -= trade_cost
                pos = None

        if pos is None and long_signal:
            strike = s
            short_strike = s * 1.20 if kind == "call_spread" else None
            price = option_mark(kind, s, strike, short_strike, OPTION_DTE, iv)
            if price > 0:
                premium = nav * OPTION_BUDGET
                units = premium / price
                trade_cost = premium * OPTION_SPREAD_COST
                nav -= trade_cost
                pos = OptionPosition(kind, strike, short_strike, units, OPTION_DTE, price)

        rows.append({
            "date": row["date"],
            "equity": nav,
            "return": nav / start_nav - 1.0 if start_nav else 0.0,
            "option_pnl": option_pnl,
            "option_trade_cost": trade_cost,
            "has_option": pos is not None,
            "iv": iv,
        })

    return pd.DataFrame(rows)


def row(label: str, m: Metrics) -> str:
    return (
        f"| {label} | ${m.final_equity:,.2f} | {fmt_pct(m.total_return)} | "
        f"{fmt_pct(m.cagr)} | {fmt_pct(m.volatility)} | {m.sharpe:.2f} | {fmt_pct(m.max_drawdown)} |"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "outputs" / "hive_strategy_improved_5y_daily.csv")
    base_equity = df["improved_equity"]
    base_returns = df["improved_return"]
    qqq_equity = df["qqq_hold_equity"]
    qqq_returns = df["QQQ"].pct_change().fillna(0.0)

    report_rows = [
        row("Improved equity only", metrics(base_equity, base_returns)),
        row("Buy & hold QQQ", metrics(qqq_equity, qqq_returns)),
    ]

    all_daily = df[["date", "HIVE", "improved_target_alloc", "improved_return", "improved_equity", "qqq_hold_equity"]].copy()

    for iv_mult in (1.00, 1.25, 1.50):
        for kind in ("call", "call_spread"):
            result = run_overlay(df, kind, iv_mult)
            label = f"{kind.replace('_', ' ').title()} overlay, IV {iv_mult:.2f}x RV"
            report_rows.append(row(label, metrics(result["equity"], result["return"])))
            all_daily[f"{kind}_iv_{iv_mult:.2f}_equity"] = result["equity"]

    out_dir = root / "outputs"
    csv_path = out_dir / "hive_options_proxy_5y_daily.csv"
    md_path = out_dir / "hive_options_proxy_5y_report.md"
    all_daily.to_csv(csv_path, index=False)

    report = f"""# HIVE Options Overlay Proxy Backtest

Period: {df['date'].iloc[0]} to {df['date'].iloc[-1]}

This is a proxy test, not a historical option-chain backtest.

Assumptions:

- Uses the improved HIVE BTC + AI equity strategy as the base.
- Adds an options sleeve only during long regimes with target allocation >= 45%.
- Option sleeve risks 5% of current equity per entry.
- Synthetic marks use Black-Scholes, 45 trading-day expiry, ATM long call or ATM/20%-OTM call spread.
- Options exit when the long signal disappears or DTE falls to {EXIT_DTE}.
- Implied volatility is modeled as a multiple of HIVE 20-day realized volatility, clipped to {MIN_IV:.0%}-{MAX_IV:.0%}.
- Option friction is {OPTION_SPREAD_COST:.0%} one-way to approximate bid/ask/slippage.

## Results

| Strategy | Final Equity | Total Return | CAGR | Ann. Vol | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(report_rows)}

## Read

Long calls add the most upside only when realized movement is not already fully priced into implied volatility.
Call spreads are less explosive, but they reduce theta/IV overpayment and are more suitable for HIVE's expensive-volatility profile.
"""
    md_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
