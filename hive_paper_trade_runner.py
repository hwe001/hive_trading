#!/usr/bin/env python3
"""
Paper-trading runner for the HIVE long/short equity strategy.

Executes HIVE equity orders (long or short) on an Alpaca paper account.
Options overlay signals are printed as advisory -- Alpaca paper does not support options.

Secrets required:
  HIVE_ALPACA_API_KEY_ID
  HIVE_ALPACA_API_SECRET_KEY
  HIVE_EXECUTE_ORDERS  (set to "true" in GitHub Actions secrets to submit orders)

Pass --execute to submit paper orders. Default is dry-run only.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

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

REBALANCE_START_NY = time(15, 45)
REBALANCE_END_NY   = time(15, 59)


def load_credentials() -> tuple[str, str]:
    key_id     = os.getenv("HIVE_ALPACA_API_KEY_ID", "")     or os.getenv("ALPACA_API_KEY_ID", "")
    secret_key = os.getenv("HIVE_ALPACA_API_SECRET_KEY", "") or os.getenv("ALPACA_API_SECRET_KEY", "")
    if not key_id or not secret_key:
        sys.exit(
            "Missing Alpaca credentials. Set HIVE_ALPACA_API_KEY_ID and "
            "HIVE_ALPACA_API_SECRET_KEY as environment variables or GitHub Actions secrets."
        )
    return key_id, secret_key


def fetch_history(symbol: str, period: str = "1y") -> pd.Series:
    hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if hist.empty:
        sys.exit(f"No price data returned for {symbol}.")
    return hist["Close"]


def build_signal_row() -> pd.Series:
    hive = fetch_history(HIVE_SYMBOL)
    btc  = fetch_history(BTC_SYMBOL)
    qqq  = fetch_history(QQQ_SYMBOL)
    vix  = fetch_history(VIX_SYMBOL)

    df = pd.DataFrame({"HIVE": hive, "BTC": btc, "QQQ": qqq, "VIX": vix}).sort_index()
    df = df[df["QQQ"].notna()].ffill().dropna()

    df["HIVE_MA5"]    = df["HIVE"].rolling(HIVE_MA_FAST).mean()
    df["HIVE_MA20"]   = df["HIVE"].rolling(HIVE_MA_SLOW).mean()
    df["HIVE_RSI14"]  = rsi(df["HIVE"], HIVE_RSI_PERIOD)
    df["BTC_MA20"]    = df["BTC"].rolling(BTC_MA_PERIOD).mean()
    df["QQQ_MA50"]    = df["QQQ"].rolling(QQQ_MA_PERIOD).mean()

    return df.dropna().iloc[-1]


def get_current_qty(trading_client: TradingClient, symbol: str) -> float:
    try:
        pos = trading_client.get_open_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0


def check_shortable(trading_client: TradingClient, symbol: str) -> bool:
    asset = trading_client.get_asset(symbol)
    return bool(asset.tradable and asset.shortable and asset.easy_to_borrow)


def cancel_open_orders(trading_client: TradingClient, symbol: str) -> None:
    for order in trading_client.get_orders():
        if getattr(order, "symbol", None) == symbol:
            print(f"Canceling open order {order.id} for {order.symbol}")
            trading_client.cancel_order_by_id(order.id)


def get_latest_price(data_client: StockHistoricalDataClient, symbol: str) -> float:
    request = StockLatestTradeRequest(symbol_or_symbols=symbol)
    trades = data_client.get_stock_latest_trade(request)
    return float(trades[symbol].price)


def compute_order(target_qty: int, current_qty: float) -> dict | None:
    delta = target_qty - current_qty
    if abs(delta) < 1:
        return None
    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
    return {"side": side, "qty": int(abs(round(delta)))}


def should_run_now(force_run: bool) -> bool:
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    print(f"New York time: {now_ny:%Y-%m-%d %H:%M:%S}")
    if force_run:
        print("--force-run: bypassing market-time gate.")
        return True
    if now_ny.weekday() >= 5:
        print("Outside weekday trading schedule. Skipping.")
        return False
    if REBALANCE_START_NY <= now_ny.time() <= REBALANCE_END_NY:
        return True
    print("Outside 3:45-3:59 PM New York rebalance window. Skipping.")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",            action="store_true", help="Submit Alpaca paper orders.")
    parser.add_argument("--force-run",          action="store_true", help="Bypass the 3:45 PM NY time gate.")
    parser.add_argument("--cancel-open-orders", action="store_true", help="Cancel existing open HIVE orders first.")
    args = parser.parse_args()

    if not should_run_now(args.force_run):
        return

    key_id, secret_key = load_credentials()
    trading_client = TradingClient(key_id, secret_key, paper=True)
    data_client    = StockHistoricalDataClient(key_id, secret_key)

    account = trading_client.get_account()
    equity  = float(account.equity)

    if args.cancel_open_orders and args.execute:
        cancel_open_orders(trading_client, HIVE_SYMBOL)

    row = build_signal_row()
    sig = classify_signal(row)

    print("=== HIVE Long / Short Paper Runner ===")
    print(f"Execute: {args.execute}")
    print(f"Regime:  {sig['regime']} ({sig['regime_description']})")
    print(f"Signal:  {sig['equity_action']}  target alloc {sig['equity_target_alloc']:+.0%}")
    print(f"Reason:  {sig['reason']}")
    print()
    print("Signals (read-only):")
    print(f"  BTC:  ${sig['btc']:>10,.0f}  MA20 ${sig['btc_ma20']:,.0f}  {'UP' if sig['btc_above_ma20'] else 'DOWN'}")
    print(f"  QQQ:  ${sig['qqq']:>10.2f}  MA50 ${sig['qqq_ma50']:.2f}  {'UP' if sig['qqq_above_ma50'] else 'DOWN'}")
    print(f"  VIX:   {sig['vix']:.2f}")
    print(f"  HIVE: ${sig['hive']:>10.4f}  MA20 ${sig['hive_ma20']:.4f}  RSI {sig['hive_rsi14']:.0f}")
    print(f"  Account equity: ${equity:,.2f}")
    print()
    print("[OPTIONS ADVISORY -- not executed]")
    for play in sig["options_plays"]:
        print(f"  ({play['priority']}) {play['strategy']}")
        print(f"     {play['detail']}")
    print()

    hive_shortable = check_shortable(trading_client, HIVE_SYMBOL)
    hive_price     = get_latest_price(data_client, HIVE_SYMBOL)

    target_alloc = sig["equity_target_alloc"]

    if target_alloc < 0 and not hive_shortable:
        print(f"WARNING: {HIVE_SYMBOL} is not currently shortable/easy-to-borrow on this account.")
        print("Falling back to FLAT for this cycle.")
        target_alloc = 0.0

    if target_alloc >= 0:
        target_qty = math.floor(equity * target_alloc / hive_price)
    else:
        target_qty = -math.floor(equity * abs(target_alloc) / hive_price)

    current_qty = get_current_qty(trading_client, HIVE_SYMBOL)
    order = compute_order(target_qty, current_qty)

    direction = "LONG" if target_qty > 0 else ("SHORT" if target_qty < 0 else "FLAT")
    print(f"{'Symbol':6s} {'Price':>10s} {'Current Qty':>12s} {'Target Qty':>12s} {'Direction':>10s}")
    print(
        f"{HIVE_SYMBOL:6s} {hive_price:10.4f} {current_qty:12.0f} "
        f"{target_qty:12.0f} {direction:>10s}"
    )

    if order is None:
        print("\nAlready at target. No orders needed.")
        return

    order_desc = f"{order['side'].value.upper()} {order['qty']}"
    print(f"Proposed order: {order_desc} {HIVE_SYMBOL}")

    if not args.execute:
        print("\nDry run only. Pass --execute or set HIVE_EXECUTE_ORDERS=true to submit paper orders.")
        return

    request = MarketOrderRequest(
        symbol=HIVE_SYMBOL,
        qty=order["qty"],
        side=order["side"],
        time_in_force=TimeInForce.DAY,
    )
    result = trading_client.submit_order(request)
    print(f"Submitted: {order_desc} {HIVE_SYMBOL} | Status: {result.status}")


if __name__ == "__main__":
    main()
