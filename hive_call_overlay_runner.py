#!/usr/bin/env python3
"""
Paper runner for the HIVE long-call overlay experiment.

The experiment maps to the proxy backtest case:
  Call overlay, IV 1.25x RV, 5% option premium sleeve.

Live paper trading uses actual Alpaca option quotes/contracts. The 1.25x RV
backtest assumption is retained as a logging/fair-value reference, not a fill.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    AssetStatus,
    ContractType,
    OrderSide,
    PositionIntent,
    TimeInForce,
)
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest

from hive_improved_strategy import (
    AI_SYMBOLS,
    add_improved_indicators,
    classify_improved_signal,
)
from hive_strategy import BTC_SYMBOL, HIVE_SYMBOL, QQQ_SYMBOL, VIX_SYMBOL


REBALANCE_START_NY = time(15, 40)
REBALANCE_END_NY = time(15, 59)
OPTION_DTE_TARGET = 45
OPTION_DTE_MIN = 35
OPTION_DTE_MAX = 60
EXIT_DTE = 5
OPTION_PREMIUM_BUDGET = 0.05
THEORETICAL_IV_MULTIPLIER = 1.25
MIN_CONTRACTS = 1
ORDER_LIMIT_BUFFER = 0.02


def load_credentials() -> tuple[str, str]:
    key_id = os.getenv("HIVE_ALPACA_API_KEY_ID", "") or os.getenv("ALPACA_API_KEY_ID", "")
    secret_key = os.getenv("HIVE_ALPACA_API_SECRET_KEY", "") or os.getenv("ALPACA_API_SECRET_KEY", "")
    if not key_id or not secret_key:
        sys.exit(
            "Missing Alpaca credentials. Set HIVE_ALPACA_API_KEY_ID and "
            "HIVE_ALPACA_API_SECRET_KEY as environment variables or GitHub Actions secrets."
        )
    return key_id, secret_key


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
    print("Outside 3:40-3:59 PM New York options rebalance window. Skipping.")
    return False


def fetch_history(symbol: str, period: str = "1y") -> pd.Series:
    hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if hist.empty:
        sys.exit(f"No price data returned for {symbol}.")
    close = hist["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close


def build_signal_row() -> pd.Series:
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
    df = add_improved_indicators(df)
    return df.iloc[-1]


def get_latest_stock_price(data_client: StockHistoricalDataClient, symbol: str) -> float:
    request = StockLatestTradeRequest(symbol_or_symbols=symbol)
    trades = data_client.get_stock_latest_trade(request)
    return float(trades[symbol].price)


def get_quote_mid(option_client: OptionHistoricalDataClient, symbol: str) -> tuple[float, float, float]:
    request = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
    quotes = option_client.get_option_latest_quote(request)
    quote = quotes[symbol]
    bid = float(getattr(quote, "bid_price", 0.0) or 0.0)
    ask = float(getattr(quote, "ask_price", 0.0) or 0.0)
    if bid <= 0 and ask <= 0:
        raise RuntimeError(f"No usable bid/ask quote for {symbol}.")
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else max(bid, ask)
    return bid, ask, mid


def get_option_contracts(trading_client: TradingClient, underlying: str, today: date) -> list:
    request = GetOptionContractsRequest(
        underlying_symbols=[underlying],
        status=AssetStatus.ACTIVE,
        type=ContractType.CALL,
        expiration_date_gte=today + timedelta(days=OPTION_DTE_MIN),
        expiration_date_lte=today + timedelta(days=OPTION_DTE_MAX),
        limit=1000,
    )
    response = trading_client.get_option_contracts(request)
    contracts = getattr(response, "option_contracts", None)
    if contracts is None and isinstance(response, dict):
        contracts = response.get("option_contracts")
    return list(contracts or [])


def choose_atm_call(contracts: list, stock_price: float, today: date):
    tradable = [c for c in contracts if bool(getattr(c, "tradable", False))]
    if not tradable:
        raise RuntimeError("No tradable HIVE call contracts found in target DTE window.")

    def score(contract) -> tuple[float, float]:
        strike = float(getattr(contract, "strike_price"))
        expiration = getattr(contract, "expiration_date")
        dte = abs((expiration - today).days - OPTION_DTE_TARGET)
        moneyness = abs(strike - stock_price)
        return (dte, moneyness)

    return min(tradable, key=score)


def is_hive_option_position(position) -> bool:
    asset_class = str(getattr(position, "asset_class", "")).lower()
    symbol = str(getattr(position, "symbol", ""))
    return "option" in asset_class and symbol.startswith(HIVE_SYMBOL)


def hive_option_positions(trading_client: TradingClient) -> list:
    return [pos for pos in trading_client.get_all_positions() if is_hive_option_position(pos)]


def position_dte(trading_client: TradingClient, symbol: str, today: date) -> int | None:
    try:
        contract = trading_client.get_option_contract(symbol)
        return (contract.expiration_date - today).days
    except Exception:
        return None


def submit_limit_order(
    trading_client: TradingClient,
    symbol: str,
    qty: int,
    side: OrderSide,
    limit_price: float,
    intent: PositionIntent,
    execute: bool,
) -> None:
    print(
        f"{'Submitting' if execute else 'Dry-run'} {side.value.upper()} {qty} {symbol} "
        f"limit ${limit_price:.2f} ({intent.value})"
    )
    if not execute:
        return
    request = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        limit_price=round(limit_price, 2),
        position_intent=intent,
    )
    result = trading_client.submit_order(request)
    print(f"Submitted order {result.id} | status: {result.status}")


def close_existing_hive_options(
    trading_client: TradingClient,
    option_client: OptionHistoricalDataClient,
    execute: bool,
    reason: str,
) -> None:
    positions = hive_option_positions(trading_client)
    if not positions:
        print("No open HIVE option positions to close.")
        return
    print(f"Closing HIVE option positions: {reason}")
    for pos in positions:
        symbol = pos.symbol
        qty = int(abs(float(pos.qty)))
        bid, ask, mid = get_quote_mid(option_client, symbol)
        limit_price = max(0.01, bid if bid > 0 else mid * (1.0 - ORDER_LIMIT_BUFFER))
        submit_limit_order(
            trading_client,
            symbol,
            qty,
            OrderSide.SELL,
            limit_price,
            PositionIntent.SELL_TO_CLOSE,
            execute,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Submit Alpaca paper option orders.")
    parser.add_argument("--force-run", action="store_true", help="Bypass the 3:40 PM NY time gate.")
    args = parser.parse_args()

    if not should_run_now(args.force_run):
        return

    key_id, secret_key = load_credentials()
    trading_client = TradingClient(key_id, secret_key, paper=True)
    stock_client = StockHistoricalDataClient(key_id, secret_key)
    option_client = OptionHistoricalDataClient(key_id, secret_key)

    account = trading_client.get_account()
    equity = float(account.equity)
    today = datetime.now(ZoneInfo("America/New_York")).date()

    row = build_signal_row()
    signal = classify_improved_signal(row)
    latest_hive = get_latest_stock_price(stock_client, HIVE_SYMBOL)

    print("=== HIVE Call Overlay Experiment ===")
    print(f"Execute: {args.execute}")
    print(f"Regime: {signal['regime']}")
    print(f"Target equity alloc: {signal['equity_target_alloc']:+.2%}")
    print(f"Call overlay active: {signal['option_call_overlay']}")
    print(f"HIVE latest: ${latest_hive:.2f} | signal close: ${signal['hive']:.2f}")
    print(f"HIVE RV20: {signal['hive_vol20']:.2%} | proxy IV ref: {signal['hive_vol20'] * THEORETICAL_IV_MULTIPLIER:.2%}")
    print(f"Theme votes: BTC={signal['btc_theme_bull']} AI={signal['ai_theme_bull']} QQQ={signal['qqq_risk_on']}")
    print(f"Account equity: ${equity:,.2f}")

    existing = hive_option_positions(trading_client)
    stale_positions = []
    for pos in existing:
        dte = position_dte(trading_client, pos.symbol, today)
        if dte is not None and dte <= EXIT_DTE:
            stale_positions.append(pos.symbol)

    if stale_positions:
        close_existing_hive_options(
            trading_client,
            option_client,
            args.execute,
            f"DTE <= {EXIT_DTE}: {', '.join(stale_positions)}",
        )
        return

    if not signal["option_call_overlay"]:
        close_existing_hive_options(trading_client, option_client, args.execute, "call overlay signal is off")
        return

    if existing:
        print("Existing HIVE option position found; no new call opened this cycle.")
        for pos in existing:
            print(f"  {pos.symbol}: qty {pos.qty}, market value {getattr(pos, 'market_value', 'n/a')}")
        return

    contracts = get_option_contracts(trading_client, HIVE_SYMBOL, today)
    contract = choose_atm_call(contracts, latest_hive, today)
    symbol = contract.symbol
    bid, ask, mid = get_quote_mid(option_client, symbol)
    entry_price = ask if ask > 0 else mid * (1.0 + ORDER_LIMIT_BUFFER)
    premium_budget = equity * OPTION_PREMIUM_BUDGET
    contract_notional = entry_price * 100.0
    qty = math.floor(premium_budget / contract_notional)

    print("Selected call:")
    print(f"  Symbol: {symbol}")
    print(f"  Expiry: {contract.expiration_date} | Strike: ${float(contract.strike_price):.2f}")
    print(f"  Bid/Ask/Mid: ${bid:.2f} / ${ask:.2f} / ${mid:.2f}")
    print(f"  Premium budget: ${premium_budget:,.2f} ({OPTION_PREMIUM_BUDGET:.0%} of equity)")

    if qty < MIN_CONTRACTS:
        print(f"Premium budget is too small for 1 contract at ${entry_price:.2f}. No order.")
        return

    submit_limit_order(
        trading_client,
        symbol,
        qty,
        OrderSide.BUY,
        entry_price,
        PositionIntent.BUY_TO_OPEN,
        args.execute,
    )


if __name__ == "__main__":
    main()
