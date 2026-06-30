#!/usr/bin/env python3
"""
HIVE Tradier Options + Equity Runner

Executes BOTH equity (long/short) AND options trades on a Tradier account.
Unlike the Alpaca runner, this runner places real options orders.

Strategy:
  Equity:  long / short / flat HIVE based on BTC/QQQ/VIX regime
  Options: regime-matched structure -- covered calls, puts, spreads, condors

Secrets (environment variables or GitHub Actions secrets):
  TRADIER_API_TOKEN       Tradier bearer token
  TRADIER_ACCOUNT_ID      Tradier account number
  TRADIER_EXECUTE_ORDERS  Set to "true" to submit orders (default: dry-run)
  TRADIER_USE_SANDBOX     "true" = paper sandbox (default), "false" = live

Flags:
  --execute               Submit orders (overrides TRADIER_EXECUTE_ORDERS)
  --force-run             Bypass 3:30 PM ET time gate
  --cancel-open-orders    Cancel existing open HIVE orders before running
  --close-options         Close all HIVE options positions before running
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

from hive_strategy import (
    BTC_MA_PERIOD, BTC_SYMBOL,
    HIVE_MA_FAST, HIVE_MA_SLOW, HIVE_RSI_PERIOD, HIVE_SYMBOL,
    QQQ_MA_PERIOD, QQQ_SYMBOL, VIX_SYMBOL,
    classify_signal, rsi,
)

REBALANCE_TIME_NY = datetime.strptime("15:30", "%H:%M").time()
REBALANCE_END_NY  = datetime.strptime("15:55", "%H:%M").time()

MIN_OI              = 5       # minimum open interest to consider an option
MAX_SPREAD_PCT      = 0.50    # reject options where spread/mid > 50%
MIN_BID             = 0.01    # reject options with zero bid
SLEEP_AFTER_EQUITY  = 10      # seconds to wait after equity order before options


# ---------------------------------------------------------------------------
# Tradier API client
# ---------------------------------------------------------------------------

class TradierClient:
    SANDBOX = "https://sandbox.tradier.com/v1"
    LIVE    = "https://api.tradier.com/v1"

    def __init__(self, token: str, account_id: str, sandbox: bool = True):
        self.base       = self.SANDBOX if sandbox else self.LIVE
        self.account_id = account_id
        self._headers   = {
            "Authorization": f"Bearer {token}",
            "Accept":        "application/json",
        }

    # -- core HTTP -----------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = requests.get(f"{self.base}{path}", headers=self._headers,
                         params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: dict) -> dict:
        r = requests.post(f"{self.base}{path}", headers=self._headers,
                          data=data, timeout=20)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> dict:
        r = requests.delete(f"{self.base}{path}", headers=self._headers, timeout=20)
        r.raise_for_status()
        return r.json()

    # -- account -------------------------------------------------------------

    def get_balances(self) -> dict:
        return self._get(f"/accounts/{self.account_id}/balances")["balances"]

    def get_positions(self) -> list[dict]:
        data = self._get(f"/accounts/{self.account_id}/positions")
        pos  = data.get("positions", {}).get("position", [])
        return pos if isinstance(pos, list) else [pos]

    def get_orders(self) -> list[dict]:
        data   = self._get(f"/accounts/{self.account_id}/orders")
        orders = data.get("orders", {}).get("order", [])
        return orders if isinstance(orders, list) else [orders]

    # -- market data ---------------------------------------------------------

    def get_quote(self, symbols: list[str]) -> dict:
        data  = self._get("/markets/quotes", {"symbols": ",".join(symbols)})
        quotes = data.get("quotes", {}).get("quote", [])
        if isinstance(quotes, dict):
            quotes = [quotes]
        return {q["symbol"]: q for q in quotes}

    def get_options_expirations(self, symbol: str) -> list[str]:
        data = self._get("/markets/options/expirations",
                         {"symbol": symbol, "includeAllRoots": "true"})
        exps = data.get("expirations", {}).get("date", [])
        return exps if isinstance(exps, list) else [exps]

    def get_options_chain(self, symbol: str, expiration: str) -> list[dict]:
        data    = self._get("/markets/options/chains",
                            {"symbol": symbol, "expiration": expiration, "greeks": "false"})
        options = data.get("options", {}).get("option", [])
        return options if isinstance(options, list) else [options]

    # -- order placement -----------------------------------------------------

    def place_equity_order(self, symbol: str, side: str, qty: int,
                           order_type: str = "market", price: float | None = None) -> dict:
        """
        side: "buy" | "sell" | "sell_short" | "buy_to_cover"
        """
        data: dict = {
            "class":    "equity",
            "symbol":   symbol,
            "side":     side,
            "quantity": str(qty),
            "type":     order_type,
            "duration": "day",
        }
        if price is not None:
            data["price"] = f"{price:.2f}"
        return self._post(f"/accounts/{self.account_id}/orders", data)

    def place_option_order(self, option_symbol: str, side: str, qty: int,
                           price: float | None = None) -> dict:
        """
        side: "buy_to_open" | "sell_to_open" | "buy_to_close" | "sell_to_close"
        Uses a limit order at provided price, or market if price is None.
        """
        underlying = re.match(r"^([A-Z]+)", option_symbol).group(1)
        order_type = "limit" if price is not None else "market"
        data: dict = {
            "class":         "option",
            "symbol":        underlying,
            "option_symbol": option_symbol,
            "side":          side,
            "quantity":      str(qty),
            "type":          order_type,
            "duration":      "day",
        }
        if price is not None:
            data["price"] = f"{price:.2f}"
        return self._post(f"/accounts/{self.account_id}/orders", data)

    def place_multileg_order(self, underlying: str, legs: list[dict],
                             net_type: str = "net_credit",
                             price: float | None = None) -> dict:
        """
        legs: [{"option_symbol": ..., "side": ..., "quantity": ...}, ...]
        net_type: "net_credit" | "net_debit" | "even"
        """
        data: dict = {
            "class":    "multileg",
            "symbol":   underlying,
            "type":     net_type,
            "duration": "day",
        }
        for i, leg in enumerate(legs):
            data[f"option_symbol[{i}]"] = leg["option_symbol"]
            data[f"side[{i}]"]          = leg["side"]
            data[f"quantity[{i}]"]      = str(leg["quantity"])
        if price is not None:
            data["price"] = f"{abs(price):.2f}"
        return self._post(f"/accounts/{self.account_id}/orders", data)

    def cancel_order(self, order_id: str | int) -> dict:
        return self._delete(f"/accounts/{self.account_id}/orders/{order_id}")


# ---------------------------------------------------------------------------
# Options selection helpers
# ---------------------------------------------------------------------------

def pick_expiration(expirations: list[str], min_dte: int, max_dte: int) -> str | None:
    """Return expiration closest to the midpoint of [min_dte, max_dte]."""
    today      = date.today()
    target_dte = (min_dte + max_dte) // 2
    candidates = []
    for exp_str in expirations:
        dte = (date.fromisoformat(exp_str) - today).days
        if min_dte <= dte <= max_dte:
            candidates.append((abs(dte - target_dte), exp_str))
    return sorted(candidates)[0][1] if candidates else None


def mid_price(opt: dict) -> float:
    bid = float(opt.get("bid") or 0)
    ask = float(opt.get("ask") or 0)
    return (bid + ask) / 2


def is_liquid(opt: dict) -> bool:
    bid = float(opt.get("bid") or 0)
    m   = mid_price(opt)
    oi  = int(opt.get("open_interest") or 0)
    if bid < MIN_BID or m <= 0 or oi < MIN_OI:
        return False
    spread = float(opt.get("ask") or 0) - bid
    return (spread / m) <= MAX_SPREAD_PCT


def pick_call(chain: list[dict], spot: float,
              pct_otm_min: float, pct_otm_max: float) -> dict | None:
    target  = spot * (1 + (pct_otm_min + pct_otm_max) / 2)
    calls   = [o for o in chain
               if o.get("option_type") == "call" and is_liquid(o)
               and pct_otm_min <= (float(o["strike"]) - spot) / spot <= pct_otm_max]
    if not calls:
        return None
    return min(calls, key=lambda o: abs(float(o["strike"]) - target))


def pick_put(chain: list[dict], spot: float,
             pct_otm_min: float, pct_otm_max: float) -> dict | None:
    target = spot * (1 - (pct_otm_min + pct_otm_max) / 2)
    puts   = [o for o in chain
              if o.get("option_type") == "put" and is_liquid(o)
              and pct_otm_min <= (spot - float(o["strike"])) / spot <= pct_otm_max]
    if not puts:
        return None
    return min(puts, key=lambda o: abs(float(o["strike"]) - target))


def pick_put_atm_or_otm(chain: list[dict], spot: float,
                         pct_otm: float) -> dict | None:
    """Pick put at or slightly OTM (for ATM leg of put spreads)."""
    target = spot * (1 - pct_otm)
    puts   = [o for o in chain
              if o.get("option_type") == "put" and is_liquid(o)
              and (spot - float(o["strike"])) / spot <= 0.08]  # within 8% OTM
    if not puts:
        return None
    return min(puts, key=lambda o: abs(float(o["strike"]) - target))


# ---------------------------------------------------------------------------
# Signal data
# ---------------------------------------------------------------------------

def fetch_history(symbol: str, period: str = "1y") -> pd.Series:
    hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if hist.empty:
        sys.exit(f"No price data for {symbol}")
    return hist["Close"]


def build_signal_row() -> pd.Series:
    hive = fetch_history(HIVE_SYMBOL)
    btc  = fetch_history(BTC_SYMBOL)
    qqq  = fetch_history(QQQ_SYMBOL)
    vix  = fetch_history(VIX_SYMBOL)

    df = pd.DataFrame({"HIVE": hive, "BTC": btc, "QQQ": qqq, "VIX": vix}).sort_index()
    df = df[df["QQQ"].notna()].ffill().dropna()

    df["HIVE_MA5"]   = df["HIVE"].rolling(HIVE_MA_FAST).mean()
    df["HIVE_MA20"]  = df["HIVE"].rolling(HIVE_MA_SLOW).mean()
    df["HIVE_RSI14"] = rsi(df["HIVE"], HIVE_RSI_PERIOD)
    df["BTC_MA20"]   = df["BTC"].rolling(BTC_MA_PERIOD).mean()
    df["QQQ_MA50"]   = df["QQQ"].rolling(QQQ_MA_PERIOD).mean()

    return df.dropna().iloc[-1]


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def parse_positions(positions: list[dict]) -> tuple[float, list[dict]]:
    """
    Returns (equity_qty, options_positions).
    equity_qty > 0 = long shares, < 0 = short shares.
    """
    equity_qty = 0.0
    options    = []
    for pos in positions:
        sym = pos.get("symbol", "")
        qty = float(pos.get("quantity", 0))
        # Options symbols contain digits and C/P after the root
        if re.search(r"\d{6}[CP]\d{8}$", sym):
            options.append(pos)
        elif sym == HIVE_SYMBOL:
            equity_qty = qty
    return equity_qty, options


def is_hive_option(sym: str) -> bool:
    return sym.startswith(HIVE_SYMBOL) and re.search(r"\d{6}[CP]\d{8}$", sym) is not None


def option_orientation(sym: str) -> str:
    """Return 'call' or 'put' from an OCC symbol."""
    m = re.search(r"\d{6}([CP])\d{8}$", sym)
    return "call" if m and m.group(1) == "C" else "put"


def options_compatible(existing: list[dict], regime: str) -> bool:
    """True if existing options positions are appropriate for the current regime."""
    if not existing:
        return True  # nothing to conflict

    bull_regimes = {"bull_strong", "bull", "bull_cautious", "btc_bull_qqq_soft"}
    bear_regimes = {"bear", "bear_oversold", "bear_panic"}
    flat_regimes = {"danger", "btc_bear_qqq_firm"}

    for pos in existing:
        sym = pos["symbol"]
        if not is_hive_option(sym):
            continue
        qty        = float(pos.get("quantity", 0))
        o_type     = option_orientation(sym)
        is_short   = qty < 0
        is_long    = qty > 0

        # Short calls are wrong in bear/flat regimes
        if o_type == "call" and is_short and regime in bear_regimes | flat_regimes:
            return False
        # Short puts are wrong in bull regimes
        if o_type == "put" and is_short and regime in bull_regimes:
            return False
        # Long puts are wrong in pure bull_strong (but OK in bull/cautious)
        if o_type == "put" and is_long and regime == "bull_strong":
            return False

    return True


# ---------------------------------------------------------------------------
# Equity management
# ---------------------------------------------------------------------------

def manage_equity(client: TradierClient, sig: dict, execute: bool) -> int:
    """
    Rebalance HIVE equity to the target allocation.
    Returns the target_qty after rebalancing (positive=long, negative=short).
    """
    balances    = client.get_balances()
    equity      = float(balances.get("total_equity", balances.get("equity", 10000)))
    positions   = client.get_positions()
    current_qty, _ = parse_positions(positions)

    quotes      = client.get_quote([HIVE_SYMBOL])
    spot        = float(quotes[HIVE_SYMBOL]["last"] or quotes[HIVE_SYMBOL]["close"])

    target_alloc = sig["equity_target_alloc"]
    if target_alloc >= 0:
        target_qty = math.floor(equity * target_alloc / spot)
    else:
        target_qty = -math.floor(equity * abs(target_alloc) / spot)

    delta = target_qty - current_qty

    print(f"[EQUITY] spot=${spot:.4f}  current={current_qty:.0f}  target={target_qty:.0f}  delta={delta:+.0f}")

    if abs(delta) < 1:
        print("[EQUITY] Already at target.")
        return int(target_qty)

    # Determine order side
    if delta > 0:
        side = "buy" if current_qty >= 0 else "buy_to_cover"
    else:
        side = "sell" if current_qty > 0 else "sell_short"

    print(f"[EQUITY] Order: {side.upper()} {abs(int(delta))} {HIVE_SYMBOL}")

    if execute:
        result = client.place_equity_order(HIVE_SYMBOL, side, abs(int(delta)))
        print(f"[EQUITY] Submitted: order_id={result.get('order', {}).get('id')}  "
              f"status={result.get('order', {}).get('status')}")
        print(f"         Waiting {SLEEP_AFTER_EQUITY}s for equity fill before options ...")
        time.sleep(SLEEP_AFTER_EQUITY)
    else:
        print("[EQUITY] Dry run -- not submitted.")

    return int(target_qty)


# ---------------------------------------------------------------------------
# Options management
# ---------------------------------------------------------------------------

def close_all_hive_options(client: TradierClient, positions: list[dict],
                            execute: bool) -> None:
    """Close every open HIVE options position at market."""
    for pos in positions:
        sym = pos["symbol"]
        if not is_hive_option(sym):
            continue
        qty      = float(pos.get("quantity", 0))
        close_side = "sell_to_close" if qty > 0 else "buy_to_close"
        print(f"[OPTIONS] Closing {sym}  qty={qty:.0f}  side={close_side}")
        if execute:
            result = client.place_option_order(sym, close_side, abs(int(qty)))
            print(f"          order_id={result.get('order', {}).get('id')}")
        else:
            print("          Dry run -- not submitted.")


def execute_options_plan(client: TradierClient, regime: str,
                          target_equity_qty: int, spot: float,
                          execute: bool) -> None:
    """
    Build and execute the options plan for the current regime.
    """
    n_contracts = max(1, abs(target_equity_qty) // 100)
    expirations = client.get_options_expirations(HIVE_SYMBOL)

    if not expirations:
        print("[OPTIONS] No expirations available -- skipping options.")
        return

    def chain_for(min_dte: int, max_dte: int) -> tuple[str | None, list[dict]]:
        exp = pick_expiration(expirations, min_dte, max_dte)
        if exp is None:
            return None, []
        return exp, client.get_options_chain(HIVE_SYMBOL, exp)

    def limit_price(opt: dict, selling: bool) -> float:
        bid = float(opt.get("bid") or 0)
        ask = float(opt.get("ask") or 0)
        m   = (bid + ask) / 2
        # Round to nearest cent; for sells, slightly below mid to get filled
        if selling:
            return max(bid, round(m * 0.98, 2))
        else:
            return min(ask, round(m * 1.02, 2))

    def _place(label: str, sym: str, side: str, qty: int, price: float) -> None:
        direction = "SELL" if "sell" in side else "BUY"
        print(f"[OPTIONS] {label}: {direction} {qty}x {sym} @ ${price:.2f} limit ({side})")
        if execute:
            result = client.place_option_order(sym, side, qty, price)
            print(f"          order_id={result.get('order', {}).get('id')}  "
                  f"status={result.get('order', {}).get('status')}")
        else:
            print("          Dry run -- not submitted.")

    def _place_multi(label: str, legs: list[dict], net_type: str, price: float) -> None:
        print(f"[OPTIONS] {label}: {net_type.upper()} net ${price:.2f}")
        for leg in legs:
            print(f"          {leg['side']:18s} {leg['quantity']}x {leg['option_symbol']}")
        if execute:
            result = client.place_multileg_order(HIVE_SYMBOL, legs, net_type, price)
            print(f"          order_id={result.get('order', {}).get('id')}  "
                  f"status={result.get('order', {}).get('status')}")
        else:
            print("          Dry run -- not submitted.")

    # ── regime-specific logic ────────────────────────────────────────────────

    if regime == "danger":
        print("[OPTIONS] DANGER regime -- options already closed above.")
        return

    elif regime == "bull_strong":
        # Sell covered call: 20-25% OTM, 30-45 DTE
        exp, chain = chain_for(30, 45)
        if not exp:
            print("[OPTIONS] bull_strong: no expiration in 30-45 DTE window.")
            return
        call = pick_call(chain, spot, 0.20, 0.25)
        if call:
            _place("covered call", call["symbol"], "sell_to_open",
                   n_contracts, limit_price(call, selling=True))
        else:
            print(f"[OPTIONS] bull_strong: no liquid call 20-25% OTM on {exp}.")

    elif regime == "bull":
        exp, chain = chain_for(25, 35)
        if not exp:
            print("[OPTIONS] bull: no expiration in 25-35 DTE window.")
            return
        call = pick_call(chain, spot, 0.15, 0.20)
        if call:
            _place("covered call", call["symbol"], "sell_to_open",
                   n_contracts, limit_price(call, selling=True))
        put = pick_put(chain, spot, 0.10, 0.15)
        if put:
            # Only buy protective put if cost < 1.5% of spot
            cost_pct = mid_price(put) / spot
            if cost_pct <= 0.015:
                _place("protective put", put["symbol"], "buy_to_open",
                       n_contracts, limit_price(put, selling=False))
            else:
                print(f"[OPTIONS] bull: protective put too expensive ({cost_pct:.1%} of spot) -- skipping.")

    elif regime == "bull_cautious":
        # Tight collar: sell 10% OTM call + buy 8-12% OTM put
        exp, chain = chain_for(25, 35)
        if not exp:
            print("[OPTIONS] bull_cautious: no expiration in 25-35 DTE window.")
            return
        call = pick_call(chain, spot, 0.08, 0.12)
        put  = pick_put(chain, spot, 0.08, 0.12)
        if call and put:
            call_credit = limit_price(call, selling=True)
            put_debit   = limit_price(put,  selling=False)
            net         = call_credit - put_debit  # positive = net credit
            legs = [
                {"option_symbol": call["symbol"], "side": "sell_to_open", "quantity": n_contracts},
                {"option_symbol": put["symbol"],  "side": "buy_to_open",  "quantity": n_contracts},
            ]
            net_type = "net_credit" if net > 0 else "net_debit"
            _place_multi("tight collar", legs, net_type, abs(net))
        else:
            if call:
                _place("collar call only", call["symbol"], "sell_to_open",
                       n_contracts, limit_price(call, selling=True))
            if put:
                _place("collar put only", put["symbol"], "buy_to_open",
                       n_contracts, limit_price(put, selling=False))

    elif regime == "btc_bull_qqq_soft":
        # Sell cash-secured put: 10-15% OTM, 30 DTE
        exp, chain = chain_for(25, 35)
        if not exp:
            print("[OPTIONS] btc_bull_qqq_soft: no expiration in 25-35 DTE window.")
            return
        put = pick_put(chain, spot, 0.10, 0.15)
        if put:
            _place("cash-secured put", put["symbol"], "sell_to_open",
                   n_contracts, limit_price(put, selling=True))
        else:
            print(f"[OPTIONS] btc_bull_qqq_soft: no liquid put 10-15% OTM on {exp}.")

    elif regime == "btc_bear_qqq_firm":
        # Iron condor: sell C20+P20 OTM, buy C30+P30 OTM (multi-leg)
        exp, chain = chain_for(30, 45)
        if not exp:
            print("[OPTIONS] btc_bear_qqq_firm: no expiration in 30-45 DTE window.")
            return
        short_call = pick_call(chain, spot, 0.18, 0.22)
        long_call  = pick_call(chain, spot, 0.28, 0.33)
        short_put  = pick_put(chain,  spot, 0.18, 0.22)
        long_put   = pick_put(chain,  spot, 0.28, 0.33)
        if all([short_call, long_call, short_put, long_put]):
            credit = (limit_price(short_call, selling=True) +
                      limit_price(short_put,  selling=True) -
                      limit_price(long_call,  selling=False) -
                      limit_price(long_put,   selling=False))
            legs = [
                {"option_symbol": short_call["symbol"], "side": "sell_to_open", "quantity": 1},
                {"option_symbol": long_call["symbol"],  "side": "buy_to_open",  "quantity": 1},
                {"option_symbol": short_put["symbol"],  "side": "sell_to_open", "quantity": 1},
                {"option_symbol": long_put["symbol"],   "side": "buy_to_open",  "quantity": 1},
            ]
            _place_multi("iron condor", legs, "net_credit", max(0.01, credit))
        else:
            print("[OPTIONS] btc_bear_qqq_firm: insufficient liquidity for iron condor -- skipping.")

    elif regime == "bear":
        exp, chain = chain_for(30, 45)
        if not exp:
            print("[OPTIONS] bear: no expiration in 30-45 DTE window.")
            return
        # 1) Sell covered put: 15-20% OTM below spot (against short)
        put = pick_put(chain, spot, 0.15, 0.20)
        if put:
            _place("covered put", put["symbol"], "sell_to_open",
                   n_contracts, limit_price(put, selling=True))
        # 2) Call spread for squeeze protection: buy 20% OTM call + sell 35% OTM call
        exp2, chain2 = chain_for(25, 35)
        long_call  = pick_call(chain2, spot, 0.18, 0.22) if chain2 else None
        short_call = pick_call(chain2, spot, 0.32, 0.38) if chain2 else None
        if long_call and short_call:
            debit = (limit_price(long_call,  selling=False) -
                     limit_price(short_call, selling=True))
            legs = [
                {"option_symbol": long_call["symbol"],  "side": "buy_to_open",  "quantity": n_contracts},
                {"option_symbol": short_call["symbol"], "side": "sell_to_open", "quantity": n_contracts},
            ]
            _place_multi("call spread (squeeze stop)", legs, "net_debit", max(0.01, debit))

    elif regime in ("bear_oversold", "bear_panic"):
        # Buy put spread: long ATM/5%OTM put + short 25% OTM put
        exp, chain = chain_for(30, 45)
        if not exp:
            print(f"[OPTIONS] {regime}: no expiration in 30-45 DTE window.")
            return
        long_pct = 0.0 if regime == "bear_oversold" else 0.05
        long_put  = pick_put_atm_or_otm(chain, spot, long_pct)
        short_put = pick_put(chain, spot, 0.22, 0.28)
        if long_put and short_put:
            debit = (limit_price(long_put,  selling=False) -
                     limit_price(short_put, selling=True))
            legs = [
                {"option_symbol": long_put["symbol"],  "side": "buy_to_open",  "quantity": 1},
                {"option_symbol": short_put["symbol"], "side": "sell_to_open", "quantity": 1},
            ]
            _place_multi("put spread", legs, "net_debit", max(0.01, debit))
        else:
            print(f"[OPTIONS] {regime}: insufficient liquidity for put spread -- skipping.")

    else:
        print(f"[OPTIONS] No options plan for regime '{regime}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_credentials() -> tuple[str, str, bool]:
    token      = os.getenv("TRADIER_API_TOKEN",   "")
    account_id = os.getenv("TRADIER_ACCOUNT_ID",  "")
    sandbox    = os.getenv("TRADIER_USE_SANDBOX", "true").lower() != "false"
    if not token or not account_id:
        sys.exit("Set TRADIER_API_TOKEN and TRADIER_ACCOUNT_ID env vars.")
    return token, account_id, sandbox


def should_run_now(force_run: bool) -> bool:
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    print(f"New York time: {now_ny:%Y-%m-%d %H:%M:%S}")
    if force_run:
        print("--force-run: bypassing market-time gate.")
        return True
    if now_ny.weekday() >= 5:
        print("Weekend -- skipping.")
        return False
    if REBALANCE_TIME_NY <= now_ny.time() <= REBALANCE_END_NY:
        return True
    print("Outside 3:30-3:55 PM ET rebalance window -- skipping.")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",          action="store_true",
                        help="Submit orders (overrides TRADIER_EXECUTE_ORDERS).")
    parser.add_argument("--force-run",        action="store_true",
                        help="Bypass 3:30 PM ET time gate.")
    parser.add_argument("--cancel-open-orders", action="store_true",
                        help="Cancel pending HIVE orders before running.")
    parser.add_argument("--close-options",    action="store_true",
                        help="Force-close all HIVE options before running.")
    args = parser.parse_args()

    execute = args.execute or os.getenv("TRADIER_EXECUTE_ORDERS", "").lower() == "true"

    if not should_run_now(args.force_run):
        return

    token, account_id, sandbox = load_credentials()
    client = TradierClient(token, account_id, sandbox)

    print(f"\n=== HIVE Tradier Runner  |  {'SANDBOX' if sandbox else 'LIVE'}  |  execute={execute} ===")

    # ── Signal ──────────────────────────────────────────────────────────────
    print("\nBuilding signal ...")
    row = build_signal_row()
    sig = classify_signal(row)

    print(f"  Regime:    {sig['regime']} -- {sig['regime_description']}")
    print(f"  Action:    {sig['equity_action']} {sig['equity_target_alloc']:+.0%}")
    print(f"  Reason:    {sig['reason']}")
    print(f"  Signals:   BTC {'UP' if sig['btc_above_ma20'] else 'DOWN'}  "
          f"QQQ {'UP' if sig['qqq_above_ma50'] else 'DOWN'}  VIX {sig['vix']:.1f}  "
          f"RSI {sig['hive_rsi14']:.0f}")

    # ── Cancel pending orders ────────────────────────────────────────────────
    if args.cancel_open_orders and execute:
        print("\nCanceling open HIVE orders ...")
        for order in client.get_orders():
            if order.get("symbol") == HIVE_SYMBOL and order.get("status") == "pending":
                client.cancel_order(order["id"])
                print(f"  Canceled order {order['id']}")

    # ── Current positions ────────────────────────────────────────────────────
    positions = client.get_positions()
    _, hive_options = parse_positions(positions)

    # ── Close incompatible or force-close options ────────────────────────────
    if args.close_options or not options_compatible(hive_options, sig["regime"]):
        reason = "--close-options" if args.close_options else "regime shift"
        print(f"\nClosing existing HIVE options ({reason}) ...")
        close_all_hive_options(client, hive_options, execute)
        time.sleep(3)
        # Refresh positions after close
        positions   = client.get_positions()
        _, hive_options = parse_positions(positions)

    # ── Equity management ────────────────────────────────────────────────────
    print("\n-- Equity --")
    target_equity_qty = manage_equity(client, sig, execute)

    # ── Options management ───────────────────────────────────────────────────
    print("\n-- Options --")
    spot = float(client.get_quote([HIVE_SYMBOL])[HIVE_SYMBOL]["last"] or 1)

    # Don't open options if we already have them in the right orientation
    if hive_options:
        print(f"[OPTIONS] {len(hive_options)} existing HIVE options position(s) -- skipping new orders.")
        for pos in hive_options:
            print(f"  {pos['symbol']}  qty={pos['quantity']}  cost=${pos.get('cost_basis', 'n/a')}")
    elif sig["regime"] == "danger":
        print("[OPTIONS] DANGER -- no new options.")
    else:
        execute_options_plan(client, sig["regime"], target_equity_qty, spot, execute)

    # ── Options advisory reminder ─────────────────────────────────────────────
    print("\n-- Options advisory (from strategy engine) --")
    for play in sig["options_plays"]:
        print(f"  ({play['priority']}) {play['strategy']}")
        print(f"     {play['detail']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
