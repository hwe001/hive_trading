#!/usr/bin/env python3
"""
HIVE Trading Signal Dashboard.

Shows the current long/short regime for HIVE, the equity position target,
and an advisory options overlay. BTC and QQQ are displayed as signal inputs only.
No broker connection. No order execution.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
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
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    VIX_SYMBOL,
    classify_signal,
    rsi,
)

CLAUDE_MODEL = "claude-haiku-4-5"
AI_SECTOR_SYMBOLS = ["NVDA", "MSFT", "GOOGL", "META", "AVGO", "AMD", "TSM", "MSTR"]

st.set_page_config(page_title="HIVE Signal", page_icon="chart", layout="wide")
st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #1d4ed8; border: 1px solid #1d4ed8;
        color: white; font-weight: 700;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1e40af; border-color: #1e40af; color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)


def password_gate() -> bool:
    password = get_secret("SIGNAL_APP_PASSWORD", "")
    if not password:
        return True
    st.sidebar.subheader("Access")
    entered = st.sidebar.text_input("Password", type="password")
    if entered == password:
        return True
    if entered:
        st.sidebar.error("Incorrect password")
    st.info("Enter the dashboard password in the sidebar.")
    return False


def pct(x: float) -> str:
    return f"{x:+.0%}" if x != 0 else "0%"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_history(symbol: str, period: str = "6mo") -> pd.Series:
    last_error = None
    for attempt in range(3):
        try:
            hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
            if not hist.empty:
                return hist["Close"].rename(symbol.replace("^", "").replace("-", "_"))
        except Exception as exc:
            last_error = exc
        time.sleep(1 + attempt)
    raise RuntimeError(f"No data for {symbol}. Last error: {last_error}")


@st.cache_data(ttl=300)
def build_data() -> pd.DataFrame:
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

    return df.dropna()


@st.cache_data(ttl=3600)
def fetch_ai_news() -> list[dict]:
    items, seen = [], set()
    for symbol in AI_SECTOR_SYMBOLS:
        try:
            for item in yf.Ticker(symbol).news[:4]:
                content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
                title = (item.get("title") or content.get("title") or "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                published = item.get("providerPublishTime") or content.get("pubDate")
                pub_text = ""
                if isinstance(published, (int, float)):
                    pub_text = datetime.fromtimestamp(published, ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
                elif isinstance(published, str):
                    pub_text = published[:10]
                items.append({
                    "symbol": symbol,
                    "title": title,
                    "publisher": item.get("publisher") or content.get("provider", {}).get("displayName", ""),
                    "published": pub_text,
                })
        except Exception:
            continue
    return items[:14]


def format_news(news_items: list[dict]) -> str:
    if not news_items:
        return "No recent headlines available."
    lines = []
    for item in news_items:
        date = f"{item['published']} | " if item.get("published") else ""
        pub  = f" | {item['publisher']}" if item.get("publisher") else ""
        lines.append(f"- {item['symbol']}: {date}{item['title']}{pub}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AI brief
# ---------------------------------------------------------------------------

def build_brief_prompt(df: pd.DataFrame, sig: dict, news_items: list[dict]) -> str:
    recent   = df.tail(6)
    hive_w   = recent["HIVE"].iloc[-1] / recent["HIVE"].iloc[0] - 1.0
    btc_w    = recent["BTC"].iloc[-1]  / recent["BTC"].iloc[0]  - 1.0
    qqq_w    = recent["QQQ"].iloc[-1]  / recent["QQQ"].iloc[0]  - 1.0
    vix_chg  = recent["VIX"].iloc[-1]  - recent["VIX"].iloc[0]

    options_summary = "\n".join(
        f"  ({p['priority']}) {p['strategy']}: {p['detail']}"
        for p in sig["options_plays"]
    )

    return f"""
You are writing a concise weekly market brief for a private HIVE trading dashboard.
The strategy trades only HIVE stock (long, short, or flat). QQQ and Bitcoin are used as
signal inputs only. An options overlay is used on the HIVE position to harvest volatility premium.

Do not give personalised financial advice. Explain the signal and risk posture in plain English.

Current signal:
- Regime: {sig['regime']} -- {sig['regime_description']}
- HIVE equity action: {sig['equity_action']} at {pct(sig['equity_target_alloc'])} of account
- Confidence: {sig['confidence']}
- Reason: {sig['reason']}

Options advisory:
{options_summary}

Key indicators:
- HIVE: ${sig['hive']:.4f}  MA20 ${sig['hive_ma20']:.4f}  RSI-14 {sig['hive_rsi14']:.0f}  ({'ABOVE' if sig['hive_above_ma20'] else 'BELOW'} MA20)
- BTC:  ${sig['btc']:,.0f}  MA20 ${sig['btc_ma20']:,.0f}  ({'ABOVE' if sig['btc_above_ma20'] else 'BELOW'} MA20)
- QQQ:  ${sig['qqq']:.2f}  MA50 ${sig['qqq_ma50']:.2f}  ({'ABOVE' if sig['qqq_above_ma50'] else 'BELOW'} MA50)
- VIX:  {sig['vix']:.2f}

Past ~5 trading days:
- HIVE: {pct(hive_w)}  BTC: {pct(btc_w)}  QQQ: {pct(qqq_w)}  VIX change: {vix_chg:+.2f}

Recent AI / Bitcoin / semiconductor headlines:
{format_news(news_items)}

Write exactly four short sections:
1. Weekly read (BTC trend, QQQ health, what it means for HIVE)
2. Current signal (equity + options -- what to do and why)
3. Risk watch (what could break or flip the trade)
4. What would change the signal
""".strip()


def generate_brief(df: pd.DataFrame, sig: dict, news_items: list[dict]) -> str:
    api_key = get_secret("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "Add ANTHROPIC_API_KEY in Streamlit secrets to enable the weekly AI brief."

    model   = get_secret("CLAUDE_MODEL", CLAUDE_MODEL)
    payload = {
        "model": model,
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": build_brief_prompt(df, sig, news_items)}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Claude API error {exc.code}: {exc.read().decode(errors='replace')}") from exc

    content = data.get("content", [])
    if not content:
        raise RuntimeError("Claude API returned no text.")
    return content[0].get("text", "").strip()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def regime_color(regime: str) -> str:
    if "danger" in regime:
        return "[RED]"
    if "bear" in regime:
        return "[ORANGE]"
    if "flat" in regime or "btc_bear_qqq_firm" in regime:
        return "[NEUTRAL]"
    if "cautious" in regime or "soft" in regime:
        return "[YELLOW]"
    return "[GREEN]"


def render_dashboard() -> None:
    st.title("HIVE Signal Dashboard")
    st.caption("Manual guide only. QQQ and BTC are signal inputs only -- HIVE is the only traded asset.")

    if st.button("Refresh market data"):
        st.cache_data.clear()
        st.rerun()

    df  = build_data()
    sig = classify_signal(df.iloc[-1])

    # Regime banner
    icon = regime_color(sig["regime"])
    st.subheader(f"{icon} {sig['regime_description']}")
    st.write(sig["reason"])
    st.caption(f"New York time: {sig['timestamp_ny']}")

    # Equity allocation row
    alloc = sig["equity_target_alloc"]
    alloc_str = (
        f"LONG {alloc:.0%}" if alloc > 0
        else (f"SHORT {abs(alloc):.0%}" if alloc < 0 else "FLAT")
    )
    cols = st.columns(4)
    cols[0].metric("Action",     sig["equity_action"])
    cols[1].metric("Allocation", alloc_str)
    cols[2].metric("Confidence", sig["confidence"])
    cols[3].metric("VIX",        f"{sig['vix']:.2f}")

    # HIVE indicators
    rsi_val  = sig["hive_rsi14"]
    rsi_flag = " [!] overbought" if sig["hive_rsi_overbought"] else (" [!] oversold" if sig["hive_rsi_oversold"] else "")
    cols = st.columns(4)
    cols[0].metric("HIVE",        f"${sig['hive']:.4f}")
    cols[1].metric("HIVE MA20",   f"${sig['hive_ma20']:.4f}", "above" if sig["hive_above_ma20"] else "below")
    cols[2].metric("HIVE RSI-14", f"{rsi_val:.0f}{rsi_flag}")
    cols[3].metric("HIVE MA5",    f"${sig['hive_ma5']:.4f}")

    # Signal inputs (read-only)
    st.divider()
    st.subheader("Signal Inputs (not traded)")
    cols = st.columns(4)
    cols[0].metric("BTC",      f"${sig['btc']:,.0f}",      "above MA20" if sig["btc_above_ma20"] else "below MA20")
    cols[1].metric("BTC MA20", f"${sig['btc_ma20']:,.0f}")
    cols[2].metric("QQQ",      f"${sig['qqq']:.2f}",       "above MA50" if sig["qqq_above_ma50"] else "below MA50")
    cols[3].metric("QQQ MA50", f"${sig['qqq_ma50']:.2f}")

    # Options advisory
    st.divider()
    st.subheader("Options Advisory")
    st.caption("Advisory only -- not executed by the paper runner.")
    for play in sig["options_plays"]:
        priority_label = "Primary" if play["priority"] == 1 else "Secondary"
        with st.expander(f"({priority_label}) {play['strategy']}", expanded=(play["priority"] == 1)):
            st.write(play["detail"])

    # AI brief
    st.divider()
    st.subheader("Weekly AI Brief")
    st.caption("Optional. Calls Claude when you press the button.")
    if st.button("Generate weekly AI brief", type="primary"):
        with st.spinner("Generating brief..."):
            news_items = fetch_ai_news()
            st.markdown(generate_brief(df, sig, news_items))
            if news_items:
                with st.expander("Headlines used"):
                    for item in news_items:
                        date = f"{item['published']} | " if item.get("published") else ""
                        st.write(f"{item['symbol']}: {date}{item['title']}")

    # Charts
    st.divider()
    st.subheader("Charts")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("HIVE with MA5 / MA20")
        st.line_chart(df[["HIVE", "HIVE_MA5", "HIVE_MA20"]].tail(80))
        st.caption("QQQ with MA50")
        st.line_chart(df[["QQQ", "QQQ_MA50"]].tail(80))
    with col2:
        st.caption("BTC with MA20")
        st.line_chart(df[["BTC", "BTC_MA20"]].tail(80))
        st.caption("VIX")
        st.line_chart(df[["VIX"]].tail(80))

    st.caption("HIVE RSI-14")
    st.line_chart(df[["HIVE_RSI14"]].tail(80))

    # Data table
    st.subheader("Recent Data")
    display_cols = ["HIVE", "HIVE_MA5", "HIVE_MA20", "HIVE_RSI14", "BTC", "BTC_MA20", "QQQ", "QQQ_MA50", "VIX"]
    st.dataframe(df[display_cols].tail(20).round(4), use_container_width=True)

    # Download
    sig_export = {k: v for k, v in sig.items() if k != "options_plays"}
    sig_export["options_plays"] = [{k: v for k, v in p.items()} for p in sig["options_plays"]]
    st.download_button(
        "Download JSON signal",
        data=json.dumps(sig_export, indent=2, default=str),
        file_name="hive_signal.json",
        mime="application/json",
    )


if password_gate():
    try:
        render_dashboard()
    except Exception as exc:
        st.error(f"Could not generate signal: {exc}")
