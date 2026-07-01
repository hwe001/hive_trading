#!/usr/bin/env python3
"""HIVE Signal Dashboard — Equity Only (Alpaca)"""

import json
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import anthropic
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

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
    VIX_CALM,
    VIX_CAUTION,
    VIX_ELEVATED,
    VIX_DANGER,
    VIX_SYMBOL,
    classify_signal,
    rsi,
)

warnings.filterwarnings("ignore")

REGIME_COLORS = {
    "bull_strong":       "rgba(16,185,129,0.18)",
    "bull":              "rgba(34,197,94,0.13)",
    "bull_cautious":     "rgba(234,179,8,0.15)",
    "btc_bull_qqq_soft": "rgba(251,191,36,0.10)",
    "btc_bear_qqq_firm": "rgba(107,114,128,0.10)",
    "bear":              "rgba(239,68,68,0.18)",
    "bear_oversold":     "rgba(249,115,22,0.15)",
    "bear_panic":        "rgba(220,38,38,0.22)",
    "danger":            "rgba(127,29,29,0.28)",
}


def _action_bg(a: float) -> str:
    return "#065f46" if a > 0 else ("#7f1d1d" if a < 0 else "#1f2937")


def _action_fg(a: float) -> str:
    return "#6ee7b7" if a > 0 else ("#fca5a5" if a < 0 else "#d1d5db")


def _alloc_color(a: float) -> str:
    return "#10b981" if a > 0 else ("#ef4444" if a < 0 else "#6b7280")


def _vix_label(vix: float) -> str:
    if vix >= VIX_DANGER:   return f"DANGER {vix:.0f}"
    if vix >= VIX_ELEVATED: return f"ELEVATED {vix:.0f}"
    if vix >= VIX_CAUTION:  return f"CAUTION {vix:.0f}"
    return f"CALM {vix:.0f}"


def _vix_color(vix: float) -> str:
    if vix >= VIX_DANGER:   return "#ef4444"
    if vix >= VIX_ELEVATED: return "#f97316"
    if vix >= VIX_CAUTION:  return "#eab308"
    return "#10b981"


def _badge(label: str, value: str, color: str) -> str:
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}44;'
        f'border-radius:6px;padding:4px 10px;font-size:.82rem;font-weight:600;'
        f'margin-right:6px;margin-bottom:4px;display:inline-block">'
        f'{label} {value}</span>'
    )


@st.cache_data(ttl=300)
def build_data(period: str = "1y") -> pd.DataFrame:
    tickers = [HIVE_SYMBOL, BTC_SYMBOL, QQQ_SYMBOL, VIX_SYMBOL]
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    df = close[tickers].copy()
    df.columns = ["HIVE", "BTC", "QQQ", "VIX"]
    df.index = pd.to_datetime(df.index)
    df = df.ffill().dropna()

    df["HIVE_MA5"]   = df["HIVE"].rolling(HIVE_MA_FAST).mean()
    df["HIVE_MA20"]  = df["HIVE"].rolling(HIVE_MA_SLOW).mean()
    df["HIVE_RSI14"] = rsi(df["HIVE"], HIVE_RSI_PERIOD)
    df["BTC_MA20"]   = df["BTC"].rolling(BTC_MA_PERIOD).mean()
    df["QQQ_MA50"]   = df["QQQ"].rolling(QQQ_MA_PERIOD).mean()
    df = df.dropna()

    signals = [classify_signal(row) for _, row in df.iterrows()]
    df["alloc"]  = [s["equity_target_alloc"] for s in signals]
    df["regime"] = [s["regime"] for s in signals]
    return df


def make_signal_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=("HIVE Price & Signals", "Allocation %", "RSI-14"),
    )

    fig.add_trace(go.Scatter(
        x=df.index, y=df["HIVE"], name="HIVE",
        line=dict(color="#60a5fa", width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["HIVE_MA5"], name=f"MA{HIVE_MA_FAST}",
        line=dict(color="#fb923c", width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["HIVE_MA20"], name=f"MA{HIVE_MA_SLOW}",
        line=dict(color="#fbbf24", width=1.2),
    ), row=1, col=1)

    # Regime-colored background bands (price panel only)
    prev, band_start = None, df.index[0]
    for date, regime in zip(df.index, df["regime"]):
        if regime != prev:
            if prev is not None:
                fig.add_vrect(
                    x0=band_start, x1=date, row=1, col=1,
                    fillcolor=REGIME_COLORS.get(prev, "rgba(150,150,150,0.08)"),
                    layer="below", line_width=0,
                )
            band_start, prev = date, regime
    if prev:
        fig.add_vrect(
            x0=band_start, x1=df.index[-1], row=1, col=1,
            fillcolor=REGIME_COLORS.get(prev, "rgba(150,150,150,0.08)"),
            layer="below", line_width=0,
        )

    # Buy/sell/flat markers at regime transitions
    ch = df[df["regime"] != df["regime"].shift(1)].copy()
    for mask, symbol, color, name, offset in [
        (ch["alloc"] > 0,  "triangle-up",   "#10b981", "Enter Long",  0.95),
        (ch["alloc"] < 0,  "triangle-down",  "#ef4444", "Enter Short", 1.05),
        (ch["alloc"] == 0, "x",              "#6b7280", "Go Flat",     1.00),
    ]:
        sub = ch[mask]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub.index, y=sub["HIVE"] * offset,
                mode="markers", name=name,
                marker=dict(symbol=symbol, size=10, color=color),
            ), row=1, col=1)

    # Allocation bars
    fig.add_trace(go.Bar(
        x=df.index, y=df["alloc"] * 100, name="Alloc %",
        marker_color=[_alloc_color(a) for a in df["alloc"]], showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_color="#4b5563", row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df["HIVE_RSI14"], name="RSI-14",
        line=dict(color="#a78bfa", width=1.2),
    ), row=3, col=1)
    fig.add_hline(y=RSI_OVERBOUGHT, line_dash="dash",
                  line_color="#ef4444", line_width=1, row=3, col=1)
    fig.add_hline(y=RSI_OVERSOLD, line_dash="dash",
                  line_color="#10b981", line_width=1, row=3, col=1)
    fig.add_hrect(
        y0=RSI_OVERSOLD, y1=RSI_OVERBOUGHT,
        fillcolor="rgba(99,102,241,0.04)", layer="below",
        line_width=0, row=3, col=1,
    )

    fig.update_layout(
        height=620,
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#d1d5db", size=11),
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=40, b=20),
        hovermode="x unified",
    )
    for row in [1, 2, 3]:
        fig.update_xaxes(gridcolor="#1f2937", showgrid=True, row=row, col=1)
        fig.update_yaxes(gridcolor="#1f2937", showgrid=True, row=row, col=1)
    return fig


def make_btc_qqq_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"BTC vs MA{BTC_MA_PERIOD}", f"QQQ vs MA{QQQ_MA_PERIOD}"),
        horizontal_spacing=0.08,
    )
    specs = [
        ("BTC", "BTC_MA20", "#f59e0b", "#fcd34d", f"BTC MA{BTC_MA_PERIOD}", 1),
        ("QQQ", "QQQ_MA50", "#38bdf8", "#7dd3fc", f"QQQ MA{QQQ_MA_PERIOD}", 2),
    ]
    for sym, ma_col, color, ma_color, ma_label, col in specs:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[sym], name=sym,
            line=dict(color=color, width=1.5),
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=df.index, y=df[ma_col], name=ma_label,
            line=dict(color=ma_color, width=1, dash="dash"),
        ), row=1, col=col)
    fig.update_layout(
        height=280,
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#d1d5db", size=11),
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=40, b=20),
        hovermode="x unified",
    )
    for col in [1, 2]:
        fig.update_xaxes(gridcolor="#1f2937", showgrid=True, row=1, col=col)
        fig.update_yaxes(gridcolor="#1f2937", showgrid=True, row=1, col=col)
    return fig


def signal_changes(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    mask = df["regime"] != df["regime"].shift(1)
    ch = df[mask].iloc[::-1].head(n).copy()
    ch["Action"]  = ch["alloc"].apply(
        lambda a: "LONG" if a > 0 else ("SHORT" if a < 0 else "FLAT"))
    ch["Alloc"]   = ch["alloc"].apply(lambda a: f"{a:+.0%}")
    ch["BTC ↑↓"]  = ch.apply(lambda r: "↑" if r["BTC"] >= r["BTC_MA20"] else "↓", axis=1)
    ch["QQQ ↑↓"]  = ch.apply(lambda r: "↑" if r["QQQ"] >= r["QQQ_MA50"] else "↓", axis=1)
    ch["VIX"]     = ch["VIX"].round(1)
    ch["RSI"]     = ch["HIVE_RSI14"].round(1)
    ch["HIVE $"]  = ch["HIVE"].round(3)
    ch.index = ch.index.strftime("%Y-%m-%d")
    ch.index.name = "Date"
    return ch[["Action", "regime", "Alloc", "HIVE $", "BTC ↑↓", "QQQ ↑↓", "VIX", "RSI"]].rename(
        columns={"regime": "Regime"})


def signal_flip_analysis(sig: dict) -> list:
    btc, btc_ma = sig["btc"], sig["btc_ma20"]
    qqq, qqq_ma = sig["qqq"], sig["qqq_ma50"]
    vix, rsi14  = sig["vix"], sig["hive_rsi14"]
    pct_btc = (btc - btc_ma) / btc_ma * 100
    pct_qqq = (qqq - qqq_ma) / qqq_ma * 100
    lines = []

    if sig["btc_above_ma20"]:
        lines.append(
            f"**BTC** is {pct_btc:+.1f}% above its {BTC_MA_PERIOD}-day MA (${btc_ma:,.0f}). "
            f"A close below ${btc_ma:,.0f} shifts to bearish BTC and would reduce or close longs."
        )
    else:
        lines.append(
            f"**BTC** is {pct_btc:+.1f}% below its {BTC_MA_PERIOD}-day MA (${btc_ma:,.0f}). "
            f"A close above ${btc_ma:,.0f} turns BTC bullish and unlocks pilot-to-full long."
        )

    if sig["qqq_above_ma50"]:
        lines.append(
            f"**QQQ** is {pct_qqq:+.1f}% above its {QQQ_MA_PERIOD}-day MA (${qqq_ma:.2f}). "
            f"A break below ${qqq_ma:.2f} removes risk-on confirmation and reduces sizing."
        )
    else:
        lines.append(
            f"**QQQ** is {pct_qqq:+.1f}% below its {QQQ_MA_PERIOD}-day MA (${qqq_ma:.2f}). "
            f"A recovery above ${qqq_ma:.2f} adds risk-on tailwind and could unlock a larger long."
        )

    if vix < VIX_CALM:
        lines.append(
            f"**VIX** {vix:.1f} is calm (<{VIX_CALM:.0f}): full-sized positions allowed. "
            f"Rising above {VIX_CAUTION:.0f} moderates sizing; above {VIX_ELEVATED:.0f} flattens bulls."
        )
    elif vix < VIX_CAUTION:
        lines.append(
            f"**VIX** {vix:.1f} in {VIX_CALM:.0f}–{VIX_CAUTION:.0f} range: moderate sizing. "
            f"Above {VIX_CAUTION:.0f} caps shorts; below {VIX_CALM:.0f} unlocks full bull allocation."
        )
    elif vix < VIX_ELEVATED:
        lines.append(
            f"**VIX** {vix:.1f} elevated ({VIX_CAUTION:.0f}–{VIX_ELEVATED:.0f}): shorts capped at 40%. "
            f"Drop below {VIX_CAUTION:.0f} for full short; above {VIX_ELEVATED:.0f} forces flat."
        )
    elif vix < VIX_DANGER:
        lines.append(
            f"**VIX** {vix:.1f} in danger zone ({VIX_ELEVATED:.0f}–{VIX_DANGER:.0f}): bulls capped at light long (35%). "
            f"Above {VIX_DANGER:.0f} forces full flat."
        )
    else:
        lines.append(
            f"**VIX** {vix:.1f} ≥ {VIX_DANGER:.0f}: crisis — all positions flat "
            f"until VIX drops below {VIX_DANGER:.0f}."
        )

    if rsi14 >= RSI_OVERBOUGHT:
        lines.append(
            f"**HIVE RSI** {rsi14:.0f} is overbought (≥{RSI_OVERBOUGHT:.0f}) — trimmed to light long. "
            f"Drop below {RSI_OVERBOUGHT:.0f} in a bull to unlock moderate-to-full sizing."
        )
    elif rsi14 <= RSI_OVERSOLD:
        lines.append(
            f"**HIVE RSI** {rsi14:.0f} is oversold (≤{RSI_OVERSOLD:.0f}) — short capped at light (25%). "
            f"Above {RSI_OVERSOLD:.0f} allows heavier short in a bear regime."
        )
    else:
        lines.append(
            f"**HIVE RSI** {rsi14:.0f} is neutral. "
            f"A spike above {RSI_OVERBOUGHT:.0f} in a bull would trim to light long (35%)."
        )
    return lines


def generate_ai_brief(sig: dict) -> str:
    try:
        client = anthropic.Anthropic()
        alloc = sig["equity_target_alloc"]
        action_str = "LONG" if alloc > 0 else ("SHORT" if alloc < 0 else "FLAT")
        prompt = (
            f"You are a quant trading analyst. Write a concise brief for HIVE (a Bitcoin miner stock).\n\n"
            f"Regime: {sig['regime']} — {sig['regime_description']}\n"
            f"Signal: {action_str} {abs(alloc):.0%} | Confidence: {sig['confidence']}\n"
            f"Reason: {sig['reason']}\n\n"
            f"BTC: ${sig['btc']:,.0f} vs MA{BTC_MA_PERIOD} ${sig['btc_ma20']:,.0f} "
            f"({'above' if sig['btc_above_ma20'] else 'below'})\n"
            f"QQQ: ${sig['qqq']:.2f} vs MA{QQQ_MA_PERIOD} ${sig['qqq_ma50']:.2f} "
            f"({'above' if sig['qqq_above_ma50'] else 'below'})\n"
            f"VIX: {sig['vix']:.1f} | HIVE RSI-14: {sig['hive_rsi14']:.1f} | "
            f"HIVE: ${sig['hive']:.3f}\n\n"
            "Write exactly 3 sections:\n"
            "**Market Read** (1–2 sentences on macro/BTC/QQQ context)\n"
            "**Today's Signal** (1–2 sentences on the regime and position sizing)\n"
            "**Risk Triggers** (2–3 bullet points on what would flip the signal)\n\n"
            "Be direct, specific, and reference the actual numbers."
        )
        with client.messages.stream(
            model="claude-opus-4-8",
            max_tokens=400,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            msg = stream.get_final_message()
        text_blocks = [b.text for b in msg.content if hasattr(b, "text") and b.type == "text"]
        return "\n\n".join(text_blocks) if text_blocks else "No response generated."
    except Exception as e:
        return f"AI brief unavailable: {e}"


def render() -> None:
    st.set_page_config(
        page_title="HIVE Signal Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""<style>
    body,.stApp{background:#0f172a;color:#e2e8f0}
    .block-container{padding-top:1.5rem}
    h1,h2,h3{color:#f1f5f9}
    [data-testid="stMetricValue"]{font-size:1.1rem}
    </style>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        period = st.selectbox("Lookback", ["3mo", "6mo", "1y", "2y"], index=2)
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("Data via Yahoo Finance · 5-min cache")
        st.markdown("---")
        st.markdown(
            f"**Strategy:** HIVE equity-only · Alpaca  \n"
            f"BTC {BTC_MA_PERIOD}-day MA + QQQ {QQQ_MA_PERIOD}-day MA + VIX + RSI-{HIVE_RSI_PERIOD}"
        )
        st.caption("No options overlay — see tradier-options branch.")

    st.title("📊 HIVE Signal Dashboard")
    st.caption("Equity-only strategy · Alpaca · No options overlay")

    with st.spinner("Loading market data…"):
        try:
            df = build_data(period)
        except Exception as e:
            st.error(f"Data error: {e}")
            st.stop()

    if df.empty:
        st.warning("No data returned. Try a different lookback period.")
        st.stop()

    sig = classify_signal(df.iloc[-1])
    alloc = sig["equity_target_alloc"]
    action_str = "BUY / LONG" if alloc > 0 else ("SELL / SHORT" if alloc < 0 else "FLAT / CASH")
    bg, fg = _action_bg(alloc), _action_fg(alloc)

    # Hero card
    st.markdown(f"""
    <div style="background:{bg};border-radius:12px;padding:20px 28px;margin-bottom:16px">
      <div style="font-size:.82rem;color:{fg}88;font-weight:600;letter-spacing:.08em">
        CURRENT SIGNAL · {sig['regime'].upper().replace('_', ' ')}
      </div>
      <div style="font-size:2.4rem;color:{fg};font-weight:800;line-height:1.1">{action_str}</div>
      <div style="font-size:1.1rem;color:{fg}cc;margin-top:4px">
        {alloc:+.0%} equity allocation · {sig['confidence'].upper()} confidence
      </div>
      <div style="font-size:.88rem;color:{fg}99;margin-top:8px">{sig['reason']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Condition badges
    btc_up  = sig["btc_above_ma20"]
    qqq_up  = sig["qqq_above_ma50"]
    rsi14   = sig["hive_rsi14"]
    rsi_c   = "#ef4444" if rsi14 >= RSI_OVERBOUGHT else ("#10b981" if rsi14 <= RSI_OVERSOLD else "#6b7280")
    rsi_lbl = "OVERBOUGHT" if rsi14 >= RSI_OVERBOUGHT else ("OVERSOLD" if rsi14 <= RSI_OVERSOLD else "NEUTRAL")

    st.markdown(
        _badge(f"BTC ${sig['btc']:,.0f}", "↑" if btc_up else "↓", "#10b981" if btc_up else "#ef4444") +
        _badge(f"QQQ ${sig['qqq']:.2f}", "↑" if qqq_up else "↓", "#10b981" if qqq_up else "#ef4444") +
        _badge("VIX", _vix_label(sig["vix"]), _vix_color(sig["vix"])) +
        _badge(f"RSI {rsi14:.0f}", rsi_lbl, rsi_c) +
        _badge("HIVE", f"${sig['hive']:.3f}", "#60a5fa"),
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    last = df.index[-1]
    last_str = last.strftime("%b %d, %Y") if hasattr(last, "strftime") else str(last)
    c1.metric("HIVE", f"${sig['hive']:.3f}")
    c2.metric(f"MA{HIVE_MA_FAST}", f"${sig['hive_ma5']:.3f}")
    c3.metric(f"MA{HIVE_MA_SLOW}", f"${sig['hive_ma20']:.3f}")
    c4.metric("RSI-14", f"{rsi14:.1f}")
    c5.metric("As of", last_str)

    st.markdown("---")

    st.subheader("Price, Allocation & RSI")
    st.plotly_chart(make_signal_chart(df), use_container_width=True)

    st.subheader("Signal Drivers: BTC & QQQ")
    st.plotly_chart(make_btc_qqq_chart(df), use_container_width=True)

    st.markdown("---")

    with st.expander("🔀 What would change the signal?", expanded=True):
        for line in signal_flip_analysis(sig):
            st.markdown(f"• {line}")

    st.subheader("📋 Recent Signal Changes")
    st.dataframe(signal_changes(df), use_container_width=True)

    st.markdown("---")

    col_ai, col_dl = st.columns([3, 1])
    with col_ai:
        if st.button("🤖 Generate AI Brief (Claude Opus 4.8)", use_container_width=True):
            with st.spinner("Generating brief…"):
                brief = generate_ai_brief(sig)
            st.markdown(brief)
    with col_dl:
        st.download_button(
            "⬇️ Download Signal JSON",
            data=json.dumps(dict(sig), indent=2, default=str),
            file_name=f"hive_signal_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")
    st.caption(
        f"Updated: {datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M ET')} · "
        "HIVE Signal Dashboard — Equity Only (Alpaca) · Not financial advice."
    )


if __name__ == "__main__":
    render()
