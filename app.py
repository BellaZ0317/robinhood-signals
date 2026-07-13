"""
app.py  —  Abby 投资信号仪表盘
Self-contained Streamlit app. No file writes needed.
Deploy: streamlit.io/cloud (free)
Local:  streamlit run app.py
"""

import time
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
from datetime import date, timedelta

# ── Personal data ───────────────────────────────────────────────────────────────
# Transactions and the monthly buy-plan/budget both live in Google Sheets (separate
# tabs of the same spreadsheet) so they're readable/writable from any device,
# gated behind the password check in check_password().
TRANSACTIONS_COLUMNS = ["date", "ticker", "action", "shares", "price", "amount_usd", "notes"]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📈 Abby 投资信号",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Indicator functions ────────────────────────────────────────────────────────

def sma(s, n):    return s.rolling(n).mean()
def ema(s, n):    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def macd(s, fast=12, slow=26, sig=9):
    ml = ema(s, fast) - ema(s, slow)
    sl = ml.ewm(span=sig, adjust=False).mean()
    return ml, sl, ml - sl

def bollinger(s, n=20, k=2):
    mid = sma(s, n); std = s.rolling(n).std()
    return mid + k * std, mid, mid - k * std

def calc_atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def pct(s, n):
    return (s.iloc[-1] / s.iloc[-n] - 1) * 100 if len(s) > n else np.nan


# ── Candlestick pattern detection ──────────────────────────────────────────────

def detect_patterns(df):
    if len(df) < 3:
        return []
    patterns = []
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    i = len(df) - 1
    rng = h[i] - l[i]; body = abs(c[i] - o[i])
    upper = h[i] - max(c[i], o[i]); lower = min(c[i], o[i]) - l[i]
    bull = c[i] > o[i]
    if rng < 0.001:
        return patterns
    br = body / rng
    if br < 0.1:
        patterns.append(("Doji",            "neutral", "十字星"))
    elif lower/rng > 0.6 and br < 0.35 and bull:
        patterns.append(("Hammer",          "bullish", "锤子线"))
    elif lower/rng > 0.6 and br < 0.35 and not bull:
        patterns.append(("Hanging Man",     "bearish", "吊颈线"))
    elif upper/rng > 0.6 and br < 0.35 and not bull:
        patterns.append(("Shooting Star",   "bearish", "射击之星"))
    elif upper/rng > 0.6 and br < 0.35 and bull:
        patterns.append(("Inverted Hammer", "bullish", "倒锤子线"))
    elif br > 0.85:
        patterns.append(("Bullish Marubozu" if bull else "Bearish Marubozu",
                          "bullish" if bull else "bearish",
                          "大阳线" if bull else "大阴线"))
    if i >= 1:
        p_lo = min(c[i-1], o[i-1]); p_hi = max(c[i-1], o[i-1])
        c_lo = min(c[i],   o[i]);   c_hi = max(c[i],   o[i])
        if c[i-1] < o[i-1] and bull and c_lo < p_lo and c_hi > p_hi:
            patterns.append(("Bullish Engulfing", "bullish", "看涨吞没"))
        elif c[i-1] > o[i-1] and not bull and c_lo < p_lo and c_hi > p_hi:
            patterns.append(("Bearish Engulfing", "bearish", "看跌吞没"))
    if i >= 2:
        mid_rng  = h[i-1] - l[i-1]
        mid_body = abs(c[i-1] - o[i-1])
        small    = mid_body / (mid_rng + 0.001) < 0.3
        if c[i-2] < o[i-2] and small and bull and c[i] > (o[i-2]+c[i-2])/2:
            patterns.append(("Morning Star", "bullish", "早晨之星"))
        if c[i-2] > o[i-2] and small and not bull and c[i] < (o[i-2]+c[i-2])/2:
            patterns.append(("Evening Star", "bearish", "黄昏之星"))
    return patterns


# ── Signal scoring ─────────────────────────────────────────────────────────────

def score_to_signal(score):
    if   score >=  1.5: return "STRONG_BUY"
    elif score >=  0.5: return "BUY"
    elif score >= -0.5: return "HOLD"
    elif score >= -1.5: return "SELL"
    else:               return "STRONG_SELL"

def analyze_df(ticker, df, bench_close):
    if df is None or len(df) < 60:
        return {"ticker": ticker, "error": "数据不足"}
    close = df["Close"]; price = close.iloc[-1]
    rsi14      = rsi(close).iloc[-1]
    s20        = sma(close, 20).iloc[-1]
    s50        = sma(close, 50).iloc[-1]
    s200       = sma(close, 200).iloc[-1] if len(close) >= 200 else np.nan
    ml, sl, hist = macd(close)
    bb_up, _, bb_lo = bollinger(close)
    bb_rng     = bb_up.iloc[-1] - bb_lo.iloc[-1]
    bb_pct     = (price - bb_lo.iloc[-1]) / bb_rng * 100 if bb_rng > 0 else np.nan
    atr14      = calc_atr(df).iloc[-1]
    vol        = df["Volume"].iloc[-1]      if "Volume" in df.columns else np.nan
    vol20      = df["Volume"].rolling(20).mean().iloc[-1] if "Volume" in df.columns else np.nan
    vol_ratio  = vol / vol20 if not np.isnan(vol20) and vol20 > 0 else np.nan
    ret_1w = pct(close, 5);  ret_1m = pct(close, 21)
    ret_3m = pct(close, 63); ret_1y = pct(close, 252)
    b_1w = pct(bench_close, 5);  b_1m = pct(bench_close, 21)
    b_3m = pct(bench_close, 63); b_1y = pct(bench_close, 252)
    patterns     = detect_patterns(df)
    pattern_bias = sum(1 if b=="bullish" else -1 if b=="bearish" else 0 for _,b,_ in patterns)

    # 1D
    s1d = 0; w1d = []
    if   rsi14 < 30: s1d += 2.0; w1d.append(f"RSI超卖({rsi14:.0f})")
    elif rsi14 < 40: s1d += 1.0; w1d.append(f"RSI偏低({rsi14:.0f})")
    elif rsi14 > 70: s1d -= 2.0; w1d.append(f"RSI超买({rsi14:.0f})")
    elif rsi14 > 60: s1d -= 1.0; w1d.append(f"RSI偏高({rsi14:.0f})")
    else:                         w1d.append(f"RSI中性({rsi14:.0f})")
    if not np.isnan(bb_pct):
        if   bb_pct < 20: s1d += 1.0; w1d.append(f"靠近布林下轨({bb_pct:.0f}%)")
        elif bb_pct > 80: s1d -= 1.0; w1d.append(f"靠近布林上轨({bb_pct:.0f}%)")
        else:                          w1d.append(f"布林带中区({bb_pct:.0f}%)")
    s1d += pattern_bias * 0.5
    for n, b, zh in patterns:
        w1d.append(f"{zh}({n})")
    if not np.isnan(vol_ratio):
        if   vol_ratio > 1.5 and s1d > 0: s1d += 0.5; w1d.append(f"放量确认(×{vol_ratio:.1f})")
        elif vol_ratio > 1.5 and s1d < 0: s1d -= 0.5; w1d.append(f"放量下跌(×{vol_ratio:.1f})")

    # 1W
    s1w = 0; w1w = []
    if   rsi14 < 35: s1w += 1.5; w1w.append(f"RSI oversold({rsi14:.0f})")
    elif rsi14 < 45: s1w += 0.5; w1w.append(f"RSI low({rsi14:.0f})")
    elif rsi14 > 70: s1w -= 1.5; w1w.append(f"RSI overbought({rsi14:.0f})")
    elif rsi14 > 60: s1w -= 0.5; w1w.append(f"RSI high({rsi14:.0f})")
    else:                         w1w.append(f"RSI neutral({rsi14:.0f})")
    s1w += 0.5 if price > s20 else -0.5; w1w.append("价格>SMA20" if price > s20 else "价格<SMA20")
    s1w += 0.5 if hist.iloc[-1] > 0 else -0.5; w1w.append("MACD+" if hist.iloc[-1] > 0 else "MACD−")

    # 1M
    s1m = 0; w1m = []
    s1m += 1.0 if price > s50 else -1.0; w1m.append("价格>SMA50" if price > s50 else "价格<SMA50")
    s1m += 0.5 if ml.iloc[-1] > sl.iloc[-1] else -0.5
    w1m.append("MACD上穿" if ml.iloc[-1] > sl.iloc[-1] else "MACD下穿")
    a1m = ret_1m - b_1m if not np.isnan(b_1m) else 0
    if a1m > 2: s1m += 0.5; w1m.append(f"跑赢SPY+{a1m:.1f}%")
    elif a1m < -2: s1m -= 0.5; w1m.append(f"跑输SPY{a1m:.1f}%")

    # 3M
    s3m = 0; w3m = []
    s3m += 0.5 if price > s50 else -0.5; w3m.append("价格>SMA50" if price > s50 else "价格<SMA50")
    if not np.isnan(s200):
        s3m += 1.0 if price > s200 else -1.0; w3m.append("价格>SMA200" if price > s200 else "价格<SMA200")
    a3m = ret_3m - b_3m if not np.isnan(b_3m) else 0
    if a3m > 3: s3m += 0.5; w3m.append(f"3M超额+{a3m:.1f}%")
    elif a3m < -3: s3m -= 0.5; w3m.append(f"3M超额{a3m:.1f}%")

    # 1Y
    s1y = 0; w1y = []
    if not np.isnan(s200):
        s1y += 1.5 if price > s200 else -1.5
        w1y.append("牛市结构(>SMA200)" if price > s200 else "熊市结构(<SMA200)")
    a1y = ret_1y - b_1y if (not np.isnan(b_1y) and not np.isnan(ret_1y)) else 0
    if a1y > 5: s1y += 0.5; w1y.append(f"年化超额+{a1y:.1f}%")
    elif a1y < -5: s1y -= 0.5; w1y.append(f"年化超额{a1y:.1f}%")

    return {
        "ticker":      ticker,
        "price":       round(price, 2),
        "rsi14":       round(rsi14, 1),
        "bb_pct":      round(bb_pct, 1)    if not np.isnan(bb_pct)    else None,
        "atr14":       round(atr14, 2),
        "vol_ratio":   round(vol_ratio, 2) if not np.isnan(vol_ratio) else None,
        "ret_1w":      round(ret_1w, 2),
        "ret_1m":      round(ret_1m, 2),
        "ret_3m":      round(ret_3m, 2),
        "ret_1y":      round(ret_1y, 2)    if not np.isnan(ret_1y)    else None,
        "patterns_zh": " | ".join(f"{zh}({n})" for n,_,zh in patterns) or "无特殊K线形态",
        "signal_1d":   score_to_signal(s1d),
        "signal_1w":   score_to_signal(s1w),
        "signal_1m":   score_to_signal(s1m),
        "signal_3m":   score_to_signal(s3m),
        "signal_1y":   score_to_signal(s1y),
        "why_1d":      " | ".join(w1d),
        "why_1w":      " | ".join(w1w),
        "why_1m":      " | ".join(w1m),
        "why_3m":      " | ".join(w3m),
        "why_1y":      " | ".join(w1y),
    }


# ── Data fetching (cached 15 min) ─────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner="📡 拉取市场数据中，约15秒…")
def fetch_data(tickers_tuple):
    end = date.today() + timedelta(days=1); start = end - timedelta(days=400)
    data = {}
    for t in tickers_tuple:
        for attempt in range(3):
            try:
                df = yf.download(t, start=str(start), end=str(end), auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index.name = "date"
                if not df.empty:
                    data[t] = df.sort_index()
                    break
            except Exception:
                pass
            if attempt < 2:
                time.sleep(1.5)
    return data


# ── Chart builder ─────────────────────────────────────────────────────────────

def scan_patterns(df):
    """Scan all candles in last 90 days, return list of (date, name, zh, bias)."""
    chart_df = df.tail(90)
    out = []
    for i in range(3, len(chart_df)):
        sub = chart_df.iloc[: i + 1]
        for name, bias, zh in detect_patterns(sub):
            out.append((chart_df.index[i], name, zh, bias))
    return out

def make_chart(df, sig_1d):
    chart_df = df.tail(90)
    close    = df["Close"]
    s20  = sma(close, 20).tail(90)
    s50  = sma(close, 50).tail(90)
    s200 = sma(close, 200).tail(90)
    bb_up, _, bb_lo = bollinger(close)
    bb_up = bb_up.tail(90); bb_lo = bb_lo.tail(90)

    fig = go.Figure()

    # BB shaded band
    x_band = list(chart_df.index) + list(chart_df.index[::-1])
    y_band = list(bb_up.values)   + list(bb_lo.values[::-1])
    fig.add_trace(go.Scatter(x=x_band, y=y_band, fill="toself",
        fillcolor="rgba(150,150,150,0.07)", line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=chart_df.index, y=bb_up,
        line=dict(color="rgba(130,130,130,0.55)", width=1, dash="dot"),
        name="BB上轨", showlegend=False))
    fig.add_trace(go.Scatter(x=chart_df.index, y=bb_lo,
        line=dict(color="rgba(130,130,130,0.55)", width=1, dash="dot"),
        name="BB下轨", showlegend=False))

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"],   close=chart_df["Close"],
        increasing_line_color="#1a8f4a", decreasing_line_color="#c84b2b",
        increasing_fillcolor="#1a8f4a", decreasing_fillcolor="#c84b2b",
        name="K线",
    ))

    # SMA lines
    for series, color, name in [(s20,"#2f5e99","SMA20"), (s50,"#a87619","SMA50"), (s200,"#646a73","SMA200")]:
        fig.add_trace(go.Scatter(x=chart_df.index, y=series,
            line=dict(color=color, width=1.5), name=name))

    # Pattern annotations on candles
    for dt, name, zh, bias in scan_patterns(df):
        if dt not in chart_df.index:
            continue
        color  = "#c84b2b" if bias == "bearish" else "#1a8f4a" if bias == "bullish" else "#a87619"
        y_val  = float(chart_df.loc[dt, "High"]) if bias == "bearish" else float(chart_df.loc[dt, "Low"])
        ay     = -28 if bias == "bearish" else 28
        fig.add_annotation(
            x=dt, y=y_val, text=zh[:3],
            showarrow=True, arrowhead=2, arrowcolor=color, arrowsize=1.2,
            font=dict(size=9, color=color),
            ay=ay, ax=0,
            bgcolor="white", bordercolor=color, borderwidth=1, borderpad=2,
        )

    # Today's overall signal — prominent label
    last_dt   = chart_df.index[-1]
    last_high = float(chart_df["High"].iloc[-1])
    last_low  = float(chart_df["Low"].iloc[-1])
    if sig_1d in ("STRONG_BUY", "BUY"):
        fig.add_annotation(x=last_dt, y=last_low, text=f"今日\n{sig_1d}",
            showarrow=True, arrowhead=2, arrowcolor="#0a4d25", arrowsize=1.5,
            font=dict(size=10, color="#0a4d25", family="Arial"),
            ay=42, ax=0, bgcolor="#d4f0de", bordercolor="#0a4d25", borderwidth=2, borderpad=3)
    elif sig_1d in ("STRONG_SELL", "SELL"):
        fig.add_annotation(x=last_dt, y=last_high, text=f"今日\n{sig_1d}",
            showarrow=True, arrowhead=2, arrowcolor="#7a1000", arrowsize=1.5,
            font=dict(size=10, color="#7a1000", family="Arial"),
            ay=-42, ax=0, bgcolor="#fde8e3", bordercolor="#7a1000", borderwidth=2, borderpad=3)

    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=8, b=0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11)),
        paper_bgcolor="white", plot_bgcolor="#fafbfc",
        xaxis=dict(showgrid=True, gridcolor="#efefef"),
        yaxis=dict(showgrid=True, gridcolor="#efefef"),
    )
    return fig


# ── Signal display helpers ─────────────────────────────────────────────────────

SIGNAL_COLOR = {
    "STRONG_BUY": "#1a8f4a", "BUY": "#247d76",
    "HOLD": "#a87619", "SELL": "#c84b2b", "STRONG_SELL": "#8b1a0a",
}
SIGNAL_ZH = {
    "STRONG_BUY": "强力买入", "BUY": "买入",
    "HOLD": "持有", "SELL": "卖出", "STRONG_SELL": "强力卖出",
}

def badge_html(sig):
    c = SIGNAL_COLOR.get(sig, "#646a73"); zh = SIGNAL_ZH.get(sig, sig)
    return (f'<span style="background:{c};color:#fff;padding:4px 12px;'
            f'border-radius:8px;font-size:12px;font-weight:700">{zh}</span>')

def colored_pct(v):
    if v is None: return "—"
    try:
        f = float(v); c = "#1a8f4a" if f >= 0 else "#c84b2b"
        return f'<span style="color:{c};font-weight:600">{f:+.1f}%</span>'
    except Exception:
        return "—"


# ── Password gate ──────────────────────────────────────────────────────────────
# Guards the transaction log / positions, which now live in Google Sheets and are
# reachable (and writable) from any device once unlocked — not just localhost.

def check_password():
    if st.session_state.get("authed"):
        return True
    with st.sidebar:
        pw = st.text_input("🔒 输入密码查看/录入持仓", type="password", key="pw_input")
        if pw:
            if pw == st.secrets.get("app_password"):
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error("密码错误")
    return False


# ── Google Sheets (transaction log) ────────────────────────────────────────────

@st.cache_resource
def _gsheet_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    return gspread.authorize(creds)

def _transactions_ws():
    return _gsheet_client().open_by_key(st.secrets["transactions_sheet_id"]).sheet1

def _buy_plan_ws():
    return _gsheet_client().open_by_key(st.secrets["transactions_sheet_id"]).worksheet("buy_plan")

def load_transactions():
    """Returns the transaction log from Google Sheets, or None if secrets aren't configured."""
    try:
        records = _transactions_ws().get_all_records()
    except Exception as e:
        st.sidebar.error(f"⚠️ 无法连接交易记录表：{e}")
        return None
    df = pd.DataFrame(records, columns=TRANSACTIONS_COLUMNS)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for col in ("shares", "price", "amount_usd"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date")

def compute_positions(txn_df):
    """Weighted-average-cost method. Returns {ticker: {shares, cost_basis, avg_cost}}."""
    positions = {}
    for _, row in txn_df.iterrows():
        t = row["ticker"]
        pos = positions.setdefault(t, {"shares": 0.0, "cost_basis": 0.0})
        if row["action"] == "buy":
            pos["shares"] += row["shares"]
            pos["cost_basis"] += row["amount_usd"]
        elif row["action"] == "sell":
            avg_cost = pos["cost_basis"] / pos["shares"] if pos["shares"] > 0 else 0
            pos["cost_basis"] -= avg_cost * row["shares"]
            pos["shares"] -= row["shares"]
    for pos in positions.values():
        pos["avg_cost"] = pos["cost_basis"] / pos["shares"] if pos["shares"] > 1e-9 else 0.0
    return {t: p for t, p in positions.items() if p["shares"] > 1e-6}

def load_buy_plan():
    """Returns {ticker: {category, monthly_budget_usd, stop_loss_pct, take_profit_pct,
    stop_loss_sell_fraction, take_profit_sell_fraction, min_hold_days}}, read from the
    "buy_plan" tab of the same Google Sheet as transactions — reachable from the public
    deploy too, gated behind the same password check.
    Budget is fresh every calendar month — unused budget does NOT roll over.
    Exit-related fields are only set for category=="speculative" tickers;
    core holdings are buy-and-hold with no exit tracking."""
    try:
        records = _buy_plan_ws().get_all_records()
    except Exception as e:
        st.sidebar.error(f"⚠️ 无法连接月预算配置表：{e}")
        return {}
    return {
        row["ticker"]: {
            "category": row["category"],
            "monthly_budget_usd": pd.to_numeric(row["monthly_budget_usd"], errors="coerce"),
            "stop_loss_pct": pd.to_numeric(row.get("stop_loss_pct"), errors="coerce"),
            "take_profit_pct": pd.to_numeric(row.get("take_profit_pct"), errors="coerce"),
            "stop_loss_sell_fraction": pd.to_numeric(row.get("stop_loss_sell_fraction"), errors="coerce"),
            "take_profit_sell_fraction": pd.to_numeric(row.get("take_profit_sell_fraction"), errors="coerce"),
            "min_hold_days": pd.to_numeric(row.get("min_hold_days"), errors="coerce"),
        }
        for row in records
        if row.get("ticker")
    }

def monthly_spent(txn_df, ticker, today=None):
    """Sum of buy amounts for this ticker in the current calendar month."""
    if txn_df is None:
        return 0.0
    today = today or date.today()
    mask = (
        (txn_df["ticker"] == ticker)
        & (txn_df["action"] == "buy")
        & (txn_df["date"].dt.year == today.year)
        & (txn_df["date"].dt.month == today.month)
    )
    return float(txn_df.loc[mask, "amount_usd"].sum())


# ── Buy-zone (加仓价位) logic ───────────────────────────────────────────────────

BUY_ZONE_TIERS = [("中等回撤", 0.85), ("深度回撤", 0.75)]
TIER_RATIOS = (0.4, 0.6)

def compute_buy_zones(df):
    """Pullback tiers off the trailing ~6-month closing high. Public-safe — no $ amounts."""
    close = df["Close"]
    recent_high = float(close.tail(126).max())
    price = float(close.iloc[-1])
    tiers = []
    for label, mult in BUY_ZONE_TIERS:
        tier_price = recent_high * mult
        tiers.append({
            "label": label,
            "price": tier_price,
            "triggered": price <= tier_price,
            "gap_pct": (price / tier_price - 1) * 100,
        })
    return {"recent_high": recent_high, "price": price, "tiers": tiers}

def render_buy_zones(results, data, positions, budgets, txn_df):
    st.markdown("### 🎯 加仓 / 建仓价位参考")
    st.caption(
        "基于近6个月收盘高点的回撤档位，越跌建议买得越多。技术面参考，不是预测。"
        + ("　|　金额按本月剩余预算计算，每月初自动重置，不累加到下月。" if budgets else "")
    )

    has_personal = bool(positions)
    header = ["现在建议买入", "标的", "现价", "近6月高点", "档2 中等回撤(-15%)", "档3 深度回撤(-25%)"]
    if has_personal:
        header += ["你的持仓"]

    rows = []
    for r in results:
        t = r["ticker"]
        if t not in data:
            continue
        z = compute_buy_zones(data[t])
        plan = budgets.get(t)
        remaining = None
        ticker_cell = f"<strong>{t}</strong>"
        if plan:
            spent = monthly_spent(txn_df, t)
            remaining = max(0.0, plan["monthly_budget_usd"] - spent)
            ticker_cell += (
                f"<br><span style='font-size:10px;color:#646a73'>"
                f"本月已花${spent:.0f}/${plan['monthly_budget_usd']:.0f}</span>"
            )

        triggered_idx = [i for i, tier in enumerate(z["tiers"]) if tier["triggered"]]
        if remaining is None:
            reco_cell = "—"
        elif triggered_idx:
            reco_amount = remaining * TIER_RATIOS[triggered_idx[-1]]
            reco_cell = f"<strong style='color:#1a8f4a;font-size:15px'>${reco_amount:.0f}</strong>"
        else:
            reco_cell = "<span style='color:#646a73'>观望</span>"

        cells = [reco_cell, ticker_cell, f"${z['price']:.2f}", f"${z['recent_high']:.2f}"]
        for tier, ratio in zip(z["tiers"], TIER_RATIOS):
            tag = "✅ 已触发" if tier["triggered"] else f"还差 {tier['gap_pct']:.1f}%"
            color = "#1a8f4a" if tier["triggered"] else "#646a73"
            line = f"${tier['price']:.2f}<br><span style='color:{color};font-size:11px'>{tag}</span>"
            if remaining is not None:
                line += f"<br><span style='font-size:11px;font-weight:700'>建议 ${remaining*ratio:.0f}</span>"
            cells.append(line)
        if has_personal:
            pos = positions.get(t)
            if pos:
                pnl_pct = (z["price"] - pos["avg_cost"]) / pos["avg_cost"] * 100 if pos["avg_cost"] else 0
                pnl_color = "#1a8f4a" if pnl_pct >= 0 else "#c84b2b"
                cells.append(
                    f"{pos['shares']:.2f}股 @ ${pos['avg_cost']:.2f}"
                    f"<br><span style='color:{pnl_color};font-size:11px'>{pnl_pct:+.1f}%</span>"
                )
            else:
                cells.append("—")
        rows.append(cells)

    if not rows:
        return

    html = (
        "<style>.buy-tbl{width:100%;border-collapse:collapse;font-size:13px}"
        ".buy-tbl th{background:#f1f4f6;padding:8px 10px;text-align:center;border-bottom:2px solid #d9dde3;white-space:nowrap}"
        ".buy-tbl td{padding:8px 10px;border-bottom:1px solid #d9dde3;text-align:center;white-space:nowrap}"
        ".buy-tbl td:nth-child(2){text-align:left}</style>"
        '<table class="buy-tbl"><thead><tr>'
        + "".join(f"<th>{h}</th>" for h in header)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
        + "</tbody></table>"
    )
    st.markdown(html, unsafe_allow_html=True)

    if has_personal:
        total_cost = sum(p["cost_basis"] for p in positions.values())
        total_value = sum(
            p["shares"] * compute_buy_zones(data[t])["price"]
            for t, p in positions.items() if t in data
        )
        total_pnl = total_value - total_cost
        total_pnl_pct = total_pnl / total_cost * 100 if total_cost else 0
        pnl_color = "#1a8f4a" if total_pnl >= 0 else "#c84b2b"
        st.markdown(
            f"**持仓总市值：${total_value:,.2f}**　"
            f"<span style='color:{pnl_color}'>总盈亏：{'+' if total_pnl>=0 else ''}{total_pnl:,.2f} "
            f"({total_pnl_pct:+.1f}%)</span>",
            unsafe_allow_html=True,
        )

def render_voo_recurring_check(data):
    """VOO recurring buy was canceled due to overlap with QQQ — flag if that overlap has eased."""
    if "VOO" not in data or "QQQ" not in data:
        return
    voo_ret = data["VOO"]["Close"].pct_change().tail(90)
    qqq_ret = data["QQQ"]["Close"].pct_change().tail(90)
    corr = voo_ret.corr(qqq_ret)
    if pd.isna(corr):
        return
    if corr < 0.85:
        st.success(f"📌 VOO 与 QQQ 近3个月日收益相关性降到 {corr:.2f}（<0.85），重叠度下降，可以考虑恢复 VOO 定投。")
    else:
        st.caption(f"📌 VOO 与 QQQ 近3个月相关性 {corr:.2f}，仍然高度重叠 —— 按计划，VOO 只低点加仓，不恢复定投。")

DEFAULT_MIN_HOLD_DAYS = 30

def render_exit_alerts(positions, budgets, data, txn_df):
    """Take-profit / stop-loss check — only for category=='speculative' tickers.
    Core holdings (VOO/QQQ/NVDA/...) are buy-and-hold by design and never flagged here.
    Two guardrails against over-trading: a minimum holding period before any alert
    becomes actionable, and a partial sell-fraction (never suggests selling 100%)."""
    alerts = []
    for t, plan in budgets.items():
        if plan.get("category") != "speculative":
            continue
        pos = positions.get(t)
        if not pos or t not in data:
            continue
        price = float(data[t]["Close"].iloc[-1])
        pnl_pct = (price - pos["avg_cost"]) / pos["avg_cost"] * 100 if pos["avg_cost"] else 0
        sl, tp = plan.get("stop_loss_pct"), plan.get("take_profit_pct")

        first_buy = txn_df.loc[(txn_df["ticker"] == t) & (txn_df["action"] == "buy"), "date"].min()
        days_held = (pd.Timestamp(date.today()) - first_buy).days if pd.notna(first_buy) else 0
        min_hold = plan.get("min_hold_days")
        min_hold = DEFAULT_MIN_HOLD_DAYS if pd.isna(min_hold) else int(min_hold)
        eligible = days_held >= min_hold

        if pd.notna(sl) and pnl_pct <= sl:
            frac = plan.get("stop_loss_sell_fraction")
            frac = 0.5 if pd.isna(frac) else float(frac)
            alerts.append(("loss", t, pnl_pct, sl, frac, eligible, days_held, min_hold))
        elif pd.notna(tp) and pnl_pct >= tp:
            frac = plan.get("take_profit_sell_fraction")
            frac = 0.33 if pd.isna(frac) else float(frac)
            alerts.append(("profit", t, pnl_pct, tp, frac, eligible, days_held, min_hold))

    if not alerts:
        return
    st.markdown("### ⚠️ 投机仓位止盈/止损提醒")
    st.caption("只针对投机性小仓位（XMAX/SPCX）。未满最短持有期不会建议操作；触发时只建议卖出一部分，不会建议清仓。")
    for kind, t, pnl_pct, threshold, frac, eligible, days_held, min_hold in alerts:
        shares = positions[t]["shares"] * frac
        if kind == "loss":
            if eligible:
                st.error(
                    f"🔴 **{t}** 浮亏 {pnl_pct:+.1f}%，跌破止损线 {threshold:.0f}%（已持有{days_held}天）。"
                    f"建议卖出约 {frac*100:.0f}%（≈{shares:.2f}股），保留剩余继续观察。"
                )
            else:
                st.warning(
                    f"🟡 **{t}** 浮亏 {pnl_pct:+.1f}% 已跌破止损线，但只持有{days_held}天"
                    f"（最短持有期{min_hold}天），先观察，暂不建议操作。"
                )
        else:
            if eligible:
                st.success(
                    f"🟢 **{t}** 浮盈 {pnl_pct:+.1f}%，达到止盈线 {threshold:.0f}%（已持有{days_held}天）。"
                    f"建议卖出约 {frac*100:.0f}%（≈{shares:.2f}股）锁定部分利润，剩余继续持有。"
                )
            else:
                st.info(
                    f"🔵 **{t}** 浮盈 {pnl_pct:+.1f}% 已达止盈线，但只持有{days_held}天"
                    f"（最短持有期{min_hold}天），先观察，暂不建议操作。"
                )


# ── Main App ───────────────────────────────────────────────────────────────────

HOLDINGS  = ["VOO", "QQQ", "NVDA"]          # 我现在持有的
WATCHLIST = ["SMH", "FCX", "AMAT", "LRCX", "MU", "ETN", "GEV", "MOD", "JBL"]  # 观察中，还没买

def render_table(results):
    header = ["标的", "价格", "今日 1D", "本周 1W", "本月 1M", "三月 3M", "全年 1Y", "1周±", "1月±", "3月±", "1年±"]
    rows = []
    for r in results:
        rows.append([
            f"<strong>{r['ticker']}</strong>",
            f"${r['price']:.2f}",
            badge_html(r["signal_1d"]),
            badge_html(r["signal_1w"]),
            badge_html(r["signal_1m"]),
            badge_html(r["signal_3m"]),
            badge_html(r["signal_1y"]),
            colored_pct(r["ret_1w"]),
            colored_pct(r["ret_1m"]),
            colored_pct(r["ret_3m"]),
            colored_pct(r.get("ret_1y")),
        ])
    html = (
        "<style>.sum-tbl{width:100%;border-collapse:collapse;font-size:13px}"
        ".sum-tbl th{background:#f1f4f6;padding:8px 10px;text-align:center;border-bottom:2px solid #d9dde3;white-space:nowrap}"
        ".sum-tbl td{padding:8px 10px;border-bottom:1px solid #d9dde3;text-align:center;white-space:nowrap}"
        ".sum-tbl td:first-child{text-align:left}</style>"
        '<table class="sum-tbl"><thead><tr>'
        + "".join(f"<th>{h}</th>" for h in header)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
        + "</tbody></table>"
    )
    st.markdown(html, unsafe_allow_html=True)

SIGNAL_EMOJI = {
    "STRONG_BUY": "🟢🟢", "BUY": "🟢",
    "HOLD": "🟡", "SELL": "🔴", "STRONG_SELL": "🔴🔴",
}

def render_charts(results, data, label):
    st.caption(f"{label}  |  绿色↑ = 看涨形态  |  红色↓ = 看跌形态  |  虚线 = Bollinger Bands")
    for r in results:
        t      = r["ticker"]
        sig_1d = r["signal_1d"]
        sig_zh = SIGNAL_ZH.get(sig_1d, sig_1d)
        emoji  = SIGNAL_EMOJI.get(sig_1d, "")
        label_str = f"{t}   ${r['price']:.2f}   —   今日：{emoji} {sig_zh}"
        with st.expander(label_str, expanded=(t in ("QQQ", "NVDA"))):
            c1, c2, c3, c4 = st.columns(4)
            bb_val = r.get("bb_pct"); vr_val = r.get("vol_ratio")
            c1.metric("RSI", f"{r['rsi14']:.1f}",
                      delta=("超买 Overbought" if r["rsi14"]>70 else "超卖 Oversold" if r["rsi14"]<30 else None),
                      delta_color="inverse")
            c2.metric("Bollinger Bands 位置", f"{bb_val:.0f}%" if bb_val is not None else "—",
                      help="0%=下轨  100%=上轨")
            c3.metric("ATR 日均波动", f"${r['atr14']:.2f}")
            c4.metric("成交量/20日均", f"{vr_val:.1f}×" if vr_val else "—")

            cols = st.columns(5)
            for col, (tf, key) in zip(cols, [
                ("今日 1D","signal_1d"),("本周 1W","signal_1w"),
                ("本月 1M","signal_1m"),("三月 3M","signal_3m"),("全年 1Y","signal_1y"),
            ]):
                s = r[key]
                col.markdown(
                    f"<div style='text-align:center'>"
                    f"<div style='font-size:11px;color:#646a73;margin-bottom:4px'>{tf}</div>"
                    f"<span style='background:{SIGNAL_COLOR.get(s,'#666')};color:#fff;"
                    f"padding:4px 12px;border-radius:8px;font-size:12px;font-weight:700'>"
                    f"{SIGNAL_ZH.get(s,s)}</span></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("")
            fig = make_chart(data[t], sig_1d)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            p_col, w_col = st.columns([1, 2])
            with p_col:
                st.markdown("**📌 今日K线形态**")
                st.markdown(r["patterns_zh"])
            with w_col:
                st.markdown("**🔍 信号理由**")
                for tf, key in [("今日","why_1d"),("本周","why_1w"),("本月","why_1m"),("三月","why_3m"),("全年","why_1y")]:
                    st.markdown(
                        f"<span style='font-weight:700;min-width:32px;display:inline-block'>{tf}</span>"
                        f"<span style='color:#646a73;font-size:13px'>&nbsp;{r.get(key,'—')}</span>",
                        unsafe_allow_html=True,
                    )

def _price_on(ticker, d):
    """Closing price on date d, or the nearest earlier trading day (weekends/holidays)."""
    hist = yf.download(ticker, start=str(d - timedelta(days=7)), end=str(d + timedelta(days=1)),
                        progress=False, auto_adjust=True)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    if hist.empty:
        return None
    return float(hist["Close"].iloc[-1])

def render_transaction_form(txn_df):
    """Sidebar form to record a new buy/sell transaction, written to Google Sheets.
    Only called once the password gate in check_password() has been passed.
    Takes a dollar amount (matches how Robinhood dollar-based buys work) — price is
    looked up automatically and shares are derived, so nothing needs to be calculated by hand."""
    st.divider()
    st.markdown("### 📝 录入交易")

    with st.form("txn_form", clear_on_submit=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker_raw = st.text_input("股票代码", placeholder="NVDA")
        with col2:
            action = st.selectbox("操作", ["buy", "sell"])

        txn_date = st.date_input("日期", value=date.today())
        amount = st.number_input("金额 ($)", min_value=0.0, step=1.0,
                                 format="%.2f", value=0.0)

        notes = st.text_input("备注（可选）", placeholder="limit buy / 定投 …")
        st.caption("股数会用当天收盘价自动换算，不用自己算")
        submitted = st.form_submit_button("✅ 确认录入", use_container_width=True,
                                          type="primary")

    if submitted:
        ticker = ticker_raw.strip().upper()
        if not ticker:
            st.error("请输入股票代码")
        elif amount <= 0:
            st.error("金额必须 > 0")
        else:
            with st.spinner(f"查询 {ticker} 在 {txn_date} 的价格…"):
                price = _price_on(ticker, txn_date)
            if price is None:
                st.error(f"⚠️ 查不到 {ticker} 的价格，请检查股票代码是否正确")
            else:
                shares = amount / price
                try:
                    _transactions_ws().append_row([
                        str(txn_date), ticker, action,
                        round(shares, 6), round(price, 2), round(amount, 2), notes,
                    ])
                except Exception as e:
                    st.error(f"⚠️ 写入交易记录表失败：{e}")
                else:
                    st.success(f"✅ {action.upper()} {ticker} ${amount:.2f} @ ${price:.2f} ≈ {shares:.4f}股")
                    st.cache_data.clear()
                    st.rerun()

    # Show last 3 transactions as confirmation
    if txn_df is not None and not txn_df.empty:
        st.caption("最近录入")
        for _, row in txn_df.tail(3).iterrows():
            emoji = "🟢" if row["action"] == "buy" else "🔴"
            st.caption(f"{emoji} {row['date'].date()}  {row['ticker']}  "
                       f"{row['shares']:.4f}股 @${row['price']:.2f}")


def main():
    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("## 📈 Abby 投资信号仪表盘")
    st.caption(f"数据来源：Yahoo Finance（15分钟延迟）&nbsp;&nbsp;|&nbsp;&nbsp;更新时间：{date.today()}")

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ 设置")
        custom_input = st.text_input(
            "➕ 临时添加标的", placeholder="TSLA, MSFT …",
            help="输入股票代码（英文大写），加到观察名单"
        )
        extra_tickers = []
        if custom_input:
            extra_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]

        if st.button("🔄 立即刷新数据", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.markdown("**我的持仓**")
        for t in HOLDINGS:  st.markdown(f"- {t}")
        st.markdown("**观察名单**")
        for t in WATCHLIST + extra_tickers: st.markdown(f"- {t}")
        st.divider()
        st.markdown("🟢 绿色↑ = 看涨K线形态\n\n🔴 红色↓ = 看跌K线形态\n\n虚线 = Bollinger Bands")

        unlocked = check_password()
        if unlocked:
            txn_df = load_transactions()
            render_transaction_form(txn_df)
        else:
            txn_df = None

    # ── Personal data (requires password; None/{} otherwise) ────────────────
    positions = compute_positions(txn_df) if txn_df is not None and not txn_df.empty else {}
    budgets   = load_buy_plan() if unlocked else {}
    extra_holdings = [t for t in positions if t not in HOLDINGS]

    # ── Fetch ─────────────────────────────────────────────────────────────
    all_tickers = list(dict.fromkeys(HOLDINGS + extra_holdings + WATCHLIST + extra_tickers))
    data = fetch_data(tuple(all_tickers + ["SPY"]))
    if "SPY" not in data:
        st.error("🚫 网络错误，无法获取数据。请点击侧边栏 [立即刷新数据] 重试。")
        return

    bench = data["SPY"]["Close"]

    def get_results(tickers):
        out = []
        for t in tickers:
            if t not in data:
                st.warning(f"⚠️ {t} 数据获取失败")
                continue
            r = analyze_df(t, data[t], bench)
            if "error" not in r:
                out.append(r)
        return out

    holding_results  = get_results(HOLDINGS + extra_holdings)
    watchlist_results = get_results(WATCHLIST + extra_tickers)

    # ── 我的持仓 ──────────────────────────────────────────────────────────
    st.markdown("### 💼 我的持仓")
    st.caption("核心仓位（VOO/QQQ/NVDA等）买入持有为主：表里的 SELL/STRONG_SELL 只是技术面偏弱的提示，不是卖出建议。真正的止盈止损提醒只针对投机仓位，见下方。")
    st.caption("⚠️ 下面 1D/1W/1M/3M/1Y 五个维度是技术面参考，不是操作指令——真正「现在该买多少钱」只有一个答案，看最下方「🎯加仓价位参考」表里的「现在建议买入」那一栏。")
    if not unlocked:
        st.caption("🔒 在侧边栏输入密码可以查看/录入你的持仓成本、盈亏和交易记录（技术信号上方任何人都能看）。")
    if holding_results:
        render_table(holding_results)
        render_voo_recurring_check(data)
        render_exit_alerts(positions, budgets, data, txn_df)
        st.divider()
        render_charts(holding_results, data, "点击展开 K 线图")

    # ── 观察名单 ──────────────────────────────────────────────────────────
    st.markdown("### 👀 观察名单 Watchlist")
    st.caption("还没买，在看时机")
    st.caption("⚠️ 下面 1D/1W/1M/3M/1Y 五个维度是技术面参考，不是操作指令——真正「现在该买多少钱」只有一个答案，看最下方「🎯加仓价位参考」表里的「现在建议买入」那一栏。")
    if watchlist_results:
        render_table(watchlist_results)
        st.divider()
        render_charts(watchlist_results, data, "点击展开 K 线图")

    # ── 加仓价位参考 ──────────────────────────────────────────────────────
    st.divider()
    render_buy_zones(holding_results + watchlist_results, data, positions, budgets, txn_df)


if __name__ == "__main__":
    main()
