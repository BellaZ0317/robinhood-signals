"""
signals.py
Compute trading signals for 5 timeframes: 1D, 1W, 1M, 3M, 1Y
Indicators: RSI, SMA, MACD, Bollinger Bands, ATR, Volume, Candlestick patterns
Run: python3 scripts/signals.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

PRICES_DIR = Path(__file__).parent.parent / "data" / "prices"
DATA_DIR   = Path(__file__).parent.parent / "data"

TICKERS   = ["VOO", "QQQ", "QQQM", "NVDA", "SMH"]
BENCHMARK = "SPY"


# ── Indicator helpers ─────────────────────────────────────────────────────────

def sma(s, n):
    return s.rolling(n).mean()

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

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
    mid = sma(s, n)
    std = s.rolling(n).std()
    return mid + k * std, mid, mid - k * std

def atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(n).mean()

def pct(s, n):
    if len(s) < n + 1:
        return np.nan
    return (s.iloc[-1] / s.iloc[-n] - 1) * 100


# ── Candlestick pattern detection ─────────────────────────────────────────────

def detect_patterns(df):
    """Returns list of (name_en, bias, description_zh) for the latest candles."""
    if len(df) < 3:
        return []

    patterns = []
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    i = len(df) - 1  # latest candle

    rng   = h[i] - l[i]
    body  = abs(c[i] - o[i])
    upper = h[i] - max(c[i], o[i])
    lower = min(c[i], o[i]) - l[i]
    bull  = c[i] > o[i]

    if rng < 0.001:
        return patterns

    br = body / rng  # body ratio

    # Single-candle
    if br < 0.1:
        patterns.append(("Doji", "neutral", "十字星：开收价几乎相同，市场犹豫，可能变盘"))
    elif lower / rng > 0.6 and br < 0.35 and bull:
        patterns.append(("Hammer", "bullish", "锤子线：下影线长，下跌后出现，看涨反转信号"))
    elif lower / rng > 0.6 and br < 0.35 and not bull:
        patterns.append(("Hanging Man", "bearish", "吊颈线：上涨后下影线长，看跌反转信号"))
    elif upper / rng > 0.6 and br < 0.35 and not bull:
        patterns.append(("Shooting Star", "bearish", "射击之星：上影线长，上涨后出现，看跌反转信号"))
    elif upper / rng > 0.6 and br < 0.35 and bull:
        patterns.append(("Inverted Hammer", "bullish", "倒锤子线：下跌后出现，潜在看涨信号"))
    elif br > 0.85:
        if bull:
            patterns.append(("Bullish Marubozu", "bullish", "大阳线：几乎无影线，多头强势，动能充足"))
        else:
            patterns.append(("Bearish Marubozu", "bearish", "大阴线：几乎无影线，空头强势，注意风险"))

    # Two-candle patterns
    if i >= 1:
        p_lo = min(c[i - 1], o[i - 1])
        p_hi = max(c[i - 1], o[i - 1])
        c_lo = min(c[i], o[i])
        c_hi = max(c[i], o[i])
        p_bear = c[i - 1] < o[i - 1]
        p_bull = c[i - 1] > o[i - 1]

        if p_bear and bull and c_lo < p_lo and c_hi > p_hi:
            patterns.append(("Bullish Engulfing", "bullish", "看涨吞没：阳线完全吞没前一阴线，多头强力反转"))
        elif p_bull and not bull and c_lo < p_lo and c_hi > p_hi:
            patterns.append(("Bearish Engulfing", "bearish", "看跌吞没：阴线完全吞没前一阳线，空头强力反转"))

    # Three-candle patterns
    if i >= 2:
        mid_rng  = h[i - 1] - l[i - 1]
        mid_body = abs(c[i - 1] - o[i - 1])
        small_mid = mid_body / (mid_rng + 0.001) < 0.3
        first_bear = c[i - 2] < o[i - 2]
        first_bull = c[i - 2] > o[i - 2]

        if first_bear and small_mid and bull:
            if c[i] > (o[i - 2] + c[i - 2]) / 2:
                patterns.append(("Morning Star", "bullish", "早晨之星：底部三K反转，强烈看涨信号"))

        if first_bull and small_mid and not bull:
            if c[i] < (o[i - 2] + c[i - 2]) / 2:
                patterns.append(("Evening Star", "bearish", "黄昏之星：顶部三K反转，强烈看跌信号"))

    return patterns


# ── Data loading ──────────────────────────────────────────────────────────────

def load(ticker):
    p = PRICES_DIR / f"{ticker}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col="date", parse_dates=True)
    return df.sort_index()


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_to_signal(score):
    if   score >=  1.5: return "STRONG_BUY"
    elif score >=  0.5: return "BUY"
    elif score >= -0.5: return "HOLD"
    elif score >= -1.5: return "SELL"
    else:               return "STRONG_SELL"


# ── Main analysis per ticker ───────────────────────────────────────────────────

def analyze(ticker, bench_close):
    df = load(ticker)
    if df is None or len(df) < 60:
        return {"ticker": ticker, "error": "insufficient data"}

    close = df["Close"]
    today = close.index[-1].date()
    price = close.iloc[-1]

    # Core indicators
    rsi14      = rsi(close).iloc[-1]
    s20        = sma(close, 20).iloc[-1]
    s50        = sma(close, 50).iloc[-1]
    s200       = sma(close, 200).iloc[-1] if len(close) >= 200 else np.nan
    ml, sl, hist = macd(close)
    bb_up, _, bb_lo = bollinger(close)
    bb_range   = bb_up.iloc[-1] - bb_lo.iloc[-1]
    bb_pct     = (price - bb_lo.iloc[-1]) / bb_range * 100 if bb_range > 0 else np.nan
    atr14      = atr(df).iloc[-1]
    vol        = df["Volume"].iloc[-1]      if "Volume" in df.columns else np.nan
    vol20_avg  = df["Volume"].rolling(20).mean().iloc[-1] if "Volume" in df.columns else np.nan
    vol_ratio  = vol / vol20_avg if not np.isnan(vol20_avg) and vol20_avg > 0 else np.nan

    # Returns
    ret_1w = pct(close, 5)
    ret_1m = pct(close, 21)
    ret_3m = pct(close, 63)
    ret_1y = pct(close, 252)
    b_1w   = pct(bench_close, 5)
    b_1m   = pct(bench_close, 21)
    b_3m   = pct(bench_close, 63)
    b_1y   = pct(bench_close, 252)

    # Candlestick patterns
    patterns     = detect_patterns(df)
    pattern_bias = sum(1 if b == "bullish" else -1 if b == "bearish" else 0 for _, b, _ in patterns)

    # ── 1D signal (日内/当日 参考) ──────────────────────────────────────────
    s1d = 0.0; w1d = []

    if   rsi14 < 30: s1d += 2.0; w1d.append(f"RSI超卖({rsi14:.0f})")
    elif rsi14 < 40: s1d += 1.0; w1d.append(f"RSI偏低({rsi14:.0f})")
    elif rsi14 > 70: s1d -= 2.0; w1d.append(f"RSI超买({rsi14:.0f})")
    elif rsi14 > 60: s1d -= 1.0; w1d.append(f"RSI偏高({rsi14:.0f})")
    else:                         w1d.append(f"RSI中性({rsi14:.0f})")

    if not np.isnan(bb_pct):
        if   bb_pct < 20: s1d += 1.0; w1d.append(f"接近布林下轨({bb_pct:.0f}%)")
        elif bb_pct > 80: s1d -= 1.0; w1d.append(f"接近布林上轨({bb_pct:.0f}%)")
        else:                          w1d.append(f"布林带中区({bb_pct:.0f}%)")

    s1d += pattern_bias * 0.5
    for name, bias, _ in patterns:
        w1d.append(f"{name}({bias})")

    if not np.isnan(vol_ratio):
        if   vol_ratio > 1.5 and s1d > 0:  s1d += 0.5; w1d.append(f"放量确认(×{vol_ratio:.1f})")
        elif vol_ratio > 1.5 and s1d < 0:  s1d -= 0.5; w1d.append(f"放量下跌(×{vol_ratio:.1f})")

    # ── 1W signal ──────────────────────────────────────────────────────────
    s1w = 0.0; w1w = []
    if   rsi14 < 35: s1w += 1.5; w1w.append(f"RSI oversold ({rsi14:.0f})")
    elif rsi14 < 45: s1w += 0.5; w1w.append(f"RSI low ({rsi14:.0f})")
    elif rsi14 > 70: s1w -= 1.5; w1w.append(f"RSI overbought ({rsi14:.0f})")
    elif rsi14 > 60: s1w -= 0.5; w1w.append(f"RSI high ({rsi14:.0f})")
    else:                         w1w.append(f"RSI neutral ({rsi14:.0f})")
    s1w += 0.5 if price > s20 else -0.5
    w1w.append("price > SMA20" if price > s20 else "price < SMA20")
    s1w += 0.5 if hist.iloc[-1] > 0 else -0.5
    w1w.append("MACD hist +" if hist.iloc[-1] > 0 else "MACD hist −")

    # ── 1M signal ──────────────────────────────────────────────────────────
    s1m = 0.0; w1m = []
    s1m += 1.0 if price > s50 else -1.0
    w1m.append("price > SMA50" if price > s50 else "price < SMA50")
    s1m += 0.5 if ml.iloc[-1] > sl.iloc[-1] else -0.5
    w1m.append("MACD above signal" if ml.iloc[-1] > sl.iloc[-1] else "MACD below signal")
    a1m = ret_1m - b_1m if not np.isnan(b_1m) else 0
    if a1m > 2:   s1m += 0.5; w1m.append(f"outperforms SPY +{a1m:.1f}%")
    elif a1m < -2: s1m -= 0.5; w1m.append(f"lags SPY {a1m:.1f}%")

    # ── 3M signal ──────────────────────────────────────────────────────────
    s3m = 0.0; w3m = []
    s3m += 0.5 if price > s50 else -0.5
    w3m.append("price > SMA50" if price > s50 else "price < SMA50")
    if not np.isnan(s200):
        s3m += 1.0 if price > s200 else -1.0
        w3m.append("price > SMA200" if price > s200 else "price < SMA200")
    a3m = ret_3m - b_3m if not np.isnan(b_3m) else 0
    if a3m > 3:   s3m += 0.5; w3m.append(f"3M alpha +{a3m:.1f}%")
    elif a3m < -3: s3m -= 0.5; w3m.append(f"3M alpha {a3m:.1f}%")

    # ── 1Y signal ──────────────────────────────────────────────────────────
    s1y = 0.0; w1y = []
    if not np.isnan(s200):
        s1y += 1.5 if price > s200 else -1.5
        w1y.append("price > SMA200 (bull trend)" if price > s200 else "price < SMA200 (bear trend)")
    a1y = ret_1y - b_1y if (not np.isnan(b_1y) and not np.isnan(ret_1y)) else 0
    if a1y > 5:   s1y += 0.5; w1y.append(f"1Y alpha +{a1y:.1f}% vs SPY")
    elif a1y < -5: s1y -= 0.5; w1y.append(f"1Y alpha {a1y:.1f}% vs SPY")

    return {
        "ticker":       ticker,
        "date":         str(today),
        "price":        round(price, 2),
        "rsi14":        round(rsi14, 1),
        "sma20":        round(s20, 2),
        "sma50":        round(s50, 2),
        "sma200":       round(s200, 2) if not np.isnan(s200) else None,
        "bb_upper":     round(bb_up.iloc[-1], 2),
        "bb_lower":     round(bb_lo.iloc[-1], 2),
        "bb_pct":       round(bb_pct, 1)   if not np.isnan(bb_pct)   else None,
        "atr14":        round(atr14, 2),
        "vol_ratio":    round(vol_ratio, 2) if not np.isnan(vol_ratio) else None,
        "ret_1w":       round(ret_1w, 2),
        "ret_1m":       round(ret_1m, 2),
        "ret_3m":       round(ret_3m, 2),
        "ret_1y":       round(ret_1y, 2)   if not np.isnan(ret_1y)   else None,
        "patterns":     " | ".join(f"{n}({b})" for n, b, _ in patterns) or "None",
        "patterns_zh":  " | ".join(d for _, _, d in patterns) or "无特殊K线形态",
        "signal_1d":    score_to_signal(s1d),
        "signal_1w":    score_to_signal(s1w),
        "signal_1m":    score_to_signal(s1m),
        "signal_3m":    score_to_signal(s3m),
        "signal_1y":    score_to_signal(s1y),
        "why_1d":       " | ".join(w1d),
        "why_1w":       " | ".join(w1w),
        "why_1m":       " | ".join(w1m),
        "why_3m":       " | ".join(w3m),
        "why_1y":       " | ".join(w1y),
    }


# ── Console output ────────────────────────────────────────────────────────────

EMOJI = {
    "STRONG_BUY":  "🟢🟢",
    "BUY":         "🟢  ",
    "HOLD":        "🟡  ",
    "SELL":        "🔴  ",
    "STRONG_SELL": "🔴🔴",
}

def print_report(rows):
    print(f"\n{'─'*80}")
    print(f"  Signal Report  {date.today()}")
    print(f"{'─'*80}\n")
    hdr = f"{'Ticker':<6}  {'Price':>8}  {'RSI':>5}  {'BB%':>5}  {'ATR':>6}  {'1D':^12}  {'1W':^12}  {'1M':^12}  {'3M':^12}  {'1Y':^12}"
    print(hdr)
    print("─" * len(hdr))

    for r in rows:
        if "error" in r:
            print(f"{r['ticker']:<6}  {r['error']}")
            continue
        r1y = f"{r['ret_1y']:>+6.1f}%" if r["ret_1y"] is not None else "   n/a"
        print(
            f"{r['ticker']:<6}  ${r['price']:>7.2f}  {r['rsi14']:>5.1f}  "
            f"{(str(r['bb_pct']) if r['bb_pct'] is not None else 'n/a'):>5}  "
            f"{r['atr14']:>6.2f}  "
            f"{EMOJI[r['signal_1d']]} {r['ret_1w']:>+5.1f}%  "
            f"{EMOJI[r['signal_1w']]} {r['ret_1w']:>+5.1f}%  "
            f"{EMOJI[r['signal_1m']]} {r['ret_1m']:>+5.1f}%  "
            f"{EMOJI[r['signal_3m']]} {r['ret_3m']:>+5.1f}%  "
            f"{EMOJI[r['signal_1y']]} {r1y}"
        )

    print(f"\n{'─'*80}")
    print("K线形态 & 信号详情:\n")
    for r in rows:
        if "error" in r:
            continue
        print(f"  {r['ticker']}  (布林带位置: {r['bb_pct']}%  ATR: {r['atr14']})")
        print(f"    K线: {r['patterns_zh']}")
        print(f"    1D: {r['why_1d']}")
        print(f"    1W: {r['why_1w']}")
        print(f"    1M: {r['why_1m']}")
        print(f"    3M: {r['why_3m']}")
        print(f"    1Y: {r['why_1y']}")
        print()


def main():
    bench = load(BENCHMARK)
    if bench is None:
        print(f"Benchmark {BENCHMARK} not found. Run fetch_prices.py first.")
        return
    bench_close = bench["Close"]

    rows = [analyze(t, bench_close) for t in TICKERS]
    print_report(rows)

    cols = [
        "ticker", "date", "price", "rsi14", "sma20", "sma50", "sma200",
        "bb_upper", "bb_lower", "bb_pct", "atr14", "vol_ratio",
        "ret_1w", "ret_1m", "ret_3m", "ret_1y",
        "patterns", "patterns_zh",
        "signal_1d", "signal_1w", "signal_1m", "signal_3m", "signal_1y",
        "why_1d", "why_1w", "why_1m", "why_3m", "why_1y",
    ]
    out = DATA_DIR / f"signals_{date.today()}.csv"
    pd.DataFrame(
        [r for r in rows if "error" not in r], columns=cols
    ).to_csv(out, index=False)
    print(f"Saved → {out.name}\n")


if __name__ == "__main__":
    main()
