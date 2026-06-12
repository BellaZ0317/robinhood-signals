"""
report.py
Generate an HTML signal report with annotated candlestick (K-line) charts.
- Pattern markers directly on chart candles
- Bollinger Bands overlay
- Beginner-friendly explanations in Chinese
Run: python3 scripts/report.py
"""

import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent))
from signals import detect_patterns, bollinger, sma

PRICES_DIR  = Path(__file__).parent.parent / "data" / "prices"
DATA_DIR    = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

TICKERS    = ["VOO", "QQQ", "NVDA", "SMH"]
BENCHMARK  = "SPY"
CHART_DAYS = 90


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_price(ticker, extra=60):
    p = PRICES_DIR / f"{ticker}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col="date", parse_dates=True).sort_index()
    return df.tail(CHART_DAYS + extra)

def df_to_candles(df):
    return [
        {"time": dt.strftime("%Y-%m-%d"),
         "open": round(float(r["Open"]), 2), "high": round(float(r["High"]), 2),
         "low":  round(float(r["Low"]),  2), "close": round(float(r["Close"]),2)}
        for dt, r in df.iterrows()
    ]

def df_to_line(idx, series):
    return [
        {"time": dt.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
        for dt, v in zip(idx, series) if not np.isnan(float(v))
    ]

def load_signals():
    f = DATA_DIR / f"signals_{date.today()}.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    return {r["ticker"]: r.to_dict() for _, r in df.iterrows()}


# ── Chart annotation helpers ──────────────────────────────────────────────────

# Patterns we care about (English name → Chinese short label for chart)
PATTERN_LABEL = {
    "Doji":              "Doji十字",
    "Hammer":            "Hammer锤",
    "Hanging Man":       "HangingMan吊",
    "Shooting Star":     "ShootingStar射",
    "Inverted Hammer":   "InvHammer倒锤",
    "Bullish Marubozu":  "大阳线",
    "Bearish Marubozu":  "大阴线",
    "Bullish Engulfing": "Engulf吞↑",
    "Bearish Engulfing": "Engulf吞↓",
    "Morning Star":      "MorningStar晨",
    "Evening Star":      "EveningStar昏",
}

def get_markers(chart_df, sig_1d):
    """Scan every candle and return Lightweight Charts marker objects."""
    raw = []
    for i in range(3, len(chart_df)):
        sub  = chart_df.iloc[: i + 1]
        pats = detect_patterns(sub)
        if not pats:
            continue
        dt = chart_df.index[i].strftime("%Y-%m-%d")
        for name, bias, _ in pats:
            label = PATTERN_LABEL.get(name, name)
            if bias == "bullish":
                raw.append({"time": dt, "position": "belowBar", "shape": "arrowUp",
                            "color": "#1a8f4a", "text": label})
            elif bias == "bearish":
                raw.append({"time": dt, "position": "aboveBar", "shape": "arrowDown",
                            "color": "#c84b2b", "text": label})
            else:
                raw.append({"time": dt, "position": "inBar", "shape": "circle",
                            "color": "#a87619", "text": label})

    # Today's overall 1D signal — large bold arrow
    last_dt = chart_df.index[-1].strftime("%Y-%m-%d")
    if sig_1d in ("STRONG_BUY", "BUY"):
        raw.append({"time": last_dt, "position": "belowBar", "shape": "arrowUp",
                    "color": "#0a4d25", "size": 2, "text": f"今日:{sig_1d}"})
    elif sig_1d in ("STRONG_SELL", "SELL"):
        raw.append({"time": last_dt, "position": "aboveBar", "shape": "arrowDown",
                    "color": "#7a1000", "size": 2, "text": f"今日:{sig_1d}"})

    # Sort by time (required by Lightweight Charts)
    raw.sort(key=lambda x: x["time"])
    return raw

def get_bb_lines(df):
    close = df["Close"]
    bb_up, bb_mid, bb_lo = bollinger(close)
    idx = df.index
    return (df_to_line(idx, bb_up.values),
            df_to_line(idx, bb_mid.values),
            df_to_line(idx, bb_lo.values))


# ── Signal display ────────────────────────────────────────────────────────────

SIGNAL_META = {
    "STRONG_BUY":  ("#1a8f4a", "强力买入"),
    "BUY":         ("#247d76", "买入"),
    "HOLD":        ("#a87619", "持有观望"),
    "SELL":        ("#c84b2b", "卖出"),
    "STRONG_SELL": ("#8b1a0a", "强力卖出"),
}

def badge(sig):
    c, zh = SIGNAL_META.get(sig, ("#646a73", sig))
    return f'<span class="badge" style="background:{c}">{zh}<br><small style="font-weight:400;font-size:10px">{sig}</small></span>'

def na(v, fmt=".2f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return format(float(v), fmt)

def fmt_ret(v):
    try:
        f = float(v)
        cls = "green" if f >= 0 else "red"
        return f'<span class="{cls}">{f:+.2f}%</span>'
    except Exception:
        return "—"


# ── Educational section (beginner-friendly) ───────────────────────────────────

def educational_section():
    return """
<details class="edu-block panel">
  <summary><strong>📖 新手必读：图表里的东西是什么意思？</strong>（点击展开）</summary>

  <div class="edu-grid">

    <div class="edu-card">
      <h3>K线图（Candlestick Chart）是什么？</h3>
      <p>每一根"蜡烛"代表一段时间（这里是一天）的价格变化，包含 4 个数字：</p>
      <div class="candle-demo">
        <div class="candle-explain">
          <div class="wick-top"></div>
          <div class="candle-body green-body"></div>
          <div class="wick-bottom"></div>
        </div>
        <div class="candle-labels">
          <div style="margin-top:0px">← 最高价 <strong>High</strong></div>
          <div style="margin-top:12px">← 收盘价 <strong>Close</strong>（绿色=涨）</div>
          <div style="margin-top:8px">← 开盘价 <strong>Open</strong></div>
          <div style="margin-top:12px">← 最低价 <strong>Low</strong></div>
        </div>
      </div>
      <p><span class="tag-green">绿色K线（阳线）</span> = 今天收盘比开盘高，价格上涨<br>
         <span class="tag-red">红色K线（阴线）</span> = 今天收盘比开盘低，价格下跌</p>
      <p>上面的细线叫 <strong>上影线（Upper Wick/Shadow）</strong>，表示当天摸到的最高价但没守住。<br>
         下面的细线叫 <strong>下影线（Lower Wick/Shadow）</strong>，表示当天跌到的最低价但反弹了。</p>
    </div>

    <div class="edu-card">
      <h3>图表上的彩色线是什么？</h3>
      <p><span class="dot" style="background:#2f5e99"></span><strong>SMA20（20日均线）</strong><br>
        过去20个交易日的平均收盘价。价格在它上面 = 短期偏强，在它下面 = 短期偏弱。</p>
      <p><span class="dot" style="background:#a87619"></span><strong>SMA50（50日均线）</strong><br>
        过去50天平均价。判断中期趋势是涨是跌。</p>
      <p><span class="dot" style="background:#646a73"></span><strong>SMA200（200日均线）</strong><br>
        过去200天平均价。最重要的长期趋势线。价格在它上面 = 牛市结构，在它下面 = 熊市结构。</p>
      <p><span class="dot" style="background:#9b9b9b;border:1px dashed #333"></span><strong>Bollinger Bands（布林带）</strong>（虚线）<br>
        上下两条灰色虚线。价格靠近<strong>下轨</strong> = 可能超卖、低估区间；靠近<strong>上轨</strong> = 可能超买、高估区间。中间宽度越窄，说明市场越"蓄势"，随时可能大幅波动。</p>
    </div>

    <div class="edu-card">
      <h3>RSI 是什么？</h3>
      <p><strong>RSI（Relative Strength Index，相对强弱指数）</strong>是一个 0–100 的数字，衡量最近一段时间买方力量和卖方力量的对比。</p>
      <ul class="edu-list">
        <li><span class="tag-red">RSI &gt; 70</span> → <strong>超买（Overbought）</strong>：涨太快了，短期可能回调，不适合追高</li>
        <li><span class="tag-green">RSI &lt; 30</span> → <strong>超卖（Oversold）</strong>：跌太猛了，短期可能反弹，可以关注买点</li>
        <li>RSI 40–60 → 中性区间，趋势不明朗</li>
      </ul>
      <p>⚠️ RSI 超买不代表立刻会跌，强势股可以在高 RSI 维持很久。RSI 只是参考信号之一。</p>
    </div>

    <div class="edu-card">
      <h3>K线形态（Candlestick Patterns）是什么？</h3>
      <p>特定形状的K线或几根K线的组合，历史上经常出现在价格转折点附近。图表上的箭头就是标注这些形态的位置。</p>
      <table class="pattern-table">
        <tr><th>形态名称</th><th>含义</th><th>箭头方向</th></tr>
        <tr><td><strong>Hammer（锤子线）</strong></td><td>下影线很长，下跌后出现，可能反弹</td><td class="green">↑ 绿色</td></tr>
        <tr><td><strong>Shooting Star（射击之星）</strong></td><td>上影线很长，上涨后出现，可能回调</td><td class="red">↓ 红色</td></tr>
        <tr><td><strong>Doji（十字星）</strong></td><td>开收价几乎一样，市场犹豫不决</td><td style="color:#a87619">● 金色</td></tr>
        <tr><td><strong>Bullish Engulfing（看涨吞没）</strong></td><td>大阳线完全盖住前一天阴线，多头强力反转</td><td class="green">↑ 绿色</td></tr>
        <tr><td><strong>Bearish Engulfing（看跌吞没）</strong></td><td>大阴线完全盖住前一天阳线，空头强力反转</td><td class="red">↓ 红色</td></tr>
        <tr><td><strong>Morning Star（早晨之星）</strong></td><td>三根K线，底部反转，看涨</td><td class="green">↑ 绿色</td></tr>
        <tr><td><strong>Evening Star（黄昏之星）</strong></td><td>三根K线，顶部反转，看跌</td><td class="red">↓ 红色</td></tr>
      </table>
      <p>⚠️ 形态只是"线索"，不是预言。需要结合 RSI、布林带位置、成交量综合判断。</p>
    </div>

    <div class="edu-card">
      <h3>信号是怎么算出来的？</h3>
      <p>每个时间框架的信号是把多个指标打分加总得出的：</p>
      <ul class="edu-list">
        <li><strong>1日信号</strong>：RSI 位置 + 布林带位置 + 当日K线形态 + 成交量</li>
        <li><strong>1周信号</strong>：RSI + 价格是否在 SMA20 上方 + MACD 方向</li>
        <li><strong>1月信号</strong>：价格是否在 SMA50 上方 + MACD 形态 + 相对 SPY 表现</li>
        <li><strong>3月信号</strong>：价格是否在 SMA200 上方 + 中期超额收益</li>
        <li><strong>1年信号</strong>：SMA200 长期结构 + 年化超额收益</li>
      </ul>
      <p>分数高 → 买入（BUY）；分数低 → 卖出（SELL）；中间 → 持有观望（HOLD）。</p>
    </div>

    <div class="edu-card">
      <h3>怎么用这个报告做决定？</h3>
      <ul class="edu-list">
        <li>多个时间框架都是绿色 → 趋势比较健康，可以考虑买入或定投</li>
        <li>短期红但中长期绿 → 正常回调，不用慌，定投继续</li>
        <li>短期+中期都是红 → 小心，先观望，等信号转好再加仓</li>
        <li>全线红色 → 趋势破坏，考虑减仓或等待</li>
      </ul>
      <p>⚠️ 这些信号是纯技术分析，不考虑公司基本面和宏观经济。实际决策建议结合多方信息。</p>
    </div>

  </div>
</details>"""


# ── Per-ticker chart section ──────────────────────────────────────────────────

def ticker_section(ticker, sigs):
    full_df = load_price(ticker, extra=60)
    if full_df is None or full_df.empty:
        return f'<div class="panel"><h2>{ticker}</h2><p>数据缺失，请先运行 fetch_prices.py</p></div>'

    chart_df = full_df.tail(CHART_DAYS)
    close    = full_df["Close"]

    # Lines (computed on full_df for warm SMA, then sliced)
    s20  = sma(close, 20).tail(CHART_DAYS)
    s50  = sma(close, 50).tail(CHART_DAYS)
    s200 = sma(close, 200).tail(CHART_DAYS)

    # BB on chart window
    bb_up_full, bb_mid_full, bb_lo_full = bollinger(close)
    bb_up  = bb_up_full.tail(CHART_DAYS)
    bb_mid = bb_mid_full.tail(CHART_DAYS)
    bb_lo  = bb_lo_full.tail(CHART_DAYS)

    r      = sigs.get(ticker, {})
    sig_1d = r.get("signal_1d", "HOLD")

    markers = get_markers(chart_df, sig_1d)

    candles_j = json.dumps(df_to_candles(chart_df))
    s20_j     = json.dumps(df_to_line(chart_df.index, s20.values))
    s50_j     = json.dumps(df_to_line(chart_df.index, s50.values))
    s200_j    = json.dumps(df_to_line(chart_df.index, s200.values))
    bb_up_j   = json.dumps(df_to_line(chart_df.index, bb_up.values))
    bb_mid_j  = json.dumps(df_to_line(chart_df.index, bb_mid.values))
    bb_lo_j   = json.dumps(df_to_line(chart_df.index, bb_lo.values))
    markers_j = json.dumps(markers)

    price   = na(r.get("price"), ".2f")
    rsi_val = na(r.get("rsi14"), ".1f")
    bb_pct  = na(r.get("bb_pct"), ".0f")
    atr_val = na(r.get("atr14"), ".2f")
    volr    = na(r.get("vol_ratio"), ".2f")
    pat_zh  = r.get("patterns_zh") or "无特殊K线形态"
    why_1d  = r.get("why_1d", "—")
    why_1w  = r.get("why_1w", "—")
    why_1m  = r.get("why_1m", "—")
    why_3m  = r.get("why_3m", "—")
    why_1y  = r.get("why_1y", "—")

    chart_id = f"chart-{ticker}"

    # RSI interpretation
    try:
        rsi_num = float(r.get("rsi14", 50))
        if rsi_num > 70:   rsi_note = f'<span class="red">超买区 Overbought（{rsi_num:.0f}），追高需谨慎</span>'
        elif rsi_num < 30: rsi_note = f'<span class="green">超卖区 Oversold（{rsi_num:.0f}），关注反弹机会</span>'
        elif rsi_num < 45: rsi_note = f'<span class="green">偏低（{rsi_num:.0f}），有反弹空间</span>'
        elif rsi_num > 60: rsi_note = f'<span class="red">偏高（{rsi_num:.0f}），上行阻力增加</span>'
        else:              rsi_note = f'中性区间（{rsi_num:.0f}）'
    except Exception:
        rsi_note = rsi_val

    # BB position interpretation
    try:
        bb_num = float(r.get("bb_pct", 50))
        if bb_num < 20:   bb_note = f'<span class="green">靠近下轨（{bb_num:.0f}%），低估区，可能反弹</span>'
        elif bb_num > 80: bb_note = f'<span class="red">靠近上轨（{bb_num:.0f}%），高估区，谨慎追高</span>'
        else:             bb_note = f'布林带中间区域（{bb_num:.0f}%），无明显偏离'
    except Exception:
        bb_note = bb_pct

    return f"""
<div class="ticker-block panel">
  <div class="ticker-header">
    <div>
      <h2>{ticker} &nbsp;<span class="price">${price}</span></h2>
      <div class="indicator-row">
        <div class="ind-item">
          <div class="ind-label">RSI（买卖力度）</div>
          <div>{rsi_note}</div>
        </div>
        <div class="ind-item">
          <div class="ind-label">Bollinger Band 位置</div>
          <div>{bb_note}</div>
        </div>
        <div class="ind-item">
          <div class="ind-label">ATR（每日平均波动）</div>
          <div>${atr_val}</div>
        </div>
        <div class="ind-item">
          <div class="ind-label">成交量 vs 20日均量</div>
          <div>{volr}×</div>
        </div>
      </div>
    </div>
    <div class="sig-row">
      <div class="sig-cell"><div class="sig-label">今日 1D</div>{badge(r.get("signal_1d","HOLD"))}</div>
      <div class="sig-cell"><div class="sig-label">本周 1W</div>{badge(r.get("signal_1w","HOLD"))}</div>
      <div class="sig-cell"><div class="sig-label">本月 1M</div>{badge(r.get("signal_1m","HOLD"))}</div>
      <div class="sig-cell"><div class="sig-label">三月 3M</div>{badge(r.get("signal_3m","HOLD"))}</div>
      <div class="sig-cell"><div class="sig-label">全年 1Y</div>{badge(r.get("signal_1y","HOLD"))}</div>
    </div>
  </div>

  <div class="chart-wrap">
    <div id="{chart_id}" class="chart-container"></div>
    <div class="chart-legend">
      <span class="dot" style="background:#2f5e99"></span>SMA20 &nbsp;
      <span class="dot" style="background:#a87619"></span>SMA50 &nbsp;
      <span class="dot" style="background:#646a73"></span>SMA200 &nbsp;
      <span class="dot" style="background:#aaa;border:1px dashed #555"></span>Bollinger Bands &nbsp;
      <span class="dot" style="background:#1a8f4a"></span>看涨信号 ↑ &nbsp;
      <span class="dot" style="background:#c84b2b"></span>看跌信号 ↓
    </div>
  </div>

  <div class="annotation-box">
    <div class="ann-label">📌 今日K线形态</div>
    <p>{pat_zh}</p>
  </div>

  <div class="why-section">
    <div class="why-label">信号理由详情</div>
    <div class="why-grid">
      <div><span class="tf-tag">今日 1D</span><span class="why-text">{why_1d}</span></div>
      <div><span class="tf-tag">本周 1W</span><span class="why-text">{why_1w}</span></div>
      <div><span class="tf-tag">本月 1M</span><span class="why-text">{why_1m}</span></div>
      <div><span class="tf-tag">三月 3M</span><span class="why-text">{why_3m}</span></div>
      <div><span class="tf-tag">全年 1Y</span><span class="why-text">{why_1y}</span></div>
    </div>
  </div>
</div>

<script>
(function() {{
  var el = document.getElementById('{chart_id}');
  var chart = LightweightCharts.createChart(el, {{
    width: el.offsetWidth || 960,
    height: 320,
    layout: {{ background: {{ color: '#fff' }}, textColor: '#151515' }},
    grid: {{ vertLines: {{ color: '#f0f3f5' }}, horzLines: {{ color: '#f0f3f5' }} }},
    rightPriceScale: {{ borderColor: '#d9dde3' }},
    timeScale: {{ borderColor: '#d9dde3', timeVisible: true }},
    crosshair: {{ mode: 1 }},
  }});

  // Candlestick series
  var cs = chart.addCandlestickSeries({{
    upColor: '#1a8f4a', downColor: '#c84b2b',
    borderUpColor: '#1a8f4a', borderDownColor: '#c84b2b',
    wickUpColor: '#1a8f4a', wickDownColor: '#c84b2b',
  }});
  cs.setData({candles_j});
  cs.setMarkers({markers_j});

  // Bollinger Bands (dashed)
  var bbOpts = {{ color: 'rgba(120,120,120,0.45)', lineWidth: 1, lineStyle: 1, priceLineVisible: false, lastValueVisible: false }};
  chart.addLineSeries(bbOpts).setData({bb_up_j});
  chart.addLineSeries(bbOpts).setData({bb_lo_j});
  var bbMidOpts = {{ color: 'rgba(120,120,120,0.25)', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false }};
  chart.addLineSeries(bbMidOpts).setData({bb_mid_j});

  // SMA lines
  chart.addLineSeries({{ color: '#2f5e99', lineWidth: 1, priceLineVisible: false, lastValueVisible: false }}).setData({s20_j});
  chart.addLineSeries({{ color: '#a87619', lineWidth: 1, priceLineVisible: false, lastValueVisible: false }}).setData({s50_j});
  chart.addLineSeries({{ color: '#646a73', lineWidth: 1, priceLineVisible: false, lastValueVisible: false }}).setData({s200_j});

  chart.timeScale().fitContent();
  new ResizeObserver(function() {{ chart.applyOptions({{ width: el.offsetWidth }}); }}).observe(el);
}})();
</script>
"""


# ── Summary table ─────────────────────────────────────────────────────────────

def summary_table(sigs):
    rows = ""
    for t in TICKERS:
        r = sigs.get(t, {})
        if not r:
            continue
        rows += f"""<tr>
          <td><strong>{t}</strong></td>
          <td>${na(r.get('price'), '.2f')}</td>
          <td>{badge(r.get('signal_1d','HOLD'))}</td>
          <td>{badge(r.get('signal_1w','HOLD'))}</td>
          <td>{badge(r.get('signal_1m','HOLD'))}</td>
          <td>{badge(r.get('signal_3m','HOLD'))}</td>
          <td>{badge(r.get('signal_1y','HOLD'))}</td>
          <td>{fmt_ret(r.get('ret_1w'))}</td>
          <td>{fmt_ret(r.get('ret_1m'))}</td>
          <td>{fmt_ret(r.get('ret_3m'))}</td>
          <td>{fmt_ret(r.get('ret_1y'))}</td>
        </tr>"""
    return f"""
<table class="summary-table">
  <thead>
    <tr>
      <th>标的</th><th>价格</th>
      <th>今日</th><th>本周</th><th>本月</th><th>三月</th><th>全年</th>
      <th>1周涨跌</th><th>1月涨跌</th><th>3月涨跌</th><th>1年涨跌</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>"""


# ── Full HTML ─────────────────────────────────────────────────────────────────

def build_html(sigs):
    today    = date.today()
    edu      = educational_section()
    tickers  = "\n".join(ticker_section(t, sigs) for t in TICKERS)
    summary  = summary_table(sigs)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Abby 信号报告 {today}</title>
  <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    :root {{
      --ink: #151515; --muted: #646a73; --line: #d9dde3;
      --green: #1a8f4a; --red: #c84b2b;
      --bg: #f7f8fa; --panel: #ffffff; --soft: #f1f4f6;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; color: var(--ink); background: var(--bg); font-size: 14px; line-height: 1.6; }}
    main {{ width: min(1200px, calc(100vw - 32px)); margin: 28px auto; display: flex; flex-direction: column; gap: 16px; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; }}
    h2 {{ font-size: 18px; margin-bottom: 12px; }}
    h3 {{ font-size: 14px; font-weight: 700; margin-bottom: 8px; color: var(--ink); }}
    p  {{ color: var(--muted); }}
    a  {{ color: #247d76; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 20px; }}
    .green {{ color: var(--green); }} .red {{ color: var(--red); }}

    /* badge */
    .badge {{ display: inline-block; border-radius: 8px; padding: 5px 12px; color: #fff; font-size: 12px; font-weight: 700; text-align: center; line-height: 1.4; min-width: 72px; }}

    /* summary table */
    .summary-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .summary-table th, .summary-table td {{ padding: 9px 8px; border-bottom: 1px solid var(--line); text-align: center; white-space: nowrap; }}
    .summary-table th:first-child, .summary-table td:first-child {{ text-align: left; }}

    /* ticker block */
    .ticker-block {{ }}
    .ticker-header {{ display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; margin-bottom: 16px; }}
    .price {{ font-size: 22px; font-weight: 700; }}
    .indicator-row {{ display: flex; gap: 24px; flex-wrap: wrap; margin-top: 10px; }}
    .ind-item {{ }}
    .ind-label {{ font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 3px; }}
    .sig-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .sig-cell {{ text-align: center; }}
    .sig-label {{ font-size: 11px; color: var(--muted); margin-bottom: 5px; }}

    /* chart */
    .chart-wrap {{ margin-bottom: 12px; }}
    .chart-container {{ width: 100%; height: 320px; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }}
    .chart-legend {{ font-size: 12px; color: var(--muted); margin-top: 6px; display: flex; flex-wrap: wrap; gap: 14px; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 3px; vertical-align: middle; }}

    /* pattern annotation */
    .annotation-box {{ background: var(--soft); border: 1px solid var(--line); border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }}
    .ann-label {{ font-size: 12px; font-weight: 700; margin-bottom: 6px; color: var(--ink); }}
    .annotation-box p {{ font-size: 13px; color: var(--ink); }}

    /* why section */
    .why-section {{ }}
    .why-label {{ font-size: 12px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; margin-bottom: 8px; }}
    .why-grid {{ display: flex; flex-direction: column; gap: 5px; }}
    .why-grid > div {{ display: flex; align-items: baseline; gap: 8px; font-size: 13px; }}
    .tf-tag {{ display: inline-block; min-width: 52px; font-weight: 700; color: var(--ink); font-size: 12px; }}
    .why-text {{ color: var(--muted); }}

    /* edu block */
    .edu-block {{ cursor: default; }}
    .edu-block summary {{ cursor: pointer; padding: 4px 0; font-size: 15px; list-style: none; }}
    .edu-block summary::-webkit-details-marker {{ display: none; }}
    .edu-block summary::before {{ content: "▶ "; font-size: 12px; }}
    details[open] summary::before {{ content: "▼ "; }}
    .edu-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-top: 16px; }}
    .edu-card {{ background: var(--soft); border: 1px solid var(--line); border-radius: 6px; padding: 16px; }}
    .edu-card h3 {{ font-size: 14px; margin-bottom: 10px; }}
    .edu-card p {{ font-size: 13px; color: var(--ink); margin-bottom: 8px; }}
    .edu-list {{ padding-left: 18px; color: var(--ink); font-size: 13px; }}
    .edu-list li {{ margin-bottom: 5px; }}
    .tag-green {{ background: #d4f0de; color: #0a4d25; padding: 1px 7px; border-radius: 4px; font-size: 12px; }}
    .tag-red   {{ background: #fde8e3; color: #7a1000; padding: 1px 7px; border-radius: 4px; font-size: 12px; }}

    /* candle demo illustration */
    .candle-demo {{ display: flex; gap: 24px; align-items: center; margin: 12px 0; }}
    .candle-explain {{ display: flex; flex-direction: column; align-items: center; width: 24px; }}
    .wick-top   {{ width: 2px; height: 18px; background: #1a8f4a; }}
    .candle-body {{ width: 16px; height: 28px; border-radius: 2px; }}
    .wick-bottom {{ width: 2px; height: 18px; background: #1a8f4a; }}
    .green-body {{ background: #1a8f4a; }}
    .candle-labels {{ font-size: 12px; color: var(--ink); line-height: 2; }}

    /* pattern table */
    .pattern-table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
    .pattern-table th {{ background: var(--line); padding: 5px 8px; text-align: left; }}
    .pattern-table td {{ padding: 5px 8px; border-bottom: 1px solid var(--line); }}

    @media (max-width: 768px) {{
      .ticker-header, .indicator-row, .sig-row {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
<main>

  <div class="panel">
    <h1>Abby 投资信号报告</h1>
    <p>生成日期：{today} &nbsp;|&nbsp; 数据来源：yfinance（公开市场价格）&nbsp;|&nbsp; 图表：最近 {CHART_DAYS} 个交易日 K 线</p>
    <p style="margin-top:6px;font-size:12px;color:#c84b2b">⚠️ 纯技术面信号，不构成投资建议。历史形态不代表未来走势，请结合基本面和自身风险承受能力决策。</p>
  </div>

  {edu}

  <div class="panel">
    <h2>信号总览</h2>
    {summary}
  </div>

  {tickers}

</main>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    sigs = load_signals()
    if not sigs:
        print("今日信号文件未找到，请先运行: python3 scripts/signals.py")
        return

    html = build_html(sigs)
    out  = REPORTS_DIR / f"signal_report_{date.today()}.html"
    out.write_text(html, encoding="utf-8")
    print(f"报告已生成 → {out}")
    import subprocess
    subprocess.run(["open", str(out)])


if __name__ == "__main__":
    main()
