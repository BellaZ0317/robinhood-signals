"""
fetch_prices.py
Pull 1 year of daily OHLCV data for the watchlist and save to data/prices/.
Run: python3 scripts/fetch_prices.py
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

# ── Watchlist ────────────────────────────────────────────────────────────────
# current holdings + target additions + benchmark
TICKERS = [
    "VOO",   # current core, broad S&P 500 — dip-buy only now, recurring canceled (overlaps QQQ)
    "QQQ",   # current + recurring buy, Nasdaq-100
    "QQQM",  # cheaper version of QQQ (same index, lower fee)
    "NVDA",  # current single-stock position
    "SMH",   # target addition, semiconductor ETF
    "SPY",   # benchmark for comparison
    "XMAX",  # current small position
    "SPCX",  # current 1-share position, very high volatility
]

PRICES_DIR = Path(__file__).parent.parent / "data" / "prices"
PRICES_DIR.mkdir(parents=True, exist_ok=True)

END   = date.today()
START = END - timedelta(days=365)


def fetch_and_save(ticker: str) -> pd.DataFrame | None:
    print(f"  {ticker} ... ", end="", flush=True)
    try:
        df = yf.download(ticker, start=str(START), end=str(END),
                         auto_adjust=True, progress=False)
        if df.empty:
            print("no data")
            return None
        # flatten multi-level columns yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "date"
        out = PRICES_DIR / f"{ticker}.csv"
        df.to_csv(out)
        print(f"ok  ({len(df)} rows → {out.name})")
        return df
    except Exception as e:
        print(f"error: {e}")
        return None


def main():
    print(f"\nFetching {len(TICKERS)} tickers  {START} → {END}\n")
    results = {}
    for t in TICKERS:
        df = fetch_and_save(t)
        if df is not None:
            results[t] = df

    print(f"\n✓ Saved {len(results)}/{len(TICKERS)} tickers to {PRICES_DIR}\n")

    # quick sanity: latest close prices
    print("Latest close prices:")
    for t, df in results.items():
        latest = df["Close"].iloc[-1]
        print(f"  {t:6s}  ${latest:>9.2f}")


if __name__ == "__main__":
    main()
