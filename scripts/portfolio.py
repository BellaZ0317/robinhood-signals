"""
portfolio.py
Compute current positions, average cost basis, and unrealized P&L from
data/transactions.csv (local-only, never committed to git).
Run: python3 scripts/portfolio.py
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
TRANSACTIONS_PATH = DATA_DIR / "transactions.csv"


def load_transactions():
    if not TRANSACTIONS_PATH.exists():
        return None
    df = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"])
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

    for t, pos in positions.items():
        pos["avg_cost"] = pos["cost_basis"] / pos["shares"] if pos["shares"] > 1e-9 else 0.0

    return positions


def fetch_current_prices(tickers):
    prices = {}
    for t in tickers:
        try:
            df = yf.download(t, period="5d", auto_adjust=True, progress=False)
            if not df.empty:
                close = df["Close"]
                prices[t] = float(close.iloc[-1].item() if hasattr(close.iloc[-1], "item") else close.iloc[-1])
        except Exception:
            pass
    return prices


def main():
    txn_df = load_transactions()
    if txn_df is None:
        print("No data/transactions.csv found — nothing to compute.")
        return

    positions = compute_positions(txn_df)
    positions = {t: p for t, p in positions.items() if p["shares"] > 1e-6}
    prices = fetch_current_prices(list(positions.keys()))

    print(f"\n{'─'*70}")
    print("  持仓成本与盈亏")
    print(f"{'─'*70}\n")
    print(f"{'标的':<6}  {'持股':>10}  {'成本价':>9}  {'现价':>9}  {'市值':>10}  {'盈亏':>10}  {'盈亏%':>8}")

    total_cost = 0.0
    total_value = 0.0
    for t, pos in sorted(positions.items()):
        price = prices.get(t)
        if price is None:
            print(f"{t:<6}  {pos['shares']:>10.4f}  ${pos['avg_cost']:>8.2f}  {'n/a':>9}")
            continue
        value = pos["shares"] * price
        pnl = value - pos["cost_basis"]
        pnl_pct = pnl / pos["cost_basis"] * 100 if pos["cost_basis"] > 0 else 0
        total_cost += pos["cost_basis"]
        total_value += value
        print(
            f"{t:<6}  {pos['shares']:>10.4f}  ${pos['avg_cost']:>8.2f}  ${price:>8.2f}  "
            f"${value:>9.2f}  {'+' if pnl>=0 else ''}{pnl:>9.2f}  {'+' if pnl_pct>=0 else ''}{pnl_pct:>7.1f}%"
        )

    total_pnl = total_value - total_cost
    total_pnl_pct = total_pnl / total_cost * 100 if total_cost > 0 else 0
    print(f"{'─'*70}")
    print(f"{'合计':<6}  {'':>10}  {'':>9}  {'':>9}  ${total_value:>9.2f}  "
          f"{'+' if total_pnl>=0 else ''}{total_pnl:>9.2f}  {'+' if total_pnl_pct>=0 else ''}{total_pnl_pct:>7.1f}%\n")


if __name__ == "__main__":
    main()
