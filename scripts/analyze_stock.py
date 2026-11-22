"""Run the cost-aware moving-average backtest on a CSV or Yahoo Finance data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trading_analysis import backtest_moving_average, performance_metrics


def load_close(args: argparse.Namespace) -> pd.Series:
    if args.csv:
        data = pd.read_csv(args.csv, parse_dates=["Date"])
        if "Close" not in data:
            raise ValueError("CSV must have Date and Close columns; use adjusted Close prices")
        return data.set_index("Date")["Close"]
    import yfinance as yf

    data = yf.download(args.ticker, start=args.start, end=args.end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"no price data returned for {args.ticker}")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError("expected one ticker")
        close = close.iloc[:, 0]
    return close


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV with Date and adjusted Close columns")
    source.add_argument("--ticker", help="Yahoo Finance ticker, such as RELIANCE.NS")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None, help="exclusive end date for ticker download")
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--allocation", type=float, default=1.0, help="fraction of capital invested on a long signal")
    parser.add_argument("--output", type=Path, help="optional CSV path for daily backtest data")
    args = parser.parse_args()

    results = backtest_moving_average(load_close(args), fast=args.fast, slow=args.slow, cost_bps=args.cost_bps, allocation=args.allocation)
    metrics = performance_metrics(results)
    clean_metrics = {key: value if not isinstance(value, float) or math.isfinite(value) else None for key, value in metrics.items()}
    print(json.dumps(clean_metrics, indent=2, allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index_label="Date")


if __name__ == "__main__":
    main()
