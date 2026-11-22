"""Run the cost-aware moving-average backtest on a CSV or Yahoo Finance data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trading_analysis import backtest_moving_average, performance_metrics
from trading_analysis.data import download_close, read_close_csv
from trading_analysis.report import write_report


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
    parser.add_argument("--report-dir", type=Path, help="optional directory for CSV, JSON, and equity chart")
    args = parser.parse_args()

    close = read_close_csv(args.csv) if args.csv else download_close(args.ticker, start=args.start, end=args.end)
    results = backtest_moving_average(close, fast=args.fast, slow=args.slow, cost_bps=args.cost_bps, allocation=args.allocation)
    metrics = performance_metrics(results)
    clean_metrics = {key: value if not isinstance(value, float) or math.isfinite(value) else None for key, value in metrics.items()}
    print(json.dumps(clean_metrics, indent=2, allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index_label="Date")
    if args.report_dir:
        write_report(results, metrics, args.report_dir, run_config={
            "source": args.ticker if args.ticker else args.csv.name,
            "source_type": "ticker" if args.ticker else "csv",
            "first_price_date": str(close.index[0].date()),
            "last_price_date": str(close.index[-1].date()),
            "fast": args.fast,
            "slow": args.slow,
            "cost_bps_per_one_way_trade": args.cost_bps,
            "allocation": args.allocation,
            "execution": "signal at close; fill next close; earn subsequent close-to-close return",
        })


if __name__ == "__main__":
    main()
