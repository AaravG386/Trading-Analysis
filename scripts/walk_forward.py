"""Run rolling out-of-sample moving-average evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trading_analysis import walk_forward_moving_average
from trading_analysis.data import download_close, read_close_csv
from trading_analysis.report import _json_safe, write_report


def _windows(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated integer windows") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("windows must be positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV with Date and adjusted Close columns")
    source.add_argument("--ticker", help="Yahoo Finance ticker, such as RELIANCE.NS")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--fast-grid", type=_windows, default=(10, 20, 30))
    parser.add_argument("--slow-grid", type=_windows, default=(50, 100))
    parser.add_argument("--train-size", type=int, default=504)
    parser.add_argument("--test-size", type=int, default=126)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--allocation", type=float, default=1.0)
    parser.add_argument("--report-dir", type=Path, default=Path("results/walk_forward"))
    args = parser.parse_args()

    close = read_close_csv(args.csv) if args.csv else download_close(args.ticker, start=args.start, end=args.end)
    result = walk_forward_moving_average(
        close, fast_windows=args.fast_grid, slow_windows=args.slow_grid,
        train_size=args.train_size, test_size=args.test_size,
        cost_bps=args.cost_bps, allocation=args.allocation,
    )
    write_report(result.daily, result.metrics, args.report_dir, folds=result.folds, prices=close, run_config={
        "source": args.ticker if args.ticker else args.csv.name,
        "source_type": "ticker" if args.ticker else "csv",
        "first_price_date": str(close.index[0].date()),
        "last_price_date": str(close.index[-1].date()),
        "fast_grid": list(args.fast_grid),
        "slow_grid": list(args.slow_grid),
        "train_size": args.train_size,
        "test_size": args.test_size,
        "cost_bps_per_one_way_trade": args.cost_bps,
        "allocation": args.allocation,
        "selection": "highest net training Sharpe; rolling training window",
    })
    print(json.dumps({
        "out_of_sample_metrics": _json_safe(result.metrics),
        "folds": json.loads(result.folds.to_json(orient="records", date_format="iso")),
        "report_directory": str(args.report_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
