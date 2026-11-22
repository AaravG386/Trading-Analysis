"""Compare a fixed crossover rule across stocks and one-way cost assumptions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trading_analysis import cost_sensitivity
from trading_analysis.data import download_close, read_close_csv
from trading_analysis.report import write_price_snapshot


def _cost_grid(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated basis-point values") from error
    if not values:
        raise argparse.ArgumentTypeError("provide at least one cost")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tickers", help="comma-separated Yahoo Finance tickers")
    source.add_argument("--csv-dir", type=Path, help="directory of Date,Close CSV files; stems become asset names")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--train-size", type=int, default=504)
    parser.add_argument("--test-size", type=int, default=126)
    parser.add_argument("--allocation", type=float, default=0.5)
    parser.add_argument("--cost-grid", type=_cost_grid, default=(0.0, 5.0, 10.0, 20.0))
    parser.add_argument("--report-dir", type=Path, default=Path("results/cost_sensitivity"))
    args = parser.parse_args()

    if args.tickers:
        labels = [ticker.strip() for ticker in args.tickers.split(",")]
        if any(not label for label in labels) or len(labels) != len(set(labels)):
            parser.error("tickers must be unique, non-empty names")
        prices = {ticker: download_close(ticker, start=args.start, end=args.end) for ticker in labels}
        source_type = "ticker"
    else:
        files = sorted(args.csv_dir.glob("*.csv"))
        if not files:
            parser.error("CSV directory contains no .csv files")
        prices = {path.stem: read_close_csv(path) for path in files}
        source_type = "csv_directory"

    results = cost_sensitivity(
        prices, costs_bps=args.cost_grid, fast=args.fast, slow=args.slow,
        train_size=args.train_size, test_size=args.test_size,
        allocation=args.allocation,
    )
    target = args.report_dir
    target.mkdir(parents=True, exist_ok=True)
    results.to_csv(target / "sensitivity.csv", index=False)
    snapshots = []
    for number, (asset, close) in enumerate(prices.items(), start=1):
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", asset)
        relative_path = f"prices/{number:02d}_{safe_name}.csv"
        digest = write_price_snapshot(close, target / relative_path)
        snapshots.append({
            "asset": asset,
            "path": relative_path,
            "sha256": digest,
            "rows": len(close),
            "first_price_date": str(close.index[0].date()),
            "last_price_date": str(close.index[-1].date()),
        })
    manifest = {
        "report_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_type": source_type,
        "fast": args.fast,
        "slow": args.slow,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "allocation": args.allocation,
        "cost_grid_bps": list(args.cost_grid),
        "snapshots": snapshots,
        "method": "fixed windows; independent out-of-sample folds per asset and cost",
    }
    (target / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = plt.get_cmap("tab10").colors
    for number, (asset, group) in enumerate(results.groupby("asset", sort=False)):
        color = palette[number % len(palette)]
        ax.plot(group["cost_bps"], 100 * group["strategy_total_return"], marker="o", color=color, label=f"{asset} strategy")
        ax.axhline(100 * group["allocation_benchmark_total_return"].iloc[0], linestyle="--", color=color, alpha=0.7, label=f"{asset} allocation benchmark")
    ax.set(xlabel="One-way trading cost (basis points)", ylabel="Out-of-sample total return (%)", title="Fixed-rule cost sensitivity")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(target / "sensitivity.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"results": json.loads(results.to_json(orient="records", date_format="iso")), "report_directory": str(target)}, indent=2))


if __name__ == "__main__":
    main()
