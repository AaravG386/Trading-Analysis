"""Write reproducible backtest data and charts."""

from __future__ import annotations

import json
import math
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _json_safe(metrics: dict[str, float | int]) -> dict[str, float | int | None]:
    return {key: value if not isinstance(value, float) or math.isfinite(value) else None for key, value in metrics.items()}


def write_price_snapshot(prices: pd.Series, path: str | Path) -> str:
    """Save exact adjusted closes used in a run and return their SHA-256 hash."""
    snapshot = prices.rename("Close").to_csv(index_label="Date", float_format="%.17g").encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(snapshot)
    return hashlib.sha256(snapshot).hexdigest()


def write_report(
    daily: pd.DataFrame,
    metrics: dict[str, float | int],
    directory: str | Path,
    *,
    folds: pd.DataFrame | None = None,
    run_config: dict[str, object] | None = None,
    prices: pd.Series | None = None,
) -> None:
    """Save daily CSV, summary JSON, equity/drawdown PNG, and optional metadata."""
    import matplotlib.pyplot as plt

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    daily.to_csv(target / "daily.csv", index_label="Date")
    (target / "summary.json").write_text(json.dumps(_json_safe(metrics), indent=2) + "\n", encoding="utf-8")
    config = dict(run_config or {})
    config["report_created_at_utc"] = datetime.now(timezone.utc).isoformat()
    if prices is not None:
        config["price_snapshot"] = "prices.csv"
        config["price_snapshot_sha256"] = write_price_snapshot(prices, target / "prices.csv")
        config["price_rows"] = len(prices)
    (target / "run.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if folds is not None:
        folds.to_csv(target / "folds.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    equity_columns = ["strategy_equity", "allocation_benchmark_equity", "benchmark_equity"]
    colors = ["#1F3A5F", "#387D66", "#A77D2D"]
    daily[equity_columns].rename(columns={
        "strategy_equity": "Strategy net of costs",
        "allocation_benchmark_equity": "Buy and hold at allocation cap",
        "benchmark_equity": "Fully invested buy and hold",
    }).plot(ax=axes[0], color=colors)
    axes[0].set_title("Growth of 1 unit of capital")
    axes[0].set_ylabel("Portfolio value")
    axes[0].grid(alpha=0.2)
    drawdowns = daily[equity_columns].copy()
    drawdowns = drawdowns / drawdowns.cummax().clip(lower=1.0) - 1
    drawdowns.rename(columns={
        "strategy_equity": "Strategy",
        "allocation_benchmark_equity": "Allocation cap",
        "benchmark_equity": "Fully invested",
    }).plot(ax=axes[1], color=colors)
    axes[1].set_title("Drawdown from previous peak")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(target / "equity_drawdown.png", dpi=160)
    plt.close(fig)
