"""Write reproducible backtest data and charts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


def _json_safe(metrics: dict[str, float | int]) -> dict[str, float | int | None]:
    return {key: value if not isinstance(value, float) or math.isfinite(value) else None for key, value in metrics.items()}


def write_report(
    daily: pd.DataFrame,
    metrics: dict[str, float | int],
    directory: str | Path,
    *,
    folds: pd.DataFrame | None = None,
    run_config: dict[str, object] | None = None,
) -> None:
    """Save daily CSV, summary JSON, equity/drawdown PNG, and optional metadata."""
    import matplotlib.pyplot as plt

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    daily.to_csv(target / "daily.csv", index_label="Date")
    (target / "summary.json").write_text(json.dumps(_json_safe(metrics), indent=2) + "\n", encoding="utf-8")
    if run_config is not None:
        (target / "run.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    if folds is not None:
        folds.to_csv(target / "folds.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    daily[["strategy_equity", "benchmark_equity"]].rename(columns={
        "strategy_equity": "Strategy net of costs",
        "benchmark_equity": "Buy and hold",
    }).plot(ax=axes[0], color=["#1F3A5F", "#A77D2D"])
    axes[0].set_title("Growth of 1 unit of capital")
    axes[0].set_ylabel("Portfolio value")
    axes[0].grid(alpha=0.2)
    drawdowns = daily[["strategy_equity", "benchmark_equity"]].copy()
    drawdowns = drawdowns / drawdowns.cummax().clip(lower=1.0) - 1
    drawdowns.rename(columns={
        "strategy_equity": "Strategy",
        "benchmark_equity": "Buy and hold",
    }).plot(ax=axes[1], color=["#1F3A5F", "#A77D2D"])
    axes[1].set_title("Drawdown from previous peak")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(target / "equity_drawdown.png", dpi=160)
    plt.close(fig)
