"""Compare one fixed trading rule across assets and assumed trading costs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Iterable

import pandas as pd

from .validation import walk_forward_moving_average


def cost_sensitivity(
    prices_by_asset: Mapping[str, pd.Series],
    *,
    costs_bps: Iterable[float] = (0.0, 5.0, 10.0, 20.0),
    fast: int = 20,
    slow: int = 50,
    train_size: int = 504,
    test_size: int = 126,
    allocation: float = 0.5,
) -> pd.DataFrame:
    """Run fixed windows on out-of-sample folds for each asset/cost pair.

    A single fast/slow pair is used throughout, so cost changes do not also
    change parameter selection. Assets are evaluated independently; this is
    not a portfolio backtest. All results retain the walk-forward evaluator's
    flat fold boundaries and identical allocation cap.
    """
    if not prices_by_asset:
        raise ValueError("provide at least one named price series")
    cost_grid = tuple(float(cost) for cost in costs_bps)
    if not cost_grid or any(not math.isfinite(cost) or cost < 0 for cost in cost_grid):
        raise ValueError("costs_bps must contain finite non-negative values")
    if len(set(cost_grid)) != len(cost_grid):
        raise ValueError("costs_bps must be unique")

    rows: list[dict[str, object]] = []
    for asset, prices in prices_by_asset.items():
        if not asset or not isinstance(asset, str):
            raise ValueError("asset names must be non-empty strings")
        for cost_bps in sorted(cost_grid):
            try:
                run = walk_forward_moving_average(
                    prices, fast_windows=(fast,), slow_windows=(slow,),
                    train_size=train_size, test_size=test_size,
                    cost_bps=cost_bps, allocation=allocation,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"{asset}: {error}") from error
            metrics = run.metrics
            rows.append({
                "asset": asset,
                "cost_bps": cost_bps,
                "fast": fast,
                "slow": slow,
                "allocation": allocation,
                "test_start": run.daily.index[0],
                "test_end": run.daily.index[-1],
                "observations": metrics["observations"],
                "trade_events": metrics["trade_events"],
                "strategy_total_return": metrics["strategy_total_return"],
                "allocation_benchmark_total_return": metrics["allocation_benchmark_total_return"],
                "fully_invested_benchmark_total_return": metrics["benchmark_total_return"],
                "return_gap_vs_allocation_benchmark": metrics["return_gap_vs_allocation_benchmark"],
                "strategy_sharpe": metrics["strategy_sharpe"],
                "strategy_max_drawdown": metrics["strategy_max_drawdown"],
            })
    return pd.DataFrame(rows)
