"""Rolling, out-of-sample evaluation for moving-average strategies."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable

import pandas as pd

from .backtest import _validate_close, backtest_moving_average, performance_metrics


@dataclass(frozen=True)
class WalkForwardResult:
    daily: pd.DataFrame
    folds: pd.DataFrame
    metrics: dict[str, float | int]


def walk_forward_moving_average(
    close: pd.Series,
    *,
    fast_windows: Iterable[int] = (10, 20, 30),
    slow_windows: Iterable[int] = (50, 100),
    train_size: int = 504,
    test_size: int = 126,
    cost_bps: float = 5.0,
    allocation: float = 1.0,
) -> WalkForwardResult:
    """Select windows on past data and evaluate on the next unseen block.

    Each rolling training window is ``train_size`` sessions. The best pair by
    *net* training Sharpe is evaluated on the next ``test_size`` sessions.
    Every test fold starts and ends flat, with entry/exit costs included. The
    benchmark is buy-and-hold over exactly the concatenated test dates.
    """
    fast_windows = tuple(fast_windows)
    slow_windows = tuple(slow_windows)
    if any(isinstance(w, bool) or not isinstance(w, int) for w in fast_windows + slow_windows):
        raise ValueError("all windows must be integers")
    pairs = sorted({(fast, slow) for fast, slow in itertools.product(fast_windows, slow_windows) if 1 <= fast < slow})
    if not pairs:
        raise ValueError("provide integer windows with at least one fast < slow pair")
    if isinstance(train_size, bool) or isinstance(test_size, bool) or not isinstance(train_size, int) or not isinstance(test_size, int):
        raise TypeError("train_size and test_size must be integers")
    if train_size < max(slow for _, slow in pairs) + 2 or test_size < 2:
        raise ValueError("training needs slow + 2 sessions; testing needs at least 2")
    prices = _validate_close(close, train_size)
    if len(prices) < train_size + test_size:
        raise ValueError("need at least one full training and test fold")

    daily_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for fold, train_end in enumerate(range(train_size, len(prices), test_size), start=1):
        test_end = min(train_end + test_size, len(prices))
        if test_end - train_end < 2:
            break
        train_start = train_end - train_size
        train_prices = prices.iloc[train_start:train_end]

        ranked = []
        for fast, slow in pairs:
            training = backtest_moving_average(train_prices, fast=fast, slow=slow, cost_bps=cost_bps, allocation=allocation)
            scores = performance_metrics(training)
            sharpe = scores["strategy_sharpe"]
            ranked.append((sharpe if math.isfinite(sharpe) else -math.inf, scores["strategy_total_return"], -fast, -slow, fast, slow))
        best = max(ranked)
        fast, slow = best[-2:]

        # Training history supplies the moving-average warmup. Only dates in
        # the unseen test block contribute to reported returns and metrics.
        test = backtest_moving_average(
            prices.iloc[train_start:test_end], fast=fast, slow=slow,
            cost_bps=cost_bps, allocation=allocation,
        )
        test = test.loc[prices.index[train_end:test_end]].copy()
        if len(test) != test_end - train_end:
            raise ArithmeticError("test dates were lost during backtest alignment")

        # A parameter change starts a fresh paper portfolio at the first
        # test close. It is liquidated at the last test close. Both trades
        # incur the same one-way cost as trades inside the fold.
        test.iloc[0, test.columns.get_loc("held_position")] = 0.0
        test.iloc[0, test.columns.get_loc("turnover")] = test["position"].iloc[0]
        test.iloc[0, test.columns.get_loc("gross_return")] = 0.0
        test.iloc[0, test.columns.get_loc("benchmark_return")] = 0.0
        final_held_position = float(test["held_position"].iloc[-1])
        test.iloc[-1, test.columns.get_loc("turnover")] = final_held_position
        test.iloc[-1, test.columns.get_loc("position")] = 0.0
        test["cost"] = test["turnover"] * cost_bps / 10_000
        test["net_return"] = test["gross_return"] - test["cost"]
        test["strategy_equity"] = (1 + test["net_return"]).cumprod()
        test["benchmark_equity"] = (1 + test["benchmark_return"]).cumprod()
        test["fold"] = fold
        test["selected_fast"] = fast
        test["selected_slow"] = slow
        daily_parts.append(test)

        test_scores = performance_metrics(test)
        fold_rows.append({
            "fold": fold,
            "train_start": train_prices.index[0],
            "train_end": train_prices.index[-1],
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "fast": fast,
            "slow": slow,
            "train_sharpe": best[0] if math.isfinite(best[0]) else math.nan,
            "test_total_return": test_scores["strategy_total_return"],
            "test_sharpe": test_scores["strategy_sharpe"],
            "test_max_drawdown": test_scores["strategy_max_drawdown"],
        })

    daily = pd.concat(daily_parts)
    daily["strategy_equity"] = (1 + daily["net_return"]).cumprod()
    daily["benchmark_equity"] = (1 + daily["benchmark_return"]).cumprod()
    return WalkForwardResult(daily=daily, folds=pd.DataFrame(fold_rows), metrics=performance_metrics(daily))
