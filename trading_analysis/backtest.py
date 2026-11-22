"""Long-or-cash moving-average backtest using adjusted daily closing prices.

Signals use a day's closing price. A signal is filled at the next close;
the resulting position first earns the following close-to-close return.
Trading costs are charged on the fill date, after that day's market return.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _validate_close(close: pd.Series, slow: int) -> pd.Series:
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must have a DatetimeIndex")
    if not close.index.is_monotonic_increasing or not close.index.is_unique:
        raise ValueError("dates must be sorted and unique")
    values = pd.to_numeric(close, errors="raise").astype(float)
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("closing prices must be finite and non-missing")
    if (values <= 0).any():
        raise ValueError("closing prices must be positive")
    if len(values) < slow + 2:
        raise ValueError("need at least slow + 2 daily closing prices")
    return values.rename("close")


def backtest_moving_average(
    close: pd.Series,
    *,
    fast: int = 20,
    slow: int = 50,
    cost_bps: float = 5.0,
    allocation: float = 1.0,
) -> pd.DataFrame:
    """Backtest a long-or-cash crossover with cost per unit of turnover.

    ``cost_bps`` is charged on every entry and exit (5 means 0.05% of
    portfolio value per one-way trade). It represents combined commission,
    spread, and slippage; there is no leverage, shorting, dividend cash flow,
    or interest on uninvested cash. ``allocation`` is the fraction of capital
    invested while the signal is long; remaining capital sits in cash. Use
    adjusted prices for split/dividend treatment. The benchmark is buy-and-hold
    over identical reported dates.
    """
    if isinstance(fast, bool) or isinstance(slow, bool) or not isinstance(fast, int) or not isinstance(slow, int):
        raise TypeError("fast and slow must be integers")
    if not 1 <= fast < slow:
        raise ValueError("require 1 <= fast < slow")
    if not math.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and non-negative")
    if not math.isfinite(allocation) or not 0 < allocation <= 1:
        raise ValueError("allocation must be in (0, 1]")
    prices = _validate_close(close, slow)

    frame = pd.DataFrame(index=prices.index)
    frame["close"] = prices
    frame["fast_ma"] = prices.rolling(fast, min_periods=fast).mean()
    frame["slow_ma"] = prices.rolling(slow, min_periods=slow).mean()
    target = (frame["fast_ma"] > frame["slow_ma"]).astype(float)
    frame["signal"] = target.where(frame["slow_ma"].notna())
    # The previous close's signal can be filled only at today's close.
    frame["position"] = frame["signal"].shift(1) * allocation
    frame["market_return"] = prices.pct_change()
    frame = frame.loc[frame["position"].notna()].copy()
    if frame.empty:
        raise ValueError("no valid strategy returns after the execution delay")

    # Yesterday's post-trade position earns today's market return. The new
    # position is established at today's close and incurs today's cost.
    frame["held_position"] = frame["position"].shift(1).fillna(0.0)
    frame["turnover"] = (frame["position"] - frame["held_position"]).abs()
    frame["cost"] = frame["turnover"] * cost_bps / 10_000
    frame["gross_return"] = frame["held_position"] * frame["market_return"]
    frame["net_return"] = frame["gross_return"] - frame["cost"]
    # Buy-and-hold purchases at the first reported close, so it cannot earn
    # the return that arrived before that close.
    frame["benchmark_return"] = frame["market_return"]
    frame.iloc[0, frame.columns.get_loc("benchmark_return")] = 0.0
    frame["strategy_equity"] = (1.0 + frame["net_return"]).cumprod()
    frame["benchmark_equity"] = (1.0 + frame["benchmark_return"]).cumprod()
    return frame


def performance_metrics(frame: pd.DataFrame, *, periods_per_year: int = TRADING_DAYS) -> dict[str, float | int]:
    """Calculate comparable strategy and buy-and-hold risk/return statistics.

    Sharpe uses a zero risk-free rate. Annualization assumes daily observations
    and ``periods_per_year`` trading days. A zero-volatility Sharpe is NaN.
    """
    required = {"net_return", "benchmark_return", "strategy_equity", "benchmark_equity", "turnover", "cost", "position", "held_position"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    if frame.empty or periods_per_year <= 0:
        raise ValueError("non-empty frame and positive periods_per_year required")

    result: dict[str, float | int] = {
        "observations": len(frame),
        "trade_events": int((frame["turnover"] > 0).sum()),
        "turnover_units": float(frame["turnover"].sum()),
        "summed_cost_rates": float(frame["cost"].sum()),
        "time_in_market": float((frame["held_position"] > 0).mean()),
        "average_exposure": float(frame["held_position"].mean()),
        "maximum_exposure": float(frame["held_position"].max()),
    }
    for label, returns_col, equity_col in (
        ("strategy", "net_return", "strategy_equity"),
        ("benchmark", "benchmark_return", "benchmark_equity"),
    ):
        returns = frame[returns_col]
        equity = frame[equity_col]
        final = float(equity.iloc[-1])
        vol = float(returns.std(ddof=1) * math.sqrt(periods_per_year)) if len(returns) > 1 else math.nan
        annual_return = final ** (periods_per_year / len(frame)) - 1 if final > 0 else math.nan
        sharpe = float(returns.mean() * periods_per_year / vol) if vol > 0 else math.nan
        drawdown = equity / equity.cummax().clip(lower=1.0) - 1
        result.update({
            f"{label}_total_return": final - 1,
            f"{label}_annualized_return": annual_return,
            f"{label}_annualized_volatility": vol,
            f"{label}_sharpe": sharpe,
            f"{label}_max_drawdown": float(drawdown.min()),
        })
    result["return_gap_vs_benchmark"] = result["strategy_total_return"] - result["benchmark_total_return"]
    return result
