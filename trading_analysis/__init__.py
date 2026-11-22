"""Research tools for the Trading-Analysis project."""

from .backtest import backtest_moving_average, performance_metrics
from .options import black_scholes, implied_volatility
from .validation import WalkForwardResult, walk_forward_moving_average

__all__ = [
    "backtest_moving_average",
    "performance_metrics",
    "black_scholes",
    "implied_volatility",
    "WalkForwardResult",
    "walk_forward_moving_average",
]
