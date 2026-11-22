"""Research tools for the Trading-Analysis project."""

from .backtest import backtest_moving_average, performance_metrics
from .options import black_scholes, implied_volatility

__all__ = [
    "backtest_moving_average",
    "performance_metrics",
    "black_scholes",
    "implied_volatility",
]
