import math
import unittest

import pandas as pd

from trading_analysis.backtest import backtest_moving_average, performance_metrics


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.close = pd.Series(
            [100, 101, 102, 103, 104, 103, 102, 101, 100, 99, 100, 101],
            index=pd.date_range("2025-01-01", periods=12, freq="B"),
        )

    def test_signal_waits_for_next_close_before_earning_return(self):
        result = backtest_moving_average(self.close, fast=2, slow=3, cost_bps=0)
        # Signal after close at index 2, execution at close index 3,
        # first position return is the index 3 -> 4 close-to-close move.
        self.assertEqual(result["position"].iloc[0], 1.0)
        self.assertAlmostEqual(result["gross_return"].iloc[0], 104 / 103 - 1)
        self.assertEqual(result["signal"].iloc[0], 1.0)

    def test_costs_are_charged_on_entries_and_exits(self):
        free = backtest_moving_average(self.close, fast=2, slow=3, cost_bps=0)
        costly = backtest_moving_average(self.close, fast=2, slow=3, cost_bps=10)
        self.assertGreater(costly["turnover"].sum(), 1)
        self.assertAlmostEqual(costly["cost"].sum(), costly["turnover"].sum() * 0.001)
        self.assertLess(costly["strategy_equity"].iloc[-1], free["strategy_equity"].iloc[-1])
        self.assertAlmostEqual(costly["benchmark_equity"].iloc[-1], free["benchmark_equity"].iloc[-1])

    def test_metrics_use_same_window_and_peak_includes_initial_capital(self):
        result = backtest_moving_average(self.close, fast=2, slow=3)
        metrics = performance_metrics(result)
        self.assertEqual(metrics["observations"], len(result))
        self.assertAlmostEqual(metrics["strategy_total_return"], result["strategy_equity"].iloc[-1] - 1)
        self.assertAlmostEqual(metrics["benchmark_total_return"], result["benchmark_equity"].iloc[-1] - 1)
        self.assertLessEqual(metrics["strategy_max_drawdown"], 0)
        self.assertTrue(math.isfinite(metrics["strategy_annualized_volatility"]))

    def test_allocation_limits_exposure_and_scales_gross_return(self):
        full = backtest_moving_average(self.close, fast=2, slow=3, allocation=1.0, cost_bps=0)
        half = backtest_moving_average(self.close, fast=2, slow=3, allocation=0.5, cost_bps=0)
        self.assertTrue((half["position"] <= 0.5).all())
        pd.testing.assert_series_equal(half["gross_return"], full["gross_return"] * 0.5)
        self.assertAlmostEqual(performance_metrics(half)["maximum_exposure"], 0.5)

    def test_rejects_bad_prices(self):
        with self.assertRaises(ValueError):
            backtest_moving_average(self.close.iloc[::-1], fast=2, slow=3)
        with self.assertRaises(ValueError):
            backtest_moving_average(self.close.mask(self.close.index == self.close.index[4]), fast=2, slow=3)


if __name__ == "__main__":
    unittest.main()
