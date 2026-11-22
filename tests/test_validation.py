import unittest

import numpy as np
import pandas as pd

from trading_analysis.validation import walk_forward_moving_average


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.close = pd.Series(
            100 + np.arange(120, dtype=float),
            index=pd.date_range("2024-01-01", periods=120, freq="B"),
        )

    def test_folds_use_disjoint_unseen_dates_and_reset_positions(self):
        result = walk_forward_moving_average(
            self.close, fast_windows=(3, 5), slow_windows=(10, 15),
            train_size=60, test_size=20, cost_bps=10,
        )
        self.assertEqual(len(result.folds), 3)
        self.assertEqual(len(result.daily), 60)
        self.assertTrue(result.daily.index.is_unique)
        self.assertEqual(result.daily.index[0], self.close.index[60])
        for fold in range(1, 4):
            block = result.daily[result.daily["fold"] == fold]
            self.assertEqual(block["held_position"].iloc[0], 0)
            self.assertEqual(block["benchmark_return"].iloc[0], 0)
            self.assertEqual(block["position"].iloc[-1], 0)
            self.assertAlmostEqual(block["cost"].sum(), block["turnover"].sum() * 0.001)
            self.assertGreaterEqual(block["turnover"].sum(), 2)
        self.assertAlmostEqual(result.metrics["strategy_total_return"], result.daily["strategy_equity"].iloc[-1] - 1)

    def test_future_prices_cannot_change_first_training_selection(self):
        kwargs = dict(fast_windows=(3, 5), slow_windows=(10, 15), train_size=60, test_size=20)
        original = walk_forward_moving_average(self.close, **kwargs)
        changed = self.close.copy()
        changed.iloc[60:] = changed.iloc[60:] + np.arange(60) ** 2
        revised = walk_forward_moving_average(changed, **kwargs)
        self.assertEqual(tuple(original.folds.loc[0, ["fast", "slow"]]), tuple(revised.folds.loc[0, ["fast", "slow"]]))

    def test_invalid_window_or_history_rejected(self):
        with self.assertRaises(ValueError):
            walk_forward_moving_average(self.close, fast_windows=(20,), slow_windows=(10,), train_size=60, test_size=20)
        with self.assertRaises(ValueError):
            walk_forward_moving_average(self.close, fast_windows=(3,), slow_windows=(15,), train_size=15, test_size=20)


if __name__ == "__main__":
    unittest.main()
