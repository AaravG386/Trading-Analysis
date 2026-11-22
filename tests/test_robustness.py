import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trading_analysis.data import read_close_csv
from trading_analysis.report import write_price_snapshot
from trading_analysis.robustness import cost_sensitivity


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        self.prices = {
            "UP": pd.Series(100 + np.arange(120, dtype=float), index=dates),
            "DOWN": pd.Series(200 - 0.25 * np.arange(120, dtype=float), index=dates),
        }

    def test_cost_sensitivity_holds_rule_fixed(self):
        result = cost_sensitivity(
            self.prices, costs_bps=(0, 20), fast=3, slow=10,
            train_size=60, test_size=20, allocation=0.5,
        )
        self.assertEqual(len(result), 4)
        for _, group in result.groupby("asset"):
            group = group.sort_values("cost_bps")
            self.assertEqual(group["fast"].tolist(), [3, 3])
            self.assertEqual(group["slow"].tolist(), [10, 10])
            self.assertLessEqual(group["strategy_total_return"].iloc[1], group["strategy_total_return"].iloc[0])
            self.assertAlmostEqual(
                group["allocation_benchmark_total_return"].iloc[0],
                group["allocation_benchmark_total_return"].iloc[1],
            )

    def test_snapshot_hash_and_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            digest = write_price_snapshot(self.prices["UP"], path)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            expected = self.prices["UP"].rename("Close").rename_axis("Date")
            pd.testing.assert_series_equal(read_close_csv(path), expected, check_freq=False)


if __name__ == "__main__":
    unittest.main()
