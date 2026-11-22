import math
import unittest

from trading_analysis.options import black_scholes, implied_volatility


class OptionsTests(unittest.TestCase):
    def test_known_prices_and_put_call_parity(self):
        call = black_scholes(100, 100, 1, 0.05, 0.2)
        put = black_scholes(100, 100, 1, 0.05, 0.2, option_type="put")
        self.assertAlmostEqual(call.price, 10.450583572185565, places=8)
        self.assertAlmostEqual(put.price, 5.573526022256971, places=8)
        self.assertAlmostEqual(call.price - put.price, 100 - 100 * math.exp(-0.05), places=8)

    def test_greeks_match_finite_differences(self):
        base = black_scholes(110, 100, 0.75, 0.04, 0.25, dividend_yield=0.01)
        h = 1e-3
        up = black_scholes(110 + h, 100, 0.75, 0.04, 0.25, dividend_yield=0.01)
        down = black_scholes(110 - h, 100, 0.75, 0.04, 0.25, dividend_yield=0.01)
        self.assertAlmostEqual(base.delta, (up.price - down.price) / (2 * h), places=6)
        self.assertAlmostEqual(base.gamma, (up.price - 2 * base.price + down.price) / h**2, places=5)
        vol_step = 1e-4
        up_vol = black_scholes(110, 100, 0.75, 0.04, 0.25 + vol_step, dividend_yield=0.01)
        down_vol = black_scholes(110, 100, 0.75, 0.04, 0.25 - vol_step, dividend_yield=0.01)
        self.assertAlmostEqual(base.vega, (up_vol.price - down_vol.price) / (2 * vol_step), places=4)

    def test_implied_volatility_round_trip_and_bounds(self):
        price = black_scholes(100, 105, 0.5, 0.03, 0.32, option_type="put").price
        self.assertAlmostEqual(implied_volatility(price, 100, 105, 0.5, 0.03, option_type="put"), 0.32, places=6)
        with self.assertRaises(ValueError):
            implied_volatility(120, 100, 100, 1, 0.03)


if __name__ == "__main__":
    unittest.main()
