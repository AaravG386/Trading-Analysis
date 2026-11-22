"""Price a European option and show its Black-Scholes Greeks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trading_analysis import black_scholes, implied_volatility


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spot", type=float, required=True)
    parser.add_argument("--strike", type=float, required=True)
    parser.add_argument("--years", type=float, required=True, help="time to expiry in years")
    parser.add_argument("--rate", type=float, required=True, help="annual decimal risk-free rate")
    parser.add_argument("--dividend-yield", type=float, default=0.0, help="annual decimal continuous yield")
    parser.add_argument("--type", choices=["call", "put"], default="call")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--volatility", type=float, help="annual decimal volatility")
    source.add_argument("--market-price", type=float, help="infer implied volatility from this option price")
    args = parser.parse_args()

    volatility = args.volatility
    if volatility is None:
        volatility = implied_volatility(
            args.market_price, args.spot, args.strike, args.years, args.rate,
            dividend_yield=args.dividend_yield, option_type=args.type,
        )
    result = black_scholes(
        args.spot, args.strike, args.years, args.rate, volatility,
        dividend_yield=args.dividend_yield, option_type=args.type,
    )
    print(json.dumps({"implied_volatility" if args.market_price is not None else "volatility": volatility, **asdict(result)}, indent=2))


if __name__ == "__main__":
    main()
