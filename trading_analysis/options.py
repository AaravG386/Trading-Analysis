"""European Black-Scholes-Merton option pricing with continuous dividends."""

from __future__ import annotations

from dataclasses import dataclass
import math


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _validate(spot: float, strike: float, years: float, rate: float, volatility: float, dividend_yield: float, option_type: str) -> None:
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if not all(math.isfinite(x) for x in (spot, strike, years, rate, volatility, dividend_yield)):
        raise ValueError("all model inputs must be finite")
    if spot <= 0 or strike <= 0 or years <= 0 or volatility <= 0:
        raise ValueError("spot, strike, years, and volatility must be positive")


@dataclass(frozen=True)
class OptionResult:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def black_scholes(
    spot: float,
    strike: float,
    years: float,
    rate: float,
    volatility: float,
    *,
    dividend_yield: float = 0.0,
    option_type: str = "call",
) -> OptionResult:
    """Return price and Greeks for a European call or put.

    Rates, dividend yield, and volatility are decimal annual rates. Theta is
    price change per year as time passes; vega and rho are per 1.0 absolute
    change in volatility and rate respectively (divide by 100 for 1 point).
    """
    _validate(spot, strike, years, rate, volatility, dividend_yield, option_type)
    root_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * years) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    discounted_spot = spot * math.exp(-dividend_yield * years)
    discounted_strike = strike * math.exp(-rate * years)
    common_theta = -(discounted_spot * _pdf(d1) * volatility) / (2.0 * root_t)
    gamma = math.exp(-dividend_yield * years) * _pdf(d1) / (spot * volatility * root_t)
    vega = discounted_spot * _pdf(d1) * root_t

    if option_type == "call":
        return OptionResult(
            price=discounted_spot * _cdf(d1) - discounted_strike * _cdf(d2),
            delta=math.exp(-dividend_yield * years) * _cdf(d1),
            gamma=gamma,
            vega=vega,
            theta=common_theta + dividend_yield * discounted_spot * _cdf(d1) - rate * discounted_strike * _cdf(d2),
            rho=years * discounted_strike * _cdf(d2),
        )
    return OptionResult(
        price=discounted_strike * _cdf(-d2) - discounted_spot * _cdf(-d1),
        delta=math.exp(-dividend_yield * years) * (_cdf(d1) - 1.0),
        gamma=gamma,
        vega=vega,
        theta=common_theta - dividend_yield * discounted_spot * _cdf(-d1) + rate * discounted_strike * _cdf(-d2),
        rho=-years * discounted_strike * _cdf(-d2),
    )


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    years: float,
    rate: float,
    *,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    tolerance: float = 1e-8,
) -> float:
    """Infer volatility by bisection for a valid European option price."""
    _validate(spot, strike, years, rate, 0.2, dividend_yield, option_type)
    if not math.isfinite(market_price) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("market_price must be finite and tolerance positive")
    discounted_spot = spot * math.exp(-dividend_yield * years)
    discounted_strike = strike * math.exp(-rate * years)
    lower = max(0.0, discounted_spot - discounted_strike) if option_type == "call" else max(0.0, discounted_strike - discounted_spot)
    upper = discounted_spot if option_type == "call" else discounted_strike
    if not lower < market_price < upper:
        raise ValueError("market price must lie strictly inside no-arbitrage bounds")

    low, high = 1e-8, 1.0
    while black_scholes(spot, strike, years, rate, high, dividend_yield=dividend_yield, option_type=option_type).price < market_price:
        high *= 2
        if high > 64:
            raise ValueError("could not bracket implied volatility")
    for _ in range(200):
        mid = (low + high) / 2
        price = black_scholes(spot, strike, years, rate, mid, dividend_yield=dividend_yield, option_type=option_type).price
        if abs(price - market_price) < tolerance:
            return mid
        if price < market_price:
            low = mid
        else:
            high = mid
    raise ArithmeticError("implied volatility did not converge")
