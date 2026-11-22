# Trading Analysis

Python research tools for a moving-average strategy and European option pricing. The notebook uses NSE-listed Reliance Industries (`RELIANCE.NS`) as an example. Results are historical simulations, not live trading results.

## Setup

From the repository root, install the requirements and open `notebooks/stock_analysis.ipynb`:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

The notebook downloads adjusted daily closing prices with `yfinance`. The analysis also works with a CSV containing `Date` and adjusted `Close` columns:

```bash
python scripts/analyze_stock.py --ticker RELIANCE.NS --start 2023-01-01 --cost-bps 5 --allocation 0.5 --output results/reliance_backtest.csv
python scripts/analyze_stock.py --csv prices.csv --fast 20 --slow 50 --cost-bps 5
```

The strategy goes long when the 20-day moving average exceeds the 50-day average and otherwise holds cash. It observes a signal at a daily close, fills at the next close, and begins earning returns after that fill. This two-session delay avoids using the closing price before it is known. The backtest charges the specified basis-point cost on each entry and exit that occurs, covering a simplified combination of commissions, bid-ask spread, and slippage. An open position is not liquidated at the final close. It does not model intraday fills, market impact, taxes, borrowing, or interest on cash. The buy-and-hold benchmark uses the same reported dates. Adjusted prices account for splits and dividends in the price series.

The summary includes total and annualized return, annualized volatility, Sharpe ratio (zero risk-free rate), maximum drawdown, turnover, time in market, and average/maximum capital exposure. `--allocation` caps the share of capital invested on long signals. The notebook plots the equity curves. These metrics are **in-sample descriptive results**, not estimates of future performance. If you tune the moving-average windows, use a separate out-of-sample period or walk-forward evaluation; avoid survivorship and look-ahead bias.

## Black-Scholes option pricer

```bash
python scripts/price_option.py --spot 100 --strike 100 --years 1 --rate 0.05 --volatility 0.20 --type call
python scripts/price_option.py --spot 100 --strike 100 --years 1 --rate 0.05 --market-price 10.45 --type call
```

The pricer returns a European call or put price and delta, gamma, vega, theta, and rho. It supports a continuous dividend yield and can infer implied volatility from a market price. Inputs are annual decimal rates, so `0.20` means 20% volatility. Theta is per year; vega and rho are for a 1.0 absolute change in volatility and interest rate (divide by 100 for a one-percentage-point change). The model assumes constant volatility and rates and European exercise; it does not reproduce all real-world option prices or model early exercise.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover signal timing, trading costs, position sizing, benchmark alignment, a known Black-Scholes price, put-call parity, numerical Greeks, and implied-volatility inversion.
