# Trading Analysis

Reproducible Python research tools for a moving-average trading strategy, rolling out-of-sample evaluation, and European option pricing. The notebook uses NSE-listed Reliance Industries (`RELIANCE.NS`) as an example. Results are historical simulations, not live trading results or forecasts.

## Setup

Use Python 3.10 or later. From the repository root, install the package and, if you want the notebook, its optional notebook dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[notebook]"
jupyter lab
```

The notebook downloads adjusted daily closing prices with `yfinance`. The analysis also works with a CSV containing `Date` and adjusted `Close` columns:

```bash
python scripts/analyze_stock.py --ticker RELIANCE.NS --start 2023-01-01 --cost-bps 5 --allocation 0.5 --report-dir results/reliance_backtest
python scripts/analyze_stock.py --csv prices.csv --fast 20 --slow 50 --cost-bps 5
```

The strategy goes long when the 20-day moving average exceeds the 50-day average and otherwise holds cash. It observes a signal at a daily close, fills at the next close, and first earns the following close-to-close return. Trading costs are charged on the fill date, after that day's return from any previously held position. This avoids trading at a price already used to form the signal. The basis-point cost on each entry and exit simplifies commissions, bid-ask spread, and slippage into one input. An open position is not liquidated at the final close of a single backtest. The model does not simulate intraday fills, market impact, taxes, borrowing, or interest on cash. Two buy-and-hold benchmarks purchase at the first reported close: one invests 100% and the other invests the same fixed allocation cap as the strategy. The allocation-cap benchmark remains invested when the strategy switches to cash, so it does not match the strategy's realized average exposure. Adjusted prices account for splits and dividends in the return series.

The summary includes total and annualized return, annualized volatility, Sharpe ratio (zero risk-free rate), maximum drawdown, trade events, turnover, time in market, and average/maximum capital exposure. `summed_cost_rates` adds the per-trade cost rates; it is not a cash amount or a percentage of initial capital. `--allocation` caps the share of capital invested on long signals. The optional report directory contains `daily.csv`, `summary.json`, `run.json` with inputs, dates and a SHA-256 checksum, `prices.csv` with the exact adjusted closes used, and `equity_drawdown.png`. The saved prices make the run reproducible even if the upstream data changes; they do not establish what data was available at the historical decision date. A single backtest is **in-sample descriptive analysis**, not an estimate of future performance.

## Rolling out-of-sample evaluation

```bash
python scripts/walk_forward.py --ticker RELIANCE.NS --start 2020-01-01 --train-size 504 --test-size 126 --fast-grid 10,20,30 --slow-grid 50,100 --cost-bps 5 --allocation 0.5 --report-dir results/reliance_walk_forward
```

For each fold, the evaluator selects the fast/slow pair with the highest **net training Sharpe** from the previous 504 sessions. It then applies that pair to the next 126 unseen sessions. Selection uses no prices from the test block. Each test fold starts and ends flat, with both boundary trades charged. Both benchmarks purchase at each fold's first test close, so no series receives the return before it could enter. Non-overlapping test blocks are concatenated to produce the out-of-sample equity curve and metrics. The output includes `folds.csv` showing training/test dates, selected parameters, and test results. The final fold can be shorter when data runs out.

This is a stronger check than reporting the best in-sample backtest, but repeated parameter searches, a single asset, data quality, survivorship, and changing market regimes still limit generalization. Do not describe simulated returns as actual trading returns.

## Multi-stock cost sensitivity

Use fixed 20/50-day windows on several stocks while varying only the one-way trading cost. This isolates the effect of the cost assumption from parameter selection:

```bash
python scripts/cost_sensitivity.py --tickers RELIANCE.NS,TCS.NS,INFY.NS --start 2020-01-01 --cost-grid 0,5,10,20 --allocation 0.5 --report-dir results/multi_stock
python scripts/cost_sensitivity.py --csv-dir my_adjusted_prices --cost-grid 0,5,10,20 --report-dir results/my_data
```

The CSV directory should contain one `Date,Close` file per stock. Each asset is evaluated independently, not as a portfolio. The report contains `sensitivity.csv`, a chart, exact price snapshots, and a `run.json` manifest with checksums. Different stocks can have different available dates and historical survivorship effects; compare their test periods before drawing cross-stock conclusions.

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

Tests cover signal/fill/cost timing, both benchmarks, snapshot round trips, cost sensitivity, rolling test separation, a known Black-Scholes price, put-call parity, numerical Greeks, and implied-volatility inversion. GitHub Actions runs the test suite on Python 3.11 and 3.13.
