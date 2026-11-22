# Published research example: Indian equities through 20 November 2026

This directory preserves the adjusted daily closing-price snapshots and the exact reports behind the resume's trading-project description. These are historical paper simulations, not actual trades or forecasts. The three tickers are evaluated independently; this is not a portfolio.

## Fixed 20/50 moving-average rule

The rule is long at a 50% capital allocation when the 20-session average exceeds the 50-session average, and otherwise holds cash. A signal observed at a close is filled at the next close. Trading cost is charged on each entry and exit, proportional to turnover. The reported dates are **10 January 2022 to 20 November 2026** for all three stocks. The evaluation uses 504-session history and non-overlapping test blocks of up to 126 sessions. The rule is fixed across assets and costs; it is not selected separately for each stock.

| Asset | Strategy, 0 bp | Strategy, 5 bp | Strategy, 20 bp | 50%-allocated buy-and-hold |
| --- | ---: | ---: | ---: | ---: |
| RELIANCE.NS | -2.9% | -3.8% | -6.7% | +15.7% |
| TCS.NS | -13.4% | -14.2% | -16.8% | -12.7% |
| INFY.NS | -12.2% | -13.1% | -15.5% | -13.4% |

At 5 bp, the rule trailed the allocation-cap benchmark on Reliance and TCS and narrowly exceeded it on Infosys. Increasing costs reduced strategy returns for every asset. This is evidence of cost sensitivity and inconsistent benchmark performance, not a profitable trading edge. See [`fixed_rule/sensitivity.csv`](fixed_rule/sensitivity.csv) for unrounded returns, Sharpe ratios, drawdowns, and the full 0/5/10/20 bp grid.

The benchmark remains 50% invested throughout each test block while the strategy can hold cash. It is matched on the maximum allocation, **not** on realized average exposure. Benchmark entry/exit costs are not charged. Comparisons use the same reported dates. On each block's first day, the benchmark receives zero market return and the strategy receives zero gross return but may pay its entry cost.

## Separately tuned Reliance test

[`tuned_reliance/summary.json`](tuned_reliance/summary.json) reports a separate rolling test that selects the fast/slow pair from 10/20/30 by 50/100 sessions using the preceding 504 sessions' *net* Sharpe, then applies it to the next unseen block. At 5 bp per one-way trade and 50% maximum allocation, its cumulative return was **-10.3%**, versus **+15.7%** for 50%-allocated buy-and-hold. Its maximum drawdown was **-16.3%**. These results must not be mixed with the fixed 20/50 cross-stock results above.

## Reproduce from the saved prices

From the repository root, after installing `requirements.txt`:

```bash
python scripts/cost_sensitivity.py --csv-dir examples/2026-11-20/inputs --fast 20 --slow 50 --train-size 504 --test-size 126 --cost-grid 0,5,10,20 --allocation 0.5 --report-dir results/recheck_fixed_rule
python scripts/walk_forward.py --csv examples/2026-11-20/inputs/RELIANCE.NS.csv --train-size 504 --test-size 126 --fast-grid 10,20,30 --slow-grid 50,100 --cost-bps 5 --allocation 0.5 --report-dir results/recheck_tuned_reliance
```

The input files are the saved `yfinance` split/dividend-adjusted close series downloaded on 22 November 2026. The generated reports include price snapshots and SHA-256 hashes in their run manifests. Future adjustments to upstream prices may differ from these snapshots. Adjusted history also does not prove exactly what data was available to a researcher on each historical date. The chosen rule, asset set, and parameter grid may reflect researcher choice after viewing historical data; the test blocks should not be treated as a fully independent prospective trial.

The European Black-Scholes-Merton pricer in `trading_analysis/options.py` is a **separate** analytical module. It does not provide signals to this stock strategy or simulate options trades.
