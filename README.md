# statarb

A pairs-trading research app. A Kalman filter (a model that updates with every new price) tracks the hedge ratio between two stocks day by day, and every strategy is tuned on one period and judged on a later, untouched one.

**[Read the case study →](https://allenwang0.github.io/statarb/)**

![The app on PEP/KO, 2018 to 2024](docs/app-1280-light.png)

## Highlights

<!-- results:start -->
- **Out of sample on PEP/KO, the rolling least-squares hedge earned a Sharpe of 0.62** after 2 bp costs (2022-03-14 to 2024-12-30), 1.06 above the Kalman hedge, and stayed ahead in all 10 fill-lag and cost settings tested. With 2.8 test years the Sharpe's standard error is about ±0.65.
- **Next-day fills did not change the ranking**: 0.77 vs −0.52.
- **Pair scanner** tests all 55 pairs of 11 tickers for cointegration with Benjamini–Hochberg correction: 4 raw hits at p ≤ 0.05, 0 after correction.
- **Fair baseline:** both hedges share the signal, sizing, cost and tuning code.
<!-- results:end -->

## Run it

```bash
git clone https://github.com/allenwang0/statarb.git && cd statarb
python3.12 -m venv .venv   # Python 3.11 or 3.12
.venv/bin/pip install -e ".[dev]"
.venv/bin/streamlit run app.py
.venv/bin/python scripts/case_study.py        # results/case_study.json from data/prices.csv
python3 scripts/build_report.py --readme README.md   # docs/index.html and this list
```

## How it works

- The filter models one stock's price as intercept plus hedge ratio times the other's, letting both drift daily.
- The trading signal is a z-score: how far today's price gap is from the model's prediction, in standard deviations.
- Enter when the z-score passes a threshold, exit when it returns toward zero; position size equals account equity.
- Entry, exit and filter speed are tuned on the training window only.

## Limitations

- Daily closes only; fills at the signal's close by default, or one close later with `fill_lag=1`.
- No borrow fees or margin interest; one pair, one test window.
- Research tool only, no live trading. Not financial advice.
