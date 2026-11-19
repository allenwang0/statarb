# statarb

A pairs-trading research app. A Kalman filter (a model that updates with every new price) tracks the hedge ratio between two stocks day by day, and every strategy is tuned on one period and judged on a later, untouched one.

![The app on PEP/KO, 2018 to 2024](docs/app-1280-light.png)

## Highlights

- **Kalman-filter hedge ratio** with a time-varying intercept, scaled so one setting adapts at the same speed for a $5 stock and a $500 stock.
- **Pair scanner** that tests for cointegration (prices that move together over time) with Benjamini-Hochberg correction, so screening 55 pairs does not produce a list of false positives.
- **Train/test discipline:** parameters are fit on the first 60% of the period and results are reported separately for the rest.
- **Fair baseline:** a rolling least-squares hedge runs through the same signal, sizing and cost code, with diagnostics showing whether the Kalman model fits a given pair.

## Run it

```bash
git clone https://github.com/allenwang0/statarb.git && cd statarb
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/streamlit run app.py
```

## How it works

- The filter models one stock's price as intercept plus hedge ratio times the other's, letting both drift daily.
- The trading signal is a z-score: how far today's price gap is from the model's prediction, in standard deviations.
- Enter when the z-score passes a threshold, exit when it returns toward zero; position size equals account equity.
- Entry, exit and filter speed are tuned on the training window only.
- On PEP/KO the simpler rolling hedge did better out of sample; the diagnostics show why.

## Limitations

- Daily closes only, with fills at the signal's close; slippage is the only allowance for not getting that price.
- No borrow fees or margin interest, and one pair is traded at a time.
- Research tool only, no live trading. Not financial advice.
