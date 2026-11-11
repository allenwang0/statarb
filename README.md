# ⚡ Statistical Arbitrage with Kalman Filters

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A daily-bar backtester for pairs trading strategies.

Instead of a rolling-window regression, it uses a **Kalman Filter** to update the hedge ratio (beta) between two assets one bar at a time. How fast the estimate adapts depends on the `delta` setting. It has not yet been compared against rolling OLS.

---

## 🚀 Key Features

### 1. Mathematical Core (The "Quant" Side)
* **Dynamic State Estimation:** Implements a linear Kalman Filter from scratch (using `numpy` matrix algebra) to track the hedge ratio and intercept of the pair, treated as hidden states.
* **Regime Adaptation:** Adjusts the hedge ratio ($\beta$) as the relationship between assets drifts. Compared with fixed-window OLS, the trade-off is lag against noise, set by `delta`.
* **Mean Reversion:** Generates trading signals based on Z-Score deviation from the predicted state.

### 2. Engineering Architecture (The "Dev" Side)
* **Bar-by-Bar Loop:** Iterates over daily bars. Each bar books P&L on the shares held since the previous bar, then updates the filter, then trades at that bar's close. Positions are sized from equity, and commission and slippage are charged on every fill.
* **Interactive Visualization:** Built with **Streamlit** and **Plotly** for interactive charts of the equity curve, z-score and hedge ratio. Max drawdown is shown as a single number.
* **Error Handling:** Reports Yahoo Finance download failures, tickers with no data, and date ranges too short for the warm-up. There is no retry.

---

## 📊 The Math Behind It: Why Kalman Filters?

In pairs trading, we model the relationship between Asset $Y$ and Asset $X$ as:
$$Y_t = \alpha + \beta_t X_t + \epsilon_t$$

### The Problem with Rolling Windows (OLS)
Standard approaches use a "Rolling OLS" (e.g., 60-day window) to find $\beta$.
* **Lag:** It reacts slowly to structural breaks.
* **Memory:** It gives equal weight to data from 59 days ago vs. yesterday.

### The Kalman Solution
We treat the hedge ratio $\beta_t$ as a "hidden state" that follows a random walk. The filter recursively updates its estimate using Bayesian inference:

1.  **Predict:** Estimate the state based on the previous step.
    $$\hat{x}_{k|k-1} = F_k \hat{x}_{k-1|k-1}$$
2.  **Update:** Correct the prediction using the new incoming price observation (Measurement).
    $$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - H_k \hat{x}_{k|k-1})$$

Because the state follows a random walk, recent bars count more than old ones. How quickly the filter follows a change is set by `delta`; it does not adapt instantly.

---

## 🛠️ Installation & Usage

### Local Development
```bash
# 1. Clone the repo
git clone https://github.com/allenwang0/statarb.git
cd statarb

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py

# 4. Run the tests (offline, synthetic data)
pip install -r requirements-dev.txt
pytest
```
