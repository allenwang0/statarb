# ⚡ Statistical Arbitrage with Kalman Filters

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url-here.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A **low-latency, event-driven backtesting engine** for pairs trading strategies. 

Unlike traditional moving-average models, this engine uses a **Kalman Filter** to dynamically estimate the hedge ratio (beta) between two assets in real-time, allowing for faster adaptation to changing market regimes without the lag of rolling windows.

**[👉 View Live Dashboard](https://your-app-url-here.streamlit.app/)**

---

## 🚀 Key Features

### 1. Mathematical Core (The "Quant" Side)
* **Dynamic State Estimation:** Implements a linear Kalman Filter from scratch (using `numpy` matrix algebra) to track the unobservable "true" spread between assets.
* **Regime Adaptation:** Automatically adjusts the hedge ratio ($\beta$) as the correlation between assets drifts, avoiding the "lookback bias" of fixed-window OLS.
* **Mean Reversion:** Generates trading signals based on Z-Score deviation from the predicted state.

### 2. Engineering Architecture (The "Dev" Side)
* **Event-Driven Engine:** Simulates a real-time feed by iterating row-by-row. Prevents **look-ahead bias** by strictly separating *current* observation from *future* data.
* **Interactive Visualization:** Built with **Streamlit** and **Plotly** for institutional-grade dashboards (Equity Curves, Drawdown analysis, interactive zooming).
* **Robust Error Handling:** Automated data validation and connection handling for Yahoo Finance APIs.

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

This allows the model to "learn" the new relationship instantly when volatility spikes.

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
