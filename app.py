import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.backtester import Backtester

# Page Config
st.set_page_config(page_title="StatArb Dashboard", layout="wide")
st.title("⚡ Kalman Filter Statistical Arbitrage")
st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
col1, col2 = st.sidebar.columns(2)
ticker_y = col1.text_input("Target (Y)", "PEP")
ticker_x = col2.text_input("Reference (X)", "KO")

st.sidebar.markdown("### Strategy Parameters")
entry_z = st.sidebar.slider("Entry Threshold (Z-Score)", 1.0, 3.0, 2.0, 0.1)
exit_z = st.sidebar.slider("Exit Threshold (Z-Score)", 0.0, 1.0, 0.0, 0.1)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))


# --- Helper Functions ---
def calculate_metrics(equity_curve):
    """Calculates key quant metrics: Total Return, Sharpe, Max Drawdown"""
    returns = equity_curve.pct_change().dropna()

    # 1. Total Return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

    # 2. Sharpe Ratio (Annualized, assuming daily data & 0 risk-free rate)
    if returns.std() == 0:
        sharpe = 0
    else:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    # 3. Max Drawdown
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_drawdown = drawdown.min() * 100

    return total_return, sharpe, max_drawdown


# --- Main Execution ---
if st.sidebar.button("Run Backtest", type="primary"):
    with st.spinner("Fetching market data & Simulating trades..."):
        try:
            # 1. Robust Data Fetching
            tickers = [ticker_x, ticker_y]
            raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)

            # Handle MultiIndex columns if present
            if isinstance(raw_data.columns, pd.MultiIndex):
                # Try to access 'Adj Close', fallback to 'Close'
                if 'Adj Close' in raw_data.columns.get_level_values(0):
                    data = raw_data['Adj Close']
                elif 'Close' in raw_data.columns.get_level_values(0):
                    data = raw_data['Close']
                else:
                    st.error("Data source missing price columns.")
                    st.stop()
            else:
                data = raw_data

            # Drop NaNs
            data = data.dropna()

            if data.shape[1] < 2:
                st.error(f"Insufficient data found. Columns: {data.columns}")
                st.stop()

            # Rename columns safely
            # We map user input tickers to the actual downloaded columns
            try:
                data = data[[ticker_x, ticker_y]].copy()
                data.columns = ['asset_x', 'asset_y']
            except KeyError:
                # Fallback: just take the first two columns if names don't match exactly
                data.columns = ['asset_x', 'asset_y']

            # 2. Run Backtest
            bt = Backtester(data, entry_threshold=entry_z, exit_threshold=exit_z)
            results = bt.run()

            # 3. Calculate Metrics
            tot_ret, sharpe, max_dd = calculate_metrics(results['equity'])

            # 4. Display Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Return", f"{tot_ret:.2f}%", delta_color="normal")
            m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
            m3.metric("Max Drawdown", f"{max_dd:.2f}%", delta_color="inverse")

            # 5. Plotly Visualizations

            # Chart A: Equity Curve
            st.subheader("Performance Over Time")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=results.index, y=results['equity'],
                mode='lines', name='Portfolio Value',
                line=dict(color='#00CC96', width=2)
            ))
            fig_eq.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_eq, use_container_width=True)

            # Chart B: The Strategy Internals (Z-Score vs Thresholds)
            st.subheader("Strategy Signals (Z-Score)")
            fig_z = go.Figure()

            # The Z-Score
            fig_z.add_trace(go.Scatter(
                x=results.index, y=results['z_score'],
                mode='lines', name='Spread Z-Score',
                line=dict(color='#636EFA', width=1)
            ))

            # Upper Threshold
            fig_z.add_trace(go.Scatter(
                x=results.index, y=[entry_z] * len(results),
                mode='lines', name='Short Threshold',
                line=dict(color='red', dash='dash')
            ))

            # Lower Threshold
            fig_z.add_trace(go.Scatter(
                x=results.index, y=[-entry_z] * len(results),
                mode='lines', name='Long Threshold',
                line=dict(color='green', dash='dash')
            ))

            fig_z.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_z, use_container_width=True)

            # Chart C: Hedge Ratio (Beta)
            with st.expander("See Dynamic Hedge Ratio (Kalman Filter State)"):
                fig_beta = go.Figure()
                fig_beta.add_trace(go.Scatter(
                    x=results.index, y=results['hedge_ratio'],
                    mode='lines', name='Beta',
                    line=dict(color='#FFA15A')
                ))
                fig_beta.update_layout(title=f"Hedge Ratio ({ticker_y} vs {ticker_x})", template="plotly_dark",
                                       height=300)
                st.plotly_chart(fig_beta, use_container_width=True)

        except Exception as e:
            st.error(f"Runtime Error: {str(e)}")
            st.exception(e)