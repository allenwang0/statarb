import streamlit as st
import pandas as pd
import yfinance as yf
from src.backtester import Backtester

st.set_page_config(layout="wide")
st.title("⚡ Kalman Filter Statistical Arbitrage")

# Sidebar Controls
st.sidebar.header("Configuration")
ticker_y = st.sidebar.text_input("Dependent Asset (Y)", "PEP")
ticker_x = st.sidebar.text_input("Independent Asset (X)", "KO")
entry_z = st.sidebar.slider("Entry Z-Score", 1.0, 3.0, 2.0)

if st.sidebar.button("Run Backtest"):
    with st.spinner("Fetching data and running simulation..."):
        # 1. Fetch Data
        data = yf.download([ticker_x, ticker_y], start="2023-01-01", end="2024-01-01")['Adj Close']
        data.columns = ['asset_x', 'asset_y'] if data.columns[0] == ticker_x else ['asset_y', 'asset_x']
        data = data.dropna()

        # 2. Run Backtest
        bt = Backtester(data, entry_threshold=entry_z)
        results = bt.run()

        # 3. Metrics
        total_return = (results['equity'].iloc[-1] / 100000 - 1) * 100
        st.metric("Total Return", f"{total_return:.2f}%")

        # 4. Visualization
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Equity Curve")
            st.line_chart(results['equity'])

        with col2:
            st.subheader("Dynamic Hedge Ratio (Beta)")
            st.line_chart(results['hedge_ratio'])

        st.subheader("Spread Z-Score")
        st.area_chart(results['z_score'])