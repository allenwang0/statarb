import pandas as pd
import streamlit as st
import yfinance as yf
from yfinance.exceptions import YFException

from src import charts
from src.data import DataError, close_prices, select_pair
from src.metrics import metrics_table
from src.research import DEFAULT_TRAIN_FRACTION, default_train_end, evaluate, takeaway
from src.scanner import DEFAULT_UNIVERSE, scan
from src.stats import SIGNIFICANCE

DEFAULT_Y, DEFAULT_X = "PEP", "KO"
WARMUP = 60  # bars used to fit the hedge model before trading

st.set_page_config(page_title="statarb", layout="wide")
st.title("statarb: Kalman-filter pairs trading")

for key, value in (("ticker_y", DEFAULT_Y), ("ticker_x", DEFAULT_X), ("ran", False)):
    st.session_state.setdefault(key, value)

sidebar = st.sidebar
sidebar.header("Pair")
col1, col2 = sidebar.columns(2)
col1.text_input("Target (Y)", key="ticker_y")
col2.text_input("Reference (X)", key="ticker_x")
start_date = sidebar.date_input("Start date", pd.to_datetime("2018-01-01"))
end_date = sidebar.date_input("End date", pd.to_datetime("2024-12-31"))
train_fraction = sidebar.slider(
    "Train share of the period", 0.4, 0.8, DEFAULT_TRAIN_FRACTION, 0.05,
    help="Thresholds and delta are picked on the train window; results are "
    "reported separately for the later test window.",
)  # fmt: skip

sidebar.header("Strategy")
tuned = sidebar.checkbox("Pick entry, exit and delta on the train window", value=True)
entry_z = sidebar.slider("Entry threshold (z)", 1.0, 3.0, 2.0, 0.1, disabled=tuned)
exit_z = sidebar.slider("Exit threshold (z)", 0.0, 1.0, 0.0, 0.1, disabled=tuned)
delta = sidebar.select_slider(
    "Kalman delta (how fast beta adapts)",
    options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    value=1e-4,
    format_func=lambda d: f"{d:.0e}",
    disabled=tuned,
)
z_method = sidebar.radio(
    "z-score", ["model", "rolling"],
    format_func=lambda m: "innovation / sqrt(S)" if m == "model" else "innovation / 30-day std",
)  # fmt: skip
commission_bps = sidebar.number_input("Commission (bps per fill)", 0.0, 50.0, 1.0, 0.5)
slippage_bps = sidebar.number_input("Slippage (bps per fill)", 0.0, 50.0, 1.0, 0.5)
if sidebar.button("Run backtest", type="primary", key="run"):
    st.session_state.ran = True


def download(tickers):
    """Adjusted daily closes, one column per ticker. Stops the app on failure."""
    try:
        raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    except (OSError, YFException) as e:  # curl_cffi network errors subclass OSError
        st.error(f"Download from Yahoo Finance failed: {e}")
        st.stop()
    try:
        return close_prices(raw)
    except DataError as e:
        st.error(f"{e} between {start_date} and {end_date}.")
        st.stop()


def show_backtest():
    ticker_y = st.session_state.ticker_y.strip().upper()
    ticker_x = st.session_state.ticker_x.strip().upper()
    with st.spinner(f"Downloading {ticker_y} and {ticker_x} and running both models..."):
        prices = download([ticker_x, ticker_y])
        try:
            data = select_pair(prices, ticker_y=ticker_y, ticker_x=ticker_x).dropna()
            ev = evaluate(
                data,
                train_end=default_train_end(data.index, train_fraction),
                tuned=tuned,
                kalman_params=dict(entry_threshold=entry_z, exit_threshold=exit_z, delta=delta),
                ols_params=dict(entry_threshold=entry_z, exit_threshold=exit_z),
                warmup=WARMUP,
                z_method=z_method,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
            )
        except DataError as e:
            st.error(f"{ticker_y}/{ticker_x} between {start_date} and {end_date}: {e}.")
            st.stop()

    stats = ev.stats
    tiles = st.columns(5)
    eg_ok = stats.cointegrated
    tiles[0].metric(f"{'✓ pass' if eg_ok else '✗ fail'}: Engle-Granger p", f"{stats.eg_pvalue:.3f}")
    adf_ok = stats.adf_pvalue <= SIGNIFICANCE
    tiles[1].metric(
        f"{'✓ pass' if adf_ok else '✗ fail'}: ADF p (spread)", f"{stats.adf_pvalue:.3f}"
    )
    tiles[2].metric("Half-life (days)", f"{stats.half_life:.1f}")
    for tile, label, start, end in (
        (tiles[3], "Train window", ev.train_start, ev.train_end),
        (tiles[4], "Test window", ev.test_start, ev.test_end),
    ):
        tile.markdown(f"{label}  \n**{start:%Y-%m-%d}**  \nto **{end:%Y-%m-%d}**")
    st.caption("Pair statistics use the train window only, with a static OLS hedge ratio.")
    if not eg_ok:
        st.warning(
            f"{ticker_y}/{ticker_x} fails the Engle-Granger test on the train window "
            f"(p = {stats.eg_pvalue:.3f} > {SIGNIFICANCE}). The spread may not mean-revert."
        )

    st.subheader("Results after costs")
    st.markdown(f"**{takeaway(ev)}**")
    table = metrics_table(
        {
            f"{name}, {window}": getattr(run, window)
            for name, run in ev.runs.items()
            for window in ("train", "test")
        }
    )
    st.dataframe(table, width="stretch")
    st.caption(
        "Chosen parameters: "
        + "; ".join(
            f"{name}: " + ", ".join(f"{k}={v:g}" for k, v in run.params.items())
            for name, run in ev.runs.items()
        )
    )

    kalman = ev.runs["Kalman"].result
    ols = ev.runs["Rolling OLS"].result

    st.subheader("Equity and drawdown")
    st.plotly_chart(charts.equity_and_drawdown(ev.runs, ev), width="stretch")

    st.subheader("Prices and spread")
    st.plotly_chart(charts.prices_and_spread(data, kalman, ev, ticker_y, ticker_x), width="stretch")

    st.subheader("Z-score and trades")
    shown = st.radio("Signals for", list(ev.runs), horizontal=True)
    params = ev.runs[shown].params
    fig_z = charts.zscore_chart(
        ev.runs[shown].result, params["entry_threshold"], params["exit_threshold"], shown
    )
    st.plotly_chart(fig_z, width="stretch")

    st.subheader("Hedge ratio")
    gap = (kalman["hedge_ratio"] - ols["hedge_ratio"]).abs().mean()
    st.caption(f"Kalman and rolling-OLS beta differ by {gap:.3f} on average after warm-up.")
    st.plotly_chart(charts.hedge_ratio_chart(kalman, ols, ticker_y, ticker_x), width="stretch")

    st.subheader("Does the model fit?")
    st.caption(charts.diagnostics_takeaway(kalman))
    st.plotly_chart(charts.innovation_diagnostics(kalman), width="stretch")


def load_selected_pair():
    rows = st.session_state.scan_table.selection.rows
    if rows:
        row = st.session_state.scan_result.iloc[rows[0]]
        st.session_state.ticker_y, st.session_state.ticker_x = row["Y"], row["X"]
        st.session_state.ran = True


def show_scanner():
    st.markdown(
        "Engle-Granger p-values for every pair in the list, adjusted with "
        "Benjamini-Hochberg at 5%. Select a row to load the pair into the backtest tab."
    )
    universe = st.text_area("Tickers", ", ".join(DEFAULT_UNIVERSE))
    if st.button("Scan pairs", key="scan"):
        tickers = sorted({t.strip().upper() for t in universe.split(",") if t.strip()})
        with st.spinner(f"Downloading {len(tickers)} tickers..."):
            prices = download(tickers)
        missing = [t for t in tickers if t not in prices or prices[t].isna().all()]
        if missing:
            st.warning(f"No price data for {', '.join(missing)}; skipped.")
        st.session_state.scan_result = scan(prices.drop(columns=missing, errors="ignore"))
    if "scan_result" in st.session_state:
        result = st.session_state.scan_result
        st.caption(f"{int(result['passes_bh'].sum())} of {len(result)} pairs pass at 5% after BH.")
        st.dataframe(
            result,
            key="scan_table",
            on_select=load_selected_pair,
            selection_mode="single-row",
            width="stretch",
            column_config={
                "p_value": st.column_config.NumberColumn("p-value", format="%.4f"),
                "adjusted_p": st.column_config.NumberColumn("BH-adjusted p", format="%.4f"),
                "half_life_days": st.column_config.NumberColumn("Half-life (days)", format="%.1f"),
                "beta": st.column_config.NumberColumn("OLS beta", format="%.2f"),
                "passes_bh": st.column_config.CheckboxColumn("Passes"),
            },
        )


backtest_tab, scanner_tab = st.tabs(["Backtest", "Pair scanner"])
with backtest_tab:
    if st.session_state.ran:
        show_backtest()
    else:
        st.markdown(
            f"Backtests one pair on daily closes from Yahoo Finance. A Kalman filter "
            f"estimates the hedge ratio; a rolling OLS regression with the same signal, "
            f"sizing and costs is the baseline. Entry, exit and delta are picked on the "
            f"train window and reported on the later test window.\n\n"
            f"Default pair: **{DEFAULT_Y}** (Y) against **{DEFAULT_X}** (X). "
            f"Set the pair and dates in the sidebar, then press **Run backtest**."
        )
with scanner_tab:
    show_scanner()
