import base64
import json
import os

import numpy as np
import pandas as pd
import yfinance
from streamlit.testing.v1 import AppTest

from conftest import make_pair
from test_data import one_ticker_failed_download

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def fake_download(prices):
    # Stands in for yf.download: ('Close', SYMBOL) columns sorted by symbol.
    def download(tickers, **kwargs):
        frame = prices[sorted(prices.columns)]
        frame.columns = pd.MultiIndex.from_product([["Close"], frame.columns])
        return frame

    return download


def start_app(monkeypatch, download):
    monkeypatch.setattr(yfinance, "download", download)
    return AppTest.from_file(APP, default_timeout=60).run()


def run_app(monkeypatch, download, ticker_y="PEP", ticker_x="KO"):
    at = start_app(monkeypatch, download)
    at.text_input(key="ticker_y").set_value(ticker_y)
    at.text_input(key="ticker_x").set_value(ticker_x)
    at.button(key="run").click().run()
    return at


def synthetic_prices(columns, n=400):
    # Synthetic prices, not market data.
    # columns[1] = 2 * columns[0] + noise, so the hedge ratio shows which
    # one the app treated as Y.
    pair = make_pair(n=n, beta=2.0, phi=0.9, level=50.0, seed=0)
    return pd.DataFrame({columns[0]: pair["asset_x"], columns[1]: pair["asset_y"]})


def chart_specs(at):
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def trace_y(trace):
    y = trace["y"]
    if isinstance(y, dict):
        return np.frombuffer(base64.b64decode(y["bdata"]), dtype=y["dtype"])
    return np.array(y, dtype=float)


def test_before_run_shows_description_and_default_pair(monkeypatch):
    at = start_app(monkeypatch, fake_download(synthetic_prices(["KO", "PEP"])))
    assert not at.exception
    assert "Default pair: **PEP** (Y) against **KO** (X)" in at.markdown[0].value
    assert not at.get("plotly_chart")


def test_every_tile_and_chart_renders(monkeypatch):
    at = run_app(monkeypatch, fake_download(synthetic_prices(["KO", "PEP"])))
    assert not at.exception and not at.error
    labels = [m.label for m in at.metric]
    assert labels[0] == "✓ pass: Engle-Granger p"
    assert "Half-life (days)" in labels
    assert any(m.value.startswith("Test window") for m in at.markdown)
    specs = chart_specs(at)
    assert len(specs) == 6
    for spec in specs:  # every panel says what it measures and in what unit
        axes = [v for k, v in spec["layout"].items() if k.startswith("yaxis")]
        assert axes and all(a.get("title", {}).get("text") for a in axes)
    assert not at.warning  # the synthetic pair is cointegrated
    table = at.dataframe[0].value
    assert list(table.columns) == [
        "Kalman, train", "Kalman, test", "Rolling OLS, train", "Rolling OLS, test",
    ]  # fmt: skip
    assert any("Sharpe on the test window" in m.value for m in at.markdown)


def test_lowercase_tickers_run_with_y_as_target(monkeypatch):
    at = run_app(monkeypatch, fake_download(synthetic_prices(["PEP", "KO"])),
                 ticker_y="ko", ticker_x="pep")  # fmt: skip
    assert not at.exception and not at.error
    hedge = [s for s in chart_specs(at) if "beta (shares" in json.dumps(s["layout"])][0]
    assert hedge["layout"]["yaxis"]["title"]["text"] == "beta (shares PEP per share KO)"
    kalman_beta = [t for t in hedge["data"] if t.get("name") == "Kalman beta"][0]
    assert abs(trace_y(kalman_beta)[-1] - 2.0) < 0.2  # KO is Y; reversed gives about 0.5


def test_unrelated_pair_gets_a_warning(monkeypatch):
    rng = np.random.default_rng(1)
    walks = 50 * np.exp(np.cumsum(0.01 * rng.standard_normal((400, 2)), axis=0))
    prices = pd.DataFrame(
        walks, columns=["KO", "PEP"], index=pd.bdate_range("2020-01-01", periods=400)
    )
    at = run_app(monkeypatch, fake_download(prices))
    assert not at.exception
    assert "fails the Engle-Granger test" in at.warning[0].value
    assert [m.label for m in at.metric][0] == "✗ fail: Engle-Granger p"


def test_ticker_with_no_data_is_named(monkeypatch):
    # An unknown symbol comes back as an all-NaN column. The user should be
    # told which ticker failed, not shown an indexing traceback.
    prices = synthetic_prices(["KO", "XYZQ"])
    prices["XYZQ"] = float("nan")
    at = run_app(monkeypatch, fake_download(prices), ticker_y="XYZQ", ticker_x="KO")
    assert not at.exception
    assert "No price data for XYZQ" in at.error[0].value


def test_download_failure_is_reported(monkeypatch):
    def broken(tickers, **kwargs):
        raise ConnectionError("network down")

    at = run_app(monkeypatch, broken)
    assert not at.exception
    assert "Download from Yahoo Finance failed" in at.error[0].value


def test_other_errors_are_not_swallowed(monkeypatch):
    # A bug in the backtest must surface as an exception, not a red box
    # that looks like a data problem.
    import src.backtester

    def boom(self):
        raise RuntimeError("bug")

    monkeypatch.setattr(src.backtester.Backtester, "run", boom)
    at = run_app(monkeypatch, fake_download(synthetic_prices(["PEP", "KO"])))
    assert not at.error
    assert "bug" in at.exception[0].message


def test_failed_ticker_does_not_blame_the_good_one(monkeypatch):
    # yfinance adds an 'Adj Close' column holding only the failed ticker.
    # The app used to read that column and report KO, which downloaded
    # fine, as missing.
    at = run_app(monkeypatch, lambda tickers, **kw: one_ticker_failed_download(),
                 ticker_y="KO", ticker_x="XYZQQ")  # fmt: skip
    assert not at.exception
    assert at.error[0].value.endswith(": No price data for XYZQQ.")


def test_unusable_pair_is_reported(monkeypatch):
    # Two tickers whose prices are exact multiples leave no spread; the
    # filter's warm-up check must reach the user as a message, not a crash.
    prices = synthetic_prices(["PEP", "KO"])
    prices["KO"] = 2.0 * prices["PEP"]
    at = run_app(monkeypatch, fake_download(prices))
    assert not at.exception
    assert "exact linear function" in at.error[0].value


def test_too_short_a_range_is_reported(monkeypatch):
    at = run_app(monkeypatch, fake_download(synthetic_prices(["PEP", "KO"], n=80)))
    assert not at.exception
    assert "train window needs more than 60 bars" in at.error[0].value


def test_scanner_lists_pairs(monkeypatch):
    prices = synthetic_prices(["KO", "PEP"])
    prices["WMT"] = 50 * np.exp(np.cumsum(0.01 * np.random.default_rng(2).standard_normal(400)))
    at = start_app(monkeypatch, fake_download(prices))
    at.text_area[0].set_value("ko, pep, wmt")
    at.button(key="scan").click().run()
    assert not at.exception
    table = at.dataframe[0].value
    assert len(table) == 3
    assert set(table.iloc[0][["Y", "X"]]) == {"KO", "PEP"}  # the planted pair ranks first
