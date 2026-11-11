import base64
import json
import os

import numpy as np

import pandas as pd
import pytest
import yfinance
from streamlit.testing.v1 import AppTest

from conftest import make_pair

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app.py')


def fake_download(prices):
    # Stands in for yf.download: ('Close', SYMBOL) columns sorted by symbol.
    def download(tickers, **kwargs):
        frame = prices[sorted(prices.columns)]
        frame.columns = pd.MultiIndex.from_product([['Close'], frame.columns])
        return frame
    return download


def run_app(monkeypatch, download, ticker_y='PEP', ticker_x='KO'):
    monkeypatch.setattr(yfinance, 'download', download)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].set_value(ticker_y)
    at.text_input[1].set_value(ticker_x)
    at.button[0].click().run()
    return at


def synthetic_prices(columns):
    # Synthetic prices, not market data.
    # columns[1] = 2 * columns[0] + noise, so the hedge ratio shows which
    # one the app treated as Y.
    pair = make_pair(n=250, beta=2.0, phi=0.9, level=50.0, seed=0)
    return pd.DataFrame({columns[0]: pair['asset_x'], columns[1]: pair['asset_y']})


def test_lowercase_tickers_run_with_y_as_target(monkeypatch):
    at = run_app(monkeypatch, fake_download(synthetic_prices(['PEP', 'KO'])),
                 ticker_y='ko', ticker_x='pep')
    assert not at.exception and not at.error
    spec = json.loads(at.get('plotly_chart')[2].proto.spec)
    assert spec['layout']['title']['text'] == 'Hedge Ratio (KO vs PEP)'
    beta = np.frombuffer(base64.b64decode(spec['data'][0]['y']['bdata']), dtype='f8')
    assert abs(beta[-1] - 2.0) < 0.2  # KO is Y; reversed would give about 0.5


def test_ticker_with_no_data_is_named(monkeypatch):
    # An unknown symbol comes back as an all-NaN column. The user should be
    # told which ticker failed, not shown an indexing traceback.
    prices = synthetic_prices(['KO', 'XYZQ'])
    prices['XYZQ'] = float('nan')
    at = run_app(monkeypatch, fake_download(prices), ticker_y='XYZQ', ticker_x='KO')
    assert not at.exception
    assert 'No price data for XYZQ' in at.error[0].value


def test_download_failure_is_reported(monkeypatch):
    def broken(tickers, **kwargs):
        raise ConnectionError('network down')
    at = run_app(monkeypatch, broken)
    assert not at.exception
    assert 'Download from Yahoo Finance failed' in at.error[0].value


def test_other_errors_are_not_swallowed(monkeypatch):
    # A bug in the backtest must surface as an exception, not a red box
    # that looks like a data problem.
    import src.backtester
    def boom(self):
        raise RuntimeError('bug')
    monkeypatch.setattr(src.backtester.Backtester, 'run', boom)
    at = run_app(monkeypatch, fake_download(synthetic_prices(['PEP', 'KO'])))
    assert not at.error
    assert 'bug' in at.exception[0].message


def test_failed_ticker_does_not_blame_the_good_one(monkeypatch):
    # yfinance adds an 'Adj Close' column holding only the failed ticker.
    # The app used to read that column and report KO, which downloaded
    # fine, as missing.
    from test_data import one_ticker_failed_download
    at = run_app(monkeypatch, lambda tickers, **kw: one_ticker_failed_download(),
                 ticker_y='KO', ticker_x='XYZQQ')
    assert not at.exception
    assert at.error[0].value.startswith('No price data for XYZQQ between')
