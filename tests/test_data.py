import numpy as np
import pandas as pd
import pytest

from src.data import DataError, close_prices, select_pair


def downloaded_prices():
    # yfinance returns upper-case symbols sorted by name: KO before PEP.
    # Prices are made up.
    return pd.DataFrame({'KO': [60.0, 61.0], 'PEP': [170.0, 171.0]})


def test_lowercase_tickers_keep_y_and_x():
    # Y="ko", X="pep" must not silently reverse the pair because the
    # downloaded columns happen to be sorted the other way.
    data = select_pair(downloaded_prices(), ticker_y=" ko", ticker_x="pep ")
    assert list(data['asset_y']) == [60.0, 61.0]
    assert list(data['asset_x']) == [170.0, 171.0]


def test_missing_ticker_is_named():
    with pytest.raises(DataError, match="XYZ"):
        select_pair(downloaded_prices(), ticker_y="KO", ticker_x="xyz")


def test_same_ticker_twice_is_rejected():
    # Y == X makes the regression exact (beta 1, zero residual), so the
    # filter's measurement noise is zero and every z-score is meaningless.
    with pytest.raises(DataError, match="two different"):
        select_pair(downloaded_prices(), ticker_y="ko", ticker_x="KO")


def one_ticker_failed_download():
    # Shape of yf.download(['KO', 'XYZQQ'], auto_adjust=True) when XYZQQ does
    # not exist (yfinance 1.8): an 'Adj Close' column appears holding only
    # the failed ticker. Prices are made up.
    dates = pd.bdate_range("2023-01-02", periods=3)
    columns = pd.MultiIndex.from_tuples(
        [('Adj Close', 'XYZQQ'), ('Close', 'KO'), ('Close', 'XYZQQ'),
         ('Volume', 'KO'), ('Volume', 'XYZQQ')], names=['Price', 'Ticker'])
    values = [[np.nan, 60.0, np.nan, 1e6, np.nan]] * 3
    return pd.DataFrame(values, index=dates, columns=columns)


def test_failed_ticker_does_not_hide_the_good_one():
    # Picking 'Adj Close' here dropped KO, and the app then told the user
    # that KO, which downloaded fine, had no data.
    prices = close_prices(one_ticker_failed_download())
    assert list(prices['KO']) == [60.0, 60.0, 60.0]
    with pytest.raises(DataError) as err:
        select_pair(prices, ticker_y="KO", ticker_x="XYZQQ")
    assert str(err.value) == "No price data for XYZQQ"


def test_empty_download_is_a_data_error():
    with pytest.raises(DataError, match="no data"):
        close_prices(pd.DataFrame())
