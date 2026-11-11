import pandas as pd
import pytest

from src.data import select_pair


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
    with pytest.raises(KeyError, match="XYZ"):
        select_pair(downloaded_prices(), ticker_y="KO", ticker_x="xyz")
