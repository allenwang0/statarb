import pandas as pd


class DataError(ValueError):
    """The downloaded prices cannot be used for a backtest. The message is
    written for the user."""


def close_prices(raw):
    """
    Return the 'Close' level of a yfinance download (one column per ticker).

    Call yf.download with auto_adjust=True so 'Close' is already adjusted.
    'Adj Close' is ignored: when one ticker fails, yfinance adds an
    'Adj Close' column for the failed ticker only, and picking it would drop
    every ticker that did download.
    """
    if raw is None or raw.empty:
        raise DataError("Yahoo Finance returned no data")
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw
    if 'Close' not in raw.columns.get_level_values(0):
        raise DataError("Yahoo Finance returned no 'Close' prices")
    return raw['Close']


def select_pair(prices, ticker_y, ticker_x):
    """
    Map downloaded close prices (one column per ticker) to the
    ['asset_x', 'asset_y'] columns the Backtester expects, by ticker name.
    Raises DataError naming any ticker with no prices.
    """
    ticker_y = ticker_y.strip().upper()
    ticker_x = ticker_x.strip().upper()
    if not ticker_y or not ticker_x:
        raise DataError("Enter both tickers")
    if ticker_y == ticker_x:
        raise DataError(f"Y and X are both {ticker_y}; pick two different tickers")
    missing = [t for t in (ticker_y, ticker_x)
               if t not in prices.columns or prices[t].isna().all()]
    if missing:
        raise DataError(f"No price data for {', '.join(missing)}")
    data = prices[[ticker_x, ticker_y]].copy()
    data.columns = ['asset_x', 'asset_y']
    return data
