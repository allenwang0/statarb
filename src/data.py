def select_pair(prices, ticker_y, ticker_x):
    """
    Map downloaded close prices (one column per ticker) to the
    ['asset_x', 'asset_y'] columns the Backtester expects, by ticker name.
    Raises KeyError naming any ticker with no prices.
    """
    ticker_y = ticker_y.strip().upper()
    ticker_x = ticker_x.strip().upper()
    missing = [t for t in (ticker_y, ticker_x)
               if t not in prices.columns or prices[t].isna().all()]
    if missing:
        raise KeyError(f"No price data for {', '.join(missing)}")
    data = prices[[ticker_x, ticker_y]].copy()
    data.columns = ['asset_x', 'asset_y']
    return data
