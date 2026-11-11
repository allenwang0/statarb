import numpy as np
import pandas as pd

TRADING_DAYS = 252  # NYSE sessions per year, used to annualize daily figures

LABELS = {
    "annual_return": "Annualized return",
    "annual_vol": "Annualized volatility",
    "sharpe": "Sharpe (rf = 0)",
    "sortino": "Sortino (rf = 0)",
    "max_drawdown": "Max drawdown",
    "trades": "Trades",
    "avg_holding_days": "Average holding period (days)",
    "hit_rate": "Hit rate (closed trades)",
    "turnover": "Annual turnover (x equity)",
    "cost_drag": "Annual cost drag",
}
PERCENT = {"annual_return", "annual_vol", "max_drawdown", "hit_rate", "cost_drag"}


def _window(result, start, end):
    """Rows in [start, end] and the equity of the bar before them as the base.
    A window starting at bar 0 uses bar 0 as the base."""
    in_window = np.ones(len(result), dtype=bool)
    if start is not None:
        in_window &= result.index >= pd.Timestamp(start)
    if end is not None:
        in_window &= result.index <= pd.Timestamp(end)
    rows = np.flatnonzero(in_window)
    if len(rows) < 2:
        raise ValueError("Metrics need at least two bars in the window")
    first = rows[0]
    if first == 0:
        return result.iloc[rows[1:]], result["equity"].iloc[0]
    return result.iloc[rows], result["equity"].iloc[first - 1]


def trade_pnls(result):
    """P&L of each closed trade, indexed by exit date: equity at the exit bar
    minus equity on the bar before entry, so both fills' costs count."""
    pos = result["position"].to_numpy()
    equity = result["equity"].to_numpy()
    pnls, exit_dates, entry = [], [], None
    for t in range(1, len(pos)):
        if pos[t - 1] == 0 and pos[t] != 0:
            entry = t
        elif pos[t - 1] != 0 and pos[t] == 0 and entry is not None:
            pnls.append(equity[t] - equity[entry - 1])
            exit_dates.append(result.index[t])
            entry = None
    return pd.Series(pnls, index=pd.Index(exit_dates), dtype=float)


def compute_metrics(result, start=None, end=None):
    """
    Performance of a Backtester result between start and end (inclusive).
    Returns are daily simple returns of equity; the first return in the
    window is measured from the previous bar's equity.
    """
    window, base = _window(result, start, end)
    equity = window["equity"]
    curve = pd.concat([pd.Series([base]), equity.reset_index(drop=True)])
    returns = curve.pct_change().dropna()
    n = len(returns)
    years = n / TRADING_DAYS
    mean, sd = returns.mean(), returns.std()
    downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))

    pos = result["position"]
    prev = pos.shift(1).fillna(0)
    trades = int(((prev == 0) & (pos != 0)).loc[window.index].sum())
    closed = trade_pnls(result)
    closed_in_window = closed[closed.index.isin(window.index)]

    avg_equity = equity.mean()
    return {
        "annual_return": (equity.iloc[-1] / base) ** (1 / years) - 1,
        "annual_vol": sd * np.sqrt(TRADING_DAYS),
        "sharpe": mean / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0,
        "sortino": mean / downside * np.sqrt(TRADING_DAYS) if downside > 0 else np.nan,
        "max_drawdown": (curve / curve.cummax() - 1).min(),
        "trades": trades,
        "avg_holding_days": (window["position"] != 0).sum() / trades if trades else np.nan,
        "hit_rate": (closed_in_window > 0).mean() if len(closed_in_window) else np.nan,
        "turnover": window["traded_notional"].sum() / avg_equity / years,
        "cost_drag": window["cost"].sum() / avg_equity / years,
    }


def drawdown(equity):
    return equity / equity.cummax() - 1


def format_metric(key, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if key in PERCENT:
        return f"{value:.1%}"
    if key == "trades":
        return f"{int(value)}"
    return f"{value:.2f}"


def metrics_table(columns):
    """columns: {name: metrics dict}. Returns a formatted DataFrame."""
    return pd.DataFrame(
        {name: [format_metric(k, m[k]) for k in LABELS] for name, m in columns.items()},
        index=list(LABELS.values()),
    )
