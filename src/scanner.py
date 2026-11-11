"""Screen every pair in a ticker list for cointegration."""

from itertools import combinations

import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

from src.stats import SIGNIFICANCE, half_life, ols_spread

# Eleven large US consumer-staples stocks; the list is a starting point, not a ranking.
DEFAULT_UNIVERSE = ("WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "KMB", "TGT")


def scan(prices, alpha=SIGNIFICANCE):
    """
    prices: one column per ticker, rows with any NaN dropped. For each
    unordered pair, run Engle-Granger with the first ticker as Y, then
    adjust the p-values with Benjamini-Hochberg so that, across many pairs,
    the expected share of false discoveries is at most `alpha`.
    """
    prices = prices.dropna()
    rows = []
    for y, x in combinations(prices.columns, 2):
        spread, beta = ols_spread(prices[y], prices[x])
        rows.append(
            {
                "Y": y,
                "X": x,
                "p_value": float(coint(prices[y], prices[x])[1]),
                "half_life_days": half_life(spread),
                "beta": float(beta),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    reject, adjusted, _, _ = multipletests(table["p_value"], alpha=alpha, method="fdr_bh")
    table["adjusted_p"] = adjusted
    table["passes_bh"] = reject
    return table.sort_values("adjusted_p").reset_index(drop=True)
