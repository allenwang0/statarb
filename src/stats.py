from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

SIGNIFICANCE = 0.05


@dataclass
class PairStats:
    eg_pvalue: float  # Engle-Granger, H0: no cointegration
    adf_pvalue: float  # ADF on the OLS spread, H0: unit root
    half_life: float  # bars, from an AR(1) fit of the spread
    hedge_ratio: float  # static OLS beta of y on x
    johansen_trace: float | None = None  # trace statistic, H0: rank 0
    johansen_crit_95: float | None = None

    @property
    def cointegrated(self):
        return self.eg_pvalue <= SIGNIFICANCE


def ols_spread(y, x):
    """Residual of a static OLS of y on [x, 1], and the slope."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    beta, alpha = np.polyfit(x, y, 1)
    return y - alpha - beta * x, beta


def half_life(spread):
    """
    Fit s_t = c + phi * s_{t-1} + e_t and return -ln 2 / ln(phi), the bars it
    takes a deviation to halve. Returns inf when phi is outside (0, 1),
    meaning the spread does not mean-revert.
    """
    s = np.asarray(spread, dtype=float)
    phi, _ = np.polyfit(s[:-1], s[1:], 1)
    if not 0 < phi < 1:
        return float("inf")
    return float(-np.log(2) / np.log(phi))


def pair_stats(y, x, johansen=False):
    """Cointegration tests for one pair of price series."""
    spread, beta = ols_spread(y, x)
    eg_pvalue = float(coint(y, x)[1])
    adf_pvalue = float(adfuller(spread, result_object=False)[1])
    stats = PairStats(eg_pvalue, adf_pvalue, half_life(spread), float(beta))
    if johansen:
        # det_order 0: constant term; k_ar_diff 1: one lagged difference.
        result = coint_johansen(np.column_stack([y, x]), det_order=0, k_ar_diff=1)
        stats.johansen_trace = float(result.lr1[0])
        stats.johansen_crit_95 = float(result.cvt[0, 1])
    return stats
