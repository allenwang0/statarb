import numpy as np

from conftest import make_pair
from src.stats import half_life, pair_stats


def test_cointegrated_pair_is_detected_with_its_half_life():
    # Synthetic pair: the spread is AR(1) with phi = 0.9, so the true
    # half-life is -ln 2 / ln 0.9 = 6.58 bars.
    data = make_pair(n=2000, beta=1.5, alpha=0.2, noise=0.01, phi=0.9, level=50.0, seed=11)
    stats = pair_stats(data["asset_y"], data["asset_x"], johansen=True)
    assert stats.eg_pvalue < 0.01 and stats.cointegrated
    assert stats.adf_pvalue < 0.01
    true_half_life = -np.log(2) / np.log(0.9)
    assert abs(stats.half_life - true_half_life) < 0.2 * true_half_life
    assert stats.johansen_trace > stats.johansen_crit_95


def test_independent_random_walks_are_rarely_called_cointegrated():
    # Under the null the Engle-Granger test should reject about 5% of the
    # time. A test that rejects more often would pass the warning banner on
    # unrelated stocks.
    not_rejected = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        x = 50 * np.exp(np.cumsum(0.01 * rng.standard_normal(500)))
        y = 50 * np.exp(np.cumsum(0.01 * rng.standard_normal(500)))
        not_rejected += pair_stats(y, x).eg_pvalue > 0.05
    assert not_rejected >= 90


def test_random_walk_spread_has_no_half_life():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.standard_normal(2000))
    assert half_life(walk) > 200 or half_life(walk) == float("inf")
