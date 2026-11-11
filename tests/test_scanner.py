import numpy as np
import pandas as pd

from src.scanner import scan


def test_scanner_finds_the_planted_pair_and_controls_false_discoveries():
    # Synthetic universe: A and B are cointegrated, C through F are
    # independent random walks. With 15 pairs, raw p < 0.05 would admit
    # about one false pair by chance; BH should leave only A/B.
    rng = np.random.default_rng(3)
    n = 750
    walks = {t: 50 * np.exp(np.cumsum(0.01 * rng.standard_normal(n))) for t in "ACDEF"}
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.8 * spread[t - 1] + 0.3 * rng.standard_normal()
    walks["B"] = 1.2 * walks["A"] + spread
    table = scan(pd.DataFrame(walks))
    assert len(table) == 15
    passing = table[table["passes_bh"]]
    assert [set(p) for p in zip(passing["Y"], passing["X"], strict=True)] == [{"A", "B"}]
    assert (table["adjusted_p"] >= table["p_value"]).all()
    assert abs(passing["half_life_days"].iloc[0] - np.log(2) / -np.log(0.8)) < 1.0
