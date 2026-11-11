import numpy as np
import pandas as pd
import pytest

from src.metrics import compute_metrics, trade_pnls

ANNUAL = np.sqrt(252)


def toy_result():
    # Hand-built six-bar result. Long spread bars 1-2, exit at 3, short at 4,
    # exit at 5. Daily returns: 0, +10%, -10%, 0, +10%.
    index = pd.bdate_range("2024-01-01", periods=6)
    return pd.DataFrame(
        {
            "equity": [100.0, 100.0, 110.0, 99.0, 99.0, 108.9],
            "position": [0, 1, 1, 0, -1, 0],
            "traded_notional": [0.0, 200.0, 0.0, 200.0, 200.0, 200.0],
            "cost": [0.0, 0.1, 0.0, 0.1, 0.1, 0.1],
        },
        index=index,
    )


def test_metrics_match_hand_computation():
    m = compute_metrics(toy_result())
    # mean 0.02, sample variance 0.028 / 4 = 0.007
    assert m["sharpe"] == pytest.approx(0.02 / np.sqrt(0.007) * ANNUAL)
    assert m["annual_vol"] == pytest.approx(np.sqrt(0.007) * ANNUAL)
    # downside deviation: sqrt(mean of [0, 0, 0.01, 0, 0])
    assert m["sortino"] == pytest.approx(0.02 / np.sqrt(0.002) * ANNUAL)
    assert m["max_drawdown"] == pytest.approx(99 / 110 - 1)
    assert m["annual_return"] == pytest.approx(1.089 ** (252 / 5) - 1)
    assert m["trades"] == 2
    assert m["avg_holding_days"] == pytest.approx(1.5)
    # Trade 1 loses 1 (100 -> 99), trade 2 gains 9.9 (99 -> 108.9).
    assert m["hit_rate"] == pytest.approx(0.5)
    avg_equity = np.mean([100.0, 110.0, 99.0, 99.0, 108.9])
    assert m["turnover"] == pytest.approx(800 / avg_equity / (5 / 252))
    assert m["cost_drag"] == pytest.approx(0.4 / avg_equity / (5 / 252))


def test_trade_pnl_includes_entry_costs():
    # Entry costs are paid on the entry bar, so a trade's P&L starts from
    # the equity before entry, not after.
    assert list(trade_pnls(toy_result())) == pytest.approx([-1.0, 9.9])


def test_window_measures_from_the_previous_bar():
    # Test-window metrics must not count train-window P&L, but the first
    # test bar's return is measured from the last train bar.
    m = compute_metrics(toy_result(), start="2024-01-04")
    assert m["max_drawdown"] == pytest.approx(99 / 110 - 1)
    assert m["annual_return"] == pytest.approx((108.9 / 110) ** (252 / 3) - 1)
    assert m["trades"] == 1
    # A trade counts toward the window it exits in: both exits (bars 3, 5) are.
    assert m["hit_rate"] == pytest.approx(0.5)
