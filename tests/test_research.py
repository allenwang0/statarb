import numpy as np

from conftest import make_pair
from src.research import evaluate, takeaway, tune


def synthetic():
    return make_pair(n=600, beta=1.5, alpha=0.2, phi=0.9, level=50.0, seed=8)


def test_tuning_never_sees_the_test_window():
    # Parameters picked with test-window prices in view would make the test
    # Sharpe in-sample. Scrambling every price after the split must not
    # change the choice.
    data = synthetic()
    train_end = data.index[359]
    scrambled = data.copy()
    rng = np.random.default_rng(0)
    scrambled.iloc[360:] = scrambled.iloc[360:] * rng.uniform(0.5, 1.5, (240, 2))
    for model in ("kalman", "ols"):
        assert tune(data, train_end, model, {}) == tune(scrambled, train_end, model, {})


def test_test_metrics_cover_only_the_test_window():
    data = synthetic()
    ev = evaluate(data, train_end=data.index[359])
    kalman = ev.runs["Kalman"]
    assert ev.test_start == data.index[360] and ev.test_end == data.index[-1]
    assert ev.train_start == data.index[60]  # scoring starts after warm-up
    equity = kalman.result["equity"]
    expected = (equity.iloc[-1] / equity.iloc[359]) ** (252 / 240) - 1
    assert np.isclose(kalman.test["annual_return"], expected)
    assert "Sharpe on the test window" in takeaway(ev)


def test_both_models_use_the_same_costs_and_warmup():
    data = synthetic()
    ev = evaluate(data, train_end=data.index[359], commission_bps=3.0, slippage_bps=2.0)
    for run in ev.runs.values():
        fills = run.result["traded_notional"] > 0
        assert np.allclose(run.result["cost"][fills], 5e-4 * run.result["traded_notional"][fills])
        assert (run.result["position"].iloc[:60] == 0).all()
