import numpy as np
from hypothesis import given, settings, strategies as st

from conftest import make_pair
from src.backtester import Backtester


def step_pair():
    # Synthetic pair whose true beta steps from 1.0 to 2.0 at bar 100.
    return make_pair(n=300, beta=1.0, beta_step_at=100, beta_after=2.0,
                     noise=0.01, phi=0.9, level=50.0, seed=1)


def test_bar_pnl_uses_shares_set_on_previous_bar():
    # P&L for bar t must come from holdings fixed at t-1. If the hedge ratio
    # used for bar t's P&L is updated with bar t's own prices, P&L is no
    # longer linear in bar t's price: that is look-ahead.
    data = step_pair()
    kw = dict(entry_threshold=0.5, commission_bps=0.0, slippage_bps=0.0)
    pnl = []
    for bump in (1.0, 1.01, 1.02):
        bumped = data.copy()
        bumped.iloc[100, bumped.columns.get_loc('asset_x')] *= bump
        eq = Backtester(bumped, **kw).run()['equity']
        pnl.append(eq.iloc[100] - eq.iloc[99])
    assert np.isclose(pnl[1] - pnl[0], pnl[2] - pnl[1], rtol=1e-9, atol=1e-6)

    res = Backtester(data, **kw).run()
    assert res['position'].iloc[99] != 0  # precondition: holding into bar 100
    d_y = data['asset_y'].iloc[100] - data['asset_y'].iloc[99]
    d_x = data['asset_x'].iloc[100] - data['asset_x'].iloc[99]
    expected = res['shares_y'].iloc[99] * d_y + res['shares_x'].iloc[99] * d_x
    assert np.isclose(res['equity'].iloc[100] - res['equity'].iloc[99], expected)


def test_no_trades_during_warmup():
    # The filter has no fitted state before the warm-up ends, so any trade
    # there is driven by an arbitrary starting guess.
    data = make_pair(n=200, beta=1.0, phi=0.9, level=50.0, seed=2)
    res = Backtester(data).run()
    assert (res['equity'].iloc[:60] == res['equity'].iloc[0]).all()
    assert (res['position'].iloc[:60] == 0).all()
    assert (res['position'] != 0).any()  # it does trade after warm-up


def test_returns_do_not_depend_on_price_level():
    # P&L must be in dollars of a position sized from equity, not one share.
    # The same pair quoted at $1 and at $100 must give the same equity curve.
    curves = []
    for level in (1.0, 100.0):
        data = make_pair(n=400, beta=1.5, alpha=0.2, phi=0.9, level=level, seed=4)
        curves.append(Backtester(data).run()['equity'].to_numpy())
    assert curves[0][-1] != curves[0][0]  # it trades
    assert np.allclose(curves[0], curves[1], rtol=1e-6)


def test_model_z_score_is_standardized():
    # The entry threshold is in standard deviations, so z must have unit
    # variance on a pair that matches the model.
    variances = []
    for seed in range(5):
        data = make_pair(n=1250, beta=1.5, alpha=0.2, level=100.0, seed=seed)
        z = Backtester(data, warmup=250, delta=1e-6).run()['z_score'].iloc[250:]
        variances.append(z.var())
    assert abs(np.mean(variances) - 1.0) < 0.1


def test_rolling_z_score_is_still_available():
    data = make_pair(n=200, beta=1.0, phi=0.9, level=50.0, seed=2)
    z = Backtester(data, z_method='rolling', z_window=30).run()['z_score']
    assert z.iloc[:89].isna().all() and z.iloc[89:].notna().all()


pair_params = dict(
    seed=st.integers(0, 10_000),
    beta=st.floats(0.3, 3.0),
    phi=st.floats(0.0, 0.95),
    level=st.floats(1.0, 500.0),
    cost_bps=st.floats(0.0, 50.0),
)


@settings(max_examples=40, deadline=None)
@given(**pair_params)
def test_equity_is_cash_plus_holdings(seed, beta, phi, level, cost_bps):
    data = make_pair(n=250, beta=beta, phi=phi, level=level, seed=seed)
    res = Backtester(data, entry_threshold=1.0, commission_bps=cost_bps,
                     slippage_bps=cost_bps).run()
    holdings = res['cash'] + res['shares_x'] * data['asset_x'] + res['shares_y'] * data['asset_y']
    assert np.allclose(res['equity'], holdings, rtol=1e-9)


@settings(max_examples=40, deadline=None)
@given(**pair_params)
def test_doubling_costs_never_raises_final_equity(seed, beta, phi, level, cost_bps):
    data = make_pair(n=250, beta=beta, phi=phi, level=level, seed=seed)
    final = [
        Backtester(data, entry_threshold=1.0, commission_bps=c, slippage_bps=c).run()['equity'].iloc[-1]
        for c in (cost_bps, 2 * cost_bps)
    ]
    assert final[1] <= final[0] * (1 + 1e-12)
