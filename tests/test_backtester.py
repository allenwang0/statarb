import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from conftest import make_pair
from src.backtester import Backtester
from src.data import DataError


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
    # First z needs 30 innovations before it: bars 60..89, so bar 90.
    assert z.iloc[:90].isna().all() and z.iloc[90:].notna().all()


def test_rolling_z_does_not_include_its_own_bar():
    # A z-score that puts bar t's innovation in its own std can never exceed
    # (w-1)/sqrt(w), about 5.3 for w=30, however large the shock.
    data = make_pair(n=200, beta=1.0, noise=0.01, level=50.0, seed=2)
    data.iloc[150, 1] += 50.0 * 0.5  # about 50 noise sd
    z = Backtester(data, z_method='rolling', z_window=30).run()['z_score']
    assert z.iloc[150] > 10


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


def test_run_twice_gives_the_same_result():
    # run() used to keep cash, position and the equity list on the instance,
    # so a second call started from the first call's end state and crashed.
    data = make_pair(n=300, beta=1.0, phi=0.9, level=50.0, seed=1)
    bt = Backtester(data)
    first, second = bt.run(), bt.run()
    assert first.equals(second)


@settings(max_examples=40, deadline=None)
@given(**pair_params)
def test_each_bar_is_self_financing(seed, beta, phi, level, cost_bps):
    # The only ways equity can change on bar t: price moves on the shares
    # held since t-1, and costs paid on bar t's fills. A sign error in cash
    # or a cost charged twice breaks this; equity == cash + holdings alone
    # cannot catch either because equity is defined that way.
    data = make_pair(n=250, beta=beta, phi=phi, level=level, seed=seed)
    res = Backtester(data, entry_threshold=1.0, commission_bps=cost_bps,
                     slippage_bps=cost_bps).run()
    prev = res.shift(1).iloc[1:]
    d_px = data.diff().iloc[1:]
    expected = (prev['shares_y'] * d_px['asset_y'] + prev['shares_x'] * d_px['asset_x']
                - res['cost'].iloc[1:])
    assert np.allclose(res['equity'].diff().iloc[1:], expected, rtol=1e-9, atol=1e-6)
    fills = res['position'].diff().fillna(0) != 0
    if cost_bps >= 0.01:  # denormal bps round the cost to zero
        assert (res['cost'][fills] > 0).all() and (res['cost'][~fills] == 0).all()


@pytest.mark.parametrize('bad', [np.nan, np.inf, 0.0, -5.0])
def test_bad_prices_are_rejected(bad):
    # One NaN used to turn the filter state into NaN for every later bar and
    # freeze equity without an error.
    data = make_pair(n=200, beta=1.0, phi=0.9, level=50.0, seed=2)
    data.iloc[100, 1] = bad
    with pytest.raises(DataError):
        Backtester(data).run()


def test_flat_x_during_warmup_is_rejected():
    # With X constant the warm-up regression cannot separate beta from
    # alpha; the backtest used to run and silently never trade.
    data = make_pair(n=200, beta=1.0, phi=0.9, level=50.0, seed=2)
    data.iloc[:60, 0] = 50.0
    with pytest.raises(DataError, match="does not move"):
        Backtester(data).run()


def test_exact_linear_pair_is_rejected():
    # Zero residual means R ~ 1e-28 and z = error / sqrt(S) is rounding noise.
    data = make_pair(n=200, beta=1.0, phi=0.9, level=50.0, seed=2)
    data['asset_y'] = 2.0 * data['asset_x']
    with pytest.raises(DataError, match="exact linear"):
        Backtester(data).run()


def test_bankrupt_account_does_not_reopen():
    # Sizing is gross_notional * equity / price; with equity <= 0 that flips
    # the side of the trade. A wiped-out account must stay flat.
    data = make_pair(n=300, beta=1.0, phi=0.9, level=50.0, seed=1)
    res = Backtester(data, initial_capital=-1.0, entry_threshold=0.5).run()
    assert (res['position'] == 0).all()



def test_ols_baseline_goes_through_the_same_rules():
    # The comparison is only fair if the baseline uses the same warm-up,
    # signal, sizing and cost code. Swapping the model must change betas
    # but keep the accounting.
    data = make_pair(n=400, beta=1.5, alpha=0.2, phi=0.9, level=50.0, seed=3)
    ols = Backtester(data, model="ols", commission_bps=5.0).run()
    assert (ols["position"].iloc[:60] == 0).all() and (ols["position"] != 0).any()
    prev = ols.shift(1).iloc[1:]
    d_px = data.diff().iloc[1:]
    expected = (prev["shares_y"] * d_px["asset_y"] + prev["shares_x"] * d_px["asset_x"]
                - ols["cost"].iloc[1:])
    assert np.allclose(ols["equity"].diff().iloc[1:], expected)
    entries = ols[(ols["position"].diff() != 0) & (ols["position"] != 0)]
    px = data.loc[entries.index]
    gross = entries["shares_y"].abs() * px["asset_y"] + entries["shares_x"].abs() * px["asset_x"]
    # Entries come from flat, so equity before the fill is the previous bar's.
    assert np.allclose(gross, ols["equity"].shift(1)[entries.index])
