import numpy as np

from conftest import make_pair
from src.math_model import KalmanFilterReg, RollingOLS


def run_filter(data, warmup, delta):
    x, y = data['asset_x'].values, data['asset_y'].values
    kf = KalmanFilterReg(delta=delta)
    kf.initialize(x[:warmup], y[:warmup])
    out = [kf.update(x[t], y[t]) for t in range(warmup, len(x))]
    beta, alpha, error, S = (np.array(v) for v in zip(*out))
    return beta, alpha, error, S


def test_initialize_starts_from_warmup_ols_fit():
    # Without a warm-up fit the filter starts at beta=0 and its first
    # innovation is the whole price of Y, which fires a trade on bar 1.
    data = make_pair(n=60, beta=1.5, alpha=0.2, level=50.0, seed=3)
    kf = KalmanFilterReg()
    kf.initialize(data['asset_x'].values, data['asset_y'].values)
    assert abs(kf.state[0] - 1.5) < 0.1
    resid_var = (50.0 * 0.01) ** 2
    assert 0.5 * resid_var < kf.R < 2.0 * resid_var


def test_standardized_innovations_have_unit_variance():
    # z = error / sqrt(S) is only a z-score if S is the model's real
    # innovation variance. Constant-beta pair with white noise of known sd.
    variances = []
    for seed in range(20):
        data = make_pair(n=1250, beta=1.5, alpha=0.2, noise=0.01, level=100.0, seed=seed)
        _, _, error, S = run_filter(data, warmup=250, delta=1e-6)
        variances.append(np.var(error / np.sqrt(S)))
    assert abs(np.mean(variances) - 1.0) < 0.1


def test_beta_error_does_not_depend_on_price_level():
    # The same pair quoted at $1 and at $100 must give the same hedge ratio.
    # Absolute noise settings make the filter chase noise at one level and
    # freeze at the other.
    errors = {}
    for level in (1.0, 100.0):
        data = make_pair(n=1000, beta=1.5, alpha=0.2, noise=0.01, level=level, seed=7)
        beta, _, _, _ = run_filter(data, warmup=60, delta=1e-4)
        errors[level] = np.sqrt(np.mean((beta - 1.5) ** 2))
    assert np.isclose(errors[1.0], errors[100.0], rtol=1e-3)


def test_rolling_ols_innovation_is_out_of_sample():
    # The baseline must be held to the same timing as the filter: bar t's
    # innovation comes from a fit on bars t-60..t-1, never bar t itself.
    data = make_pair(n=200, beta=1.5, alpha=0.2, phi=0.9, level=50.0, seed=5)
    x, y = data["asset_x"].to_numpy(), data["asset_y"].to_numpy()
    ols = RollingOLS(window=60)
    ols.initialize(x[:60], y[:60])
    for t in range(60, 120):
        beta_prev, alpha_prev = np.polyfit(x[t - 60 : t], y[t - 60 : t], 1)
        beta, alpha, error, _ = ols.update(x[t], y[t])
        assert np.isclose(error, y[t] - (alpha_prev + beta_prev * x[t]))
    assert np.allclose([beta, alpha], np.polyfit(x[60:120], y[60:120], 1))


def test_rolling_ols_standardized_innovations_have_unit_variance():
    # S = s^2 (1 + h (H'H)^-1 h') is the OLS prediction variance; with s^2
    # estimated on 58 degrees of freedom the ratio is t-distributed, variance
    # 58/56, so about 1.04.
    variances = []
    for seed in range(10):
        data = make_pair(n=1000, beta=1.5, alpha=0.2, level=100.0, seed=seed)
        x, y = data["asset_x"].to_numpy(), data["asset_y"].to_numpy()
        ols = RollingOLS(window=60)
        ols.initialize(x[:60], y[:60])
        z = [e / np.sqrt(S) for _, _, e, S in (ols.update(x[t], y[t]) for t in range(60, 1000))]
        variances.append(np.var(z))
    assert abs(np.mean(variances) - 58 / 56) < 0.1


def test_filter_with_no_state_noise_is_expanding_ols():
    # With Q = 0 the Kalman filter is recursive least squares: after bar t
    # its state equals OLS on every bar so far. This pins the update
    # equations, not just their statistical behaviour.
    data = make_pair(n=150, beta=1.5, alpha=0.2, phi=0.9, level=50.0, seed=6)
    x, y = data["asset_x"].to_numpy(), data["asset_y"].to_numpy()
    kf = KalmanFilterReg(delta=0.0)
    kf.initialize(x[:60], y[:60])
    for t in range(60, 150):
        beta, alpha, _, _ = kf.update(x[t], y[t])
    assert np.allclose([beta, alpha], np.polyfit(x, y, 1), rtol=1e-8)
