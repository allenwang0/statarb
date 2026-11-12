import numpy as np
import pandas as pd


def make_pair(
    n=500,
    beta=1.0,
    alpha=0.0,
    noise=0.01,
    phi=0.0,
    level=1.0,
    beta_step_at=None,
    beta_after=None,
    seed=0,
):
    """Synthetic (not market) pair: Y = level*alpha + beta*X + level*eps.

    X is a geometric random walk starting at `level`. eps is AR(1) with
    coefficient `phi` and innovation sd `noise` (phi=0 gives white noise).
    If `beta_step_at` is set, beta switches to `beta_after` from that bar on.
    """
    rng = np.random.default_rng(seed)
    x = level * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
    eps = np.zeros(n)
    shocks = noise * rng.standard_normal(n)
    for t in range(n):
        eps[t] = (phi * eps[t - 1] if t > 0 else 0.0) + shocks[t]
    betas = np.full(n, beta, dtype=float)
    if beta_step_at is not None:
        betas[beta_step_at:] = beta_after
    y = level * alpha + betas * x + level * eps
    index = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"asset_x": x, "asset_y": y}, index=index)
