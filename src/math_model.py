from collections import deque

import numpy as np

from src.data import DataError

# Residual variance below this fraction of var(y) is float rounding, not
# noise: Y is an exact linear function of X (for example the same ticker
# twice) and z = error / sqrt(S) would divide by roughly zero.
MIN_RELATIVE_RESIDUAL_VAR = 1e-12


def ols_fit(x, y):
    """
    OLS of y on [x, 1]. Returns (coef [beta, alpha], residual variance,
    inverse of H'H). Raises DataError when the fit is degenerate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) <= 2:
        raise DataError(f"Regression needs more than 2 bars, got {len(x)}")
    if np.ptp(x) == 0:
        raise DataError("X does not move during the warm-up, so beta is not identified")
    H = np.column_stack([x, np.ones(len(x))])
    coef, _, _, _ = np.linalg.lstsq(H, y, rcond=None)
    resid = y - H @ coef
    resid_var = float(np.dot(resid, resid) / (len(x) - 2))
    if not resid_var > MIN_RELATIVE_RESIDUAL_VAR * np.var(y):
        raise DataError(
            "Y is an exact linear function of X during the warm-up; there is no spread to trade"
        )
    return coef, resid_var, np.linalg.inv(H.T @ H)


class KalmanFilterReg:
    """
    Kalman filter for Y_t = alpha_t + beta_t * X_t + e_t, e_t ~ N(0, R),
    with state [beta_t, alpha_t] following a random walk with covariance Q.
    """

    def __init__(self, delta=1e-4):
        self.n = 2
        self.delta = delta
        self.state = np.zeros(self.n)
        self.P = np.eye(self.n)
        self.Q = np.eye(self.n) * delta / (1 - delta)
        self.R = 1.0

    def initialize(self, x, y):
        """
        Set the state, P and R from an OLS fit of y on x over warm-up bars.
        Q = delta / (1 - delta) * (R * (H'H / n)^-1): the per-bar OLS
        covariance, so delta means the same thing at any price level.
        """
        coef, resid_var, hth_inv = ols_fit(x, y)
        self.state = coef
        self.R = resid_var
        self.P = self.R * hth_inv
        self.Q = self.delta / (1 - self.delta) * self.P * len(x)

    @property
    def beta_sd(self):
        return float(np.sqrt(self.P[0, 0]))

    def update(self, x, y):
        """
        One predict/update step with bar (x, y). Returns beta and alpha after
        the update, the one-step-ahead innovation y - (alpha + beta x) from the
        prior state, and its variance S.
        """
        self.P = self.P + self.Q  # predict: state_t|t-1 = state_t-1
        H = np.array([x, 1.0])
        error = y - H @ self.state
        S = H @ self.P @ H + self.R
        K = self.P @ H / S
        self.state = self.state + K * error
        self.P = (np.eye(self.n) - np.outer(K, H)) @ self.P
        return self.state[0], self.state[1], error, S


class RollingOLS:
    """
    Baseline with the same interface as KalmanFilterReg: OLS of y on x over
    the last `window` bars, refit every bar.
    """

    def __init__(self, window=60):
        self.window = window
        self.xs = deque(maxlen=window)
        self.ys = deque(maxlen=window)
        self.state = np.zeros(2)
        self.resid_var = 1.0
        self.hth_inv = np.eye(2)

    def _fit(self):
        self.state, self.resid_var, self.hth_inv = ols_fit(self.xs, self.ys)

    def initialize(self, x, y):
        self.xs.extend(np.asarray(x, dtype=float)[-self.window :])
        self.ys.extend(np.asarray(y, dtype=float)[-self.window :])
        self._fit()

    @property
    def beta_sd(self):
        return float(np.sqrt(self.resid_var * self.hth_inv[0, 0]))

    def update(self, x, y):
        """
        The innovation and S come from the fit on the previous `window` bars
        (S = s^2 (1 + h (H'H)^-1 h'), the OLS prediction variance); the
        returned beta and alpha include bar t.
        """
        H = np.array([x, 1.0])
        error = y - H @ self.state
        S = self.resid_var * (1.0 + H @ self.hth_inv @ H)
        self.xs.append(float(x))
        self.ys.append(float(y))
        self._fit()
        return self.state[0], self.state[1], error, S
