import numpy as np
import pandas as pd

from src.data import DataError
from src.math_model import KalmanFilterReg, RollingOLS

BPS = 1e4
DEFAULT_WARMUP = 60  # bars, about three months of daily closes
DEFAULT_Z_WINDOW = 30  # bars for the rolling-std z-score


class Backtester:
    """
    Trades one pair on daily closes. Each bar, in order: book P&L on the
    shares held since t-1, update the hedge model with bar t, decide the new
    position from z_t and fill at bar t's close.

    fill_lag: bars between the decision and the fill. 0 (the default) fills
    at the close that produced the signal; 1 fills at the next bar's close,
    sized with the hedge ratio known at decision time.

    data: DataFrame with columns ['asset_x', 'asset_y'].
    model: 'kalman' (KalmanFilterReg) or 'ols' (RollingOLS over ols_window).
    z_method: 'model' uses innovation / sqrt(S); 'rolling' uses innovation /
    std of the previous z_window innovations.
    """

    def __init__(
        self,
        data,
        entry_threshold=2.0,
        exit_threshold=0.0,
        initial_capital=100000.0,
        gross_notional=1.0,
        commission_bps=1.0,
        slippage_bps=1.0,
        warmup=DEFAULT_WARMUP,
        delta=1e-4,
        z_method="model",
        z_window=DEFAULT_Z_WINDOW,
        model="kalman",
        ols_window=DEFAULT_WARMUP,
        fill_lag=0,
    ):
        if z_method not in ("model", "rolling"):
            raise ValueError(f"z_method must be 'model' or 'rolling', got {z_method!r}")
        if model not in ("kalman", "ols"):
            raise ValueError(f"model must be 'kalman' or 'ols', got {model!r}")
        if model == "ols" and ols_window > warmup:
            raise ValueError("ols_window cannot exceed warmup")
        if fill_lag < 0:
            raise ValueError(f"fill_lag must be 0 or more, got {fill_lag}")
        self.data = data.copy()
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.warmup = warmup
        self.delta = delta
        self.z_method = z_method
        self.z_window = z_window
        self.model = model
        self.ols_window = ols_window
        self.fill_lag = int(fill_lag)
        self.initial_capital = initial_capital
        self.gross_notional = gross_notional  # gross exposure as a fraction of equity
        self.cost_rate = (commission_bps + slippage_bps) / BPS  # on traded notional

    def _hedge_model(self):
        if self.model == "ols":
            return RollingOLS(window=self.ols_window)
        return KalmanFilterReg(delta=self.delta)

    def _next_position(self, position, z):
        """1 = long spread (long Y, short X), -1 = short spread, 0 = flat."""
        if position == 0:
            if z > self.entry_threshold:
                return -1
            if z < -self.entry_threshold:
                return 1
        elif position == 1 and z >= -self.exit_threshold:
            return 0
        elif position == -1 and z <= self.exit_threshold:
            return 0
        return position

    def run(self):
        x = self.data["asset_x"].to_numpy(dtype=float)
        y = self.data["asset_y"].to_numpy(dtype=float)
        n = len(x)
        if n <= self.warmup:
            raise DataError(f"Need more than {self.warmup} bars of data, got {n}")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise DataError("Prices contain missing or infinite values")
        if (x <= 0).any() or (y <= 0).any():
            raise DataError("Prices must be positive")

        hedge = self._hedge_model()
        hedge.initialize(x[: self.warmup], y[: self.warmup])

        cols = [
            "equity", "hedge_ratio", "beta_sd", "alpha", "innovation", "z_score",
            "position", "cash", "shares_x", "shares_y", "traded_notional", "cost",
        ]  # fmt: skip
        out = {c: np.zeros(n) for c in cols}
        for c in ("hedge_ratio", "beta_sd", "alpha", "innovation", "z_score"):
            out[c][:] = np.nan

        position = 0  # held
        decided = 0  # latest decision; equals position when fill_lag is 0
        pending: list[tuple[int, int, float]] = []  # (fill bar, position, beta at decision)
        cash = float(self.initial_capital)
        shares_x = shares_y = 0.0
        innovations: list[float] = []

        for t in range(n):
            px_x, px_y = x[t], y[t]
            if t >= self.warmup:
                beta, alpha, error, S = hedge.update(px_x, px_y)
                if self.z_method == "rolling":
                    # Previous z_window innovations only; including bar t
                    # caps |z| at (w-1)/sqrt(w).
                    if len(innovations) < self.z_window:
                        z = np.nan
                    else:
                        sd = np.std(innovations[-self.z_window :])
                        z = error / sd if sd > 0 else 0.0
                    innovations.append(error)
                else:
                    z = error / np.sqrt(S)

                new_position = self._next_position(decided, z)
                if new_position != decided:
                    decided = new_position
                    pending.append((t + self.fill_lag, new_position, beta))

                while pending and pending[0][0] == t:
                    _, new_position, fill_beta = pending.pop(0)
                    target_x = target_y = 0.0
                    equity = cash + shares_x * px_x + shares_y * px_y
                    # A bankrupt account stays flat: sizing from equity <= 0
                    # would flip the trade.
                    if new_position != 0 and equity > 0:
                        units = self.gross_notional * equity / (px_y + abs(fill_beta) * px_x)
                        target_y = new_position * units
                        target_x = -fill_beta * target_y
                    elif new_position != 0:
                        new_position = 0
                    notional = abs(target_y - shares_y) * px_y + abs(target_x - shares_x) * px_x
                    cost = self.cost_rate * notional
                    cash -= (target_y - shares_y) * px_y + (target_x - shares_x) * px_x + cost
                    out["traded_notional"][t] = notional
                    out["cost"][t] = cost
                    shares_x, shares_y = target_x, target_y
                    position = new_position
                    if not pending:
                        decided = position  # a bankrupt account stays flat

                out["hedge_ratio"][t] = beta
                out["beta_sd"][t] = hedge.beta_sd
                out["alpha"][t] = alpha
                out["innovation"][t] = error
                out["z_score"][t] = z

            out["equity"][t] = cash + shares_x * px_x + shares_y * px_y
            out["position"][t] = position
            out["cash"][t] = cash
            out["shares_x"][t] = shares_x
            out["shares_y"][t] = shares_y

        return pd.DataFrame(out, index=self.data.index)
