import pandas as pd
import numpy as np
from src.data import DataError
from src.math_model import KalmanFilterReg

class Backtester:
    def __init__(self, data, entry_threshold=2.0, exit_threshold=0.0,
                 initial_capital=100000.0, gross_notional=1.0,
                 commission_bps=1.0, slippage_bps=1.0, warmup=60,
                 delta=1e-4, z_method='model', z_window=30):
        self.data = data.copy() # Expects columns ['asset_x', 'asset_y']
        self.kf = KalmanFilterReg(delta=delta)
        self.entry_threshold = entry_threshold # Z-score to open trade
        self.exit_threshold = exit_threshold   # Z-score to close trade
        self.warmup = warmup                   # Bars used to fit the filter, no trading
        self.z_method = z_method               # 'model': error / sqrt(S), 'rolling': error / rolling std
        self.z_window = z_window               # Window for the rolling z-score

        # Sizing and costs
        self.initial_capital = initial_capital
        self.gross_notional = gross_notional   # Gross exposure as a fraction of equity
        self.cost_rate = (commission_bps + slippage_bps) / 1e4  # Charged on traded notional


    def run(self):
        x = self.data['asset_x'].to_numpy(dtype=float)
        y = self.data['asset_y'].to_numpy(dtype=float)
        n = len(x)
        if n <= self.warmup:
            raise DataError(f"Need more than {self.warmup} bars of data, got {n}")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise DataError("Prices contain missing or infinite values")
        if (x <= 0).any() or (y <= 0).any():
            raise DataError("Prices must be positive")

        position = 0  # 1 = long spread, -1 = short spread, 0 = flat
        cash = self.initial_capital

        # Warm-up: fit the filter on the first bars, take no positions
        self.kf.initialize(x[:self.warmup], y[:self.warmup])

        # Lists to store history for analysis
        hedge_ratios = np.full(n, np.nan)
        z_scores = np.full(n, np.nan)
        positions = np.zeros(n)
        cash_hist = np.zeros(n)
        equity_hist = np.zeros(n)
        cost_hist = np.zeros(n)
        shares_x_hist = np.zeros(n)
        shares_y_hist = np.zeros(n)
        spreads = []
        shares_x = 0.0
        shares_y = 0.0

        for t in range(n):
            x_price = x[t]
            y_price = y[t]

            if t >= self.warmup:
                # 1. Update Math Model with bar t (P&L for bar t is already
                # carried by the shares held since t-1)
                beta, alpha, spread_error, S = self.kf.update(x_price, y_price)

                if self.z_method == 'rolling':
                    spreads.append(spread_error)
                    if len(spreads) < self.z_window:
                        z_score = np.nan # Not enough history yet
                    else:
                        std_spread = np.std(spreads[-self.z_window:])
                        z_score = spread_error / std_spread if std_spread > 0 else 0
                else:
                    z_score = spread_error / np.sqrt(S)

                # 2. Generate Signal on the close of bar t
                new_position = position

                # Mean Reversion Logic
                if position == 0:
                    if z_score > self.entry_threshold:
                        new_position = -1 # Sell the spread (Short Y, Long X)
                    elif z_score < -self.entry_threshold:
                        new_position = 1  # Buy the spread (Long Y, Short X)

                elif position == 1: # Long Spread
                    if z_score >= -self.exit_threshold:
                        new_position = 0 # Exit

                elif position == -1: # Short Spread
                    if z_score <= self.exit_threshold:
                        new_position = 0 # Exit

                # 3. Execute at the close: 1 share of Y against -beta shares of X
                if new_position != position:
                    if new_position == 0:
                        target_y = 0.0
                        target_x = 0.0
                    else:
                        equity = cash + shares_x * x_price + shares_y * y_price
                        if equity <= 0:
                            # Sizing from non-positive equity would flip or
                            # zero the trade; a bankrupt account stays flat.
                            new_position = 0
                        units = self.gross_notional * equity / (y_price + abs(beta) * x_price)
                        target_y = new_position * units
                        target_x = -beta * target_y
                    trade_y = target_y - shares_y
                    trade_x = target_x - shares_x
                    cost = self.cost_rate * (abs(trade_y) * y_price + abs(trade_x) * x_price)
                    cash -= trade_y * y_price + trade_x * x_price + cost
                    cost_hist[t] = cost
                    shares_x = target_x
                    shares_y = target_y
                    position = new_position

                hedge_ratios[t] = beta
                z_scores[t] = z_score

            # 4. Mark to Market
            equity_hist[t] = cash + shares_x * x_price + shares_y * y_price
            positions[t] = position
            cash_hist[t] = cash
            shares_x_hist[t] = shares_x
            shares_y_hist[t] = shares_y

        return pd.DataFrame({
            'equity': equity_hist,
            'hedge_ratio': hedge_ratios,
            'z_score': z_scores,
            'position': positions,
            'cash': cash_hist,
            'shares_x': shares_x_hist,
            'shares_y': shares_y_hist,
            'cost': cost_hist
        }, index=self.data.index)
