import pandas as pd
import numpy as np
from src.math_model import KalmanFilterReg

class Backtester:
    def __init__(self, data, entry_threshold=2.0, exit_threshold=0.0):
        self.data = data.copy() # Expects columns ['asset_x', 'asset_y']
        self.kf = KalmanFilterReg()
        self.entry_threshold = entry_threshold # Z-score to open trade
        self.exit_threshold = exit_threshold   # Z-score to close trade

        # Portfolio State
        self.position = 0 # 1 = Long Spread, -1 = Short Spread, 0 = Flat
        self.cash = 100000.0
        self.equity_curve = []

    def run(self):
        # Lists to store history for analysis
        hedge_ratios = []
        spreads = []
        z_scores = []

        # Warmup period for Kalman Filter (burn first 20 days)
        # Event-Driven Loop
        for t, row in self.data.iterrows():
            x_price = row['asset_x']
            y_price = row['asset_y']

            # 1. Update Math Model
            beta, alpha, spread_error = self.kf.update(x_price, y_price)

            # Calculate standard deviation of the spread (simple rolling window)
            spreads.append(spread_error)
            if len(spreads) < 30:
                std_spread = 1.0 # Avoid division by zero during warmup
            else:
                std_spread = np.std(spreads[-30:])

            z_score = spread_error / std_spread if std_spread > 0 else 0

            # 2. Generate Signal & Execute
            # Note: We use yesterday's state to trade today to be conservative,
            # or trade on the close (as shown here).

            prev_position = self.position

            # Mean Reversion Logic
            if self.position == 0:
                if z_score > self.entry_threshold:
                    self.position = -1 # Sell the spread (Short Y, Long X)
                elif z_score < -self.entry_threshold:
                    self.position = 1  # Buy the spread (Long Y, Short X)

            elif self.position == 1: # Long Spread
                if z_score >= -self.exit_threshold:
                    self.position = 0 # Exit

            elif self.position == -1: # Short Spread
                if z_score <= self.exit_threshold:
                    self.position = 0 # Exit

            # 3. Mark to Market (simplified PnL calculation)
            # In a real engine, you track specific share counts.
            # Here we approximate roughly: PnL += Position * (Change in Spread)
            # This is a simplification for the MVP.

            if len(self.equity_curve) > 0:
                # Price change vector
                px_change_y = y_price - self.data.iloc[self.data.index.get_loc(t)-1]['asset_y']
                px_change_x = x_price - self.data.iloc[self.data.index.get_loc(t)-1]['asset_x']

                # PnL = QtyY * dY + QtyX * dX
                # Hedge Ratio dictates: 1 unit of Y against 'beta' units of X
                daily_pnl = prev_position * (px_change_y - beta * px_change_x)
                curr_equity = self.equity_curve[-1] + daily_pnl
            else:
                curr_equity = 100000.0

            self.equity_curve.append(curr_equity)
            hedge_ratios.append(beta)
            z_scores.append(z_score)

        return pd.DataFrame({
            'equity': self.equity_curve,
            'hedge_ratio': hedge_ratios,
            'z_score': z_scores
        }, index=self.data.index)
