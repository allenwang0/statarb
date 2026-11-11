import numpy as np


class KalmanFilterReg:
    """
    A simple Kalman Filter implementation to estimate the dynamic slope (beta)
    and intercept (alpha) of a linear relationship: Y = alpha + beta * X
    """

    def __init__(self, delta=1e-4):
        # delta: System noise scale (how much the "truth" changes per bar)
        # Q = delta / (1 - delta) * I and R (measurement noise) are set in
        # initialize() from an OLS fit on warm-up bars, so they match the
        # price scale of the pair.

        self.n = 2  # Number of states (slope, intercept)
        self.delta = delta

        # State vector [beta, alpha]
        self.state = np.zeros(self.n)

        # Error covariance matrix (uncertainty about the state)
        self.P = np.eye(self.n)

        # System noise matrix
        self.Q = np.eye(self.n) * delta / (1 - delta)

        # Measurement noise variance
        self.R = 1.0

    def initialize(self, x, y):
        """
        Set the state, its covariance and R from an OLS fit of y on x
        over warm-up bars.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        H = np.column_stack([x, np.ones(len(x))])
        coef, _, _, _ = np.linalg.lstsq(H, y, rcond=None)
        resid = y - H @ coef
        self.state = coef
        self.R = np.dot(resid, resid) / (len(x) - self.n)
        self.P = self.R * np.linalg.inv(H.T @ H)
        # Scale the system noise by the per-bar OLS covariance so that
        # delta means the same thing at any price level
        self.Q = self.delta / (1 - self.delta) * self.P * len(x)

    def update(self, x, y):
        """
        Update the state based on new observation (x, y)
        x: Price of Independent Asset (e.g., Coke)
        y: Price of Dependent Asset (e.g., Pepsi)
        """
        # 1. Prediction Step (Random Walk prior: state doesn't change)
        # state_pred = state_prev
        # P_pred = P_prev + Q
        self.P = self.P + self.Q

        # 2. Observation Matrix H = [x, 1]
        H = np.array([x, 1.0])

        # 3. Measurement Residual (Innovation)
        # y_pred = H @ state
        y_pred = np.dot(H, self.state)
        error = y - y_pred

        # 4. Residual Covariance
        S = np.dot(H, np.dot(self.P, H.T)) + self.R

        # 5. Kalman Gain
        K = np.dot(self.P, H.T) / S

        # 6. Update State
        self.state = self.state + K * error

        # 7. Update Error Covariance
        self.P = (np.eye(self.n) - np.outer(K, H)) @ self.P

        # Return the estimated Hedge Ratio (beta), Intercept (alpha),
        # the innovation and its variance
        return self.state[0], self.state[1], error, S
