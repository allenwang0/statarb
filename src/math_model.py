import numpy as np


class KalmanFilterReg:
    """
    A simple Kalman Filter implementation to estimate the dynamic slope (beta)
    and intercept (alpha) of a linear relationship: Y = alpha + beta * X
    """

    def __init__(self, delta=1e-5, R=1e-3):
        # delta: System noise covariance (how much the "truth" changes)
        # R: Measurement noise covariance (how noisy the data is)

        self.n = 2  # Number of states (slope, intercept)

        # State vector [beta, alpha]
        self.state = np.zeros(self.n)

        # Error covariance matrix (uncertainty about the state)
        self.P = np.eye(self.n)

        # System noise matrix
        self.Q = np.eye(self.n) * delta

        # Measurement noise variance
        self.R = R

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

        # Return the estimated Hedge Ratio (beta) and Intercept (alpha)
        return self.state[0], self.state[1], error
