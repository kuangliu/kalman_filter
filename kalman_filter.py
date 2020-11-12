'''Linear Kalman-Filter.

Reference: https://www.youtube.com/watch?v=VFXf1lIZ3p8

Prediction:
 - x = Ax + Bu
 - P = APA^T + Q

Update:
 - K = (PC^T) / (CPC^T + R)
 - x = x + K(y-Cx)
 - P = (I-KC)P
'''

import numpy as np


class LinearKalmanFilter:
    def __init__(self, A, B, C, P, Q, R, x):
        self.A = A
        self.B = B
        self.C = C
        self.P = P
        self.Q = Q
        self.R = R
        self.x = x

    def step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.x

    def predict(self, u):
        self.x = self.A * self.x + self.B * u
        self.P = self.A * self.P * self.A.T + self.Q

    def update(self, y):
        tmp = self.P * self.C.T
        self.K = tmp * np.linalg.inv(self.C * tmp + self.R)
        self.x = self.x + self.K * (y - self.C * self.x)
        self.P = (np.eye(self.x.shape[0]) - self.K * self.C) * self.P
