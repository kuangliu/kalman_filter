import random
import numpy as np
from kalman_filter import LinearKalmanFilter


class Simulator:
    def __init__(self, true_value):
        self.true_value = true_value

    def get_true_value(self):
        return self.true_value

    def get_noisy_value(self):
        return random.gauss(self.true_value, 2)


def main():
    A = np.matrix([1])
    B = np.matrix([0])
    C = np.matrix([1])
    P = np.matrix([1])
    Q = np.matrix([0.00001])
    R = np.matrix([0.1])
    x = np.matrix([3])

    kf = LinearKalmanFilter(A, B, C, P, Q, R, x)
    sim = Simulator(1)

    for i in range(100):
        y = sim.get_noisy_value()
        x = kf.step(np.matrix([0]), y)
        print(i, y, x)


if __name__ == '__main__':
    main()
