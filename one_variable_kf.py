import random
import numpy as np
from kalman_filter import LinearKalmanFilter


def main():
    A = np.eye(4)
    B = np.zeros_like(A)
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    P = np.eye(4)
    Q = np.eye(4) * 1e-5
    R = np.eye(2) * 1e-1
    x = np.array([[0], [0], [0], [0]])
    kf = LinearKalmanFilter(A, B, C, P, Q, R, x)

    x0 = 10
    y0 = 10
    vx = 5
    vy = 2
    d1 = 0
    d2 = 0
    for t in range(100):
        xt = x0 + vx * t
        yt = y0 + vy * t
        xt_noisy = random.gauss(xt, 10)
        yt_noisy = random.gauss(yt, 10)

        kf.A[0, 2] = 1
        kf.A[1, 3] = 1
        u = np.array([[0], [0], [0], [0]])
        y = np.array([[xt_noisy], [yt_noisy]])
        out = kf.step(u, y)
        print(xt, yt, int(xt_noisy), int(yt_noisy),
              int(out[0][0]), int(out[1][0]))

        d1 += abs(xt - xt_noisy) + abs(yt-yt_noisy)
        d2 += abs(xt - out[0][0]) + abs(yt-out[1][0])
    print(d1)
    print(d2)


if __name__ == '__main__':
    main()
