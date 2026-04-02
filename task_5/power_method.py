import numpy as np
from tools import DELTA, RTOL

def power_method(A):
    n = A.shape[0]
    z = np.ones(n)
    z = z / np.linalg.norm(z)
    lambda_prev = 0.0
    while True:
        y = A @ z
        norms = np.linalg.norm(y)
        if norms == 0:
            break
        z_new = y / norms
        mask = np.abs(z) > DELTA
        if not np.any(mask):
            break
        lambdas = y[mask] / z[mask]
        lambda_curr = np.mean(lambdas)
        if np.abs(lambda_curr - lambda_prev) < RTOL * max(np.abs(lambda_curr), np.abs(lambda_prev), 1e-15):
            return lambda_curr, z_new
        z = z_new
        lambda_prev = lambda_curr
    return lambda_prev, z
