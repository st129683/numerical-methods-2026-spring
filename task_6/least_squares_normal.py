import numpy as np


def make_vandermonde(x, n):
    m = len(x)
    E = np.ones((m, n+1))
    for j in range(1, n+1):
        E[:, j] = E[:, j-1] * x
    return E


def solve_normal(x_data, y_data, n):
    E = make_vandermonde(x_data, n)
    lhs = E.T @ E
    rhs = E.T @ y_data
    return np.linalg.solve(lhs, rhs)


def eval_normal(x, coeffs):
    result = np.zeros_like(x)
    for j, c in enumerate(coeffs):
        result += c * (x**j)
    return result
