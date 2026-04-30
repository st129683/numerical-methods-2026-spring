import numpy as np


def solve_orthogonal(x_data, y_data, n):
    m = len(x_data)
    q_vals = np.zeros((n+1, m))
    q_vals[0, :] = 1.0
    alpha = np.zeros(n+1)
    beta = np.zeros(n+1)
    alpha[1] = np.sum(x_data) / m
    q_vals[1, :] = x_data - alpha[1]
    for j in range(1, n):
        alpha[j+1] = np.sum(x_data * q_vals[j, :]**2) / np.sum(q_vals[j, :]**2)
        beta[j] = np.sum(x_data * q_vals[j, :] * q_vals[j-1, :]) / np.sum(q_vals[j-1, :]**2)
        q_vals[j+1, :] = x_data * q_vals[j, :] - alpha[j+1] * q_vals[j, :] - beta[j] * q_vals[j-1, :]
    a = np.zeros(n+1)
    for k in range(n+1):
        a[k] = np.sum(y_data * q_vals[k, :]) / np.sum(q_vals[k, :]**2)
    return a, alpha, beta


def eval_orthogonal(x, coeffs, alpha, beta, n):
    q_eval = np.zeros((n+1, len(x)))
    q_eval[0, :] = 1.0
    if n >= 1:
        q_eval[1, :] = x - alpha[1]
    for j in range(1, n):
        q_eval[j+1, :] = x * q_eval[j, :] - alpha[j+1] * q_eval[j, :] - beta[j] * q_eval[j-1, :]
    res = np.zeros(len(x))
    for k in range(n+1):
        res += coeffs[k] * q_eval[k, :]
    return res
