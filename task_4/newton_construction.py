import numpy as np
from tools import generate_equidistant_nodes, generate_optimal_nodes, evaluate_function


def divided_differences(nodes, values):
    n = len(nodes)
    table = np.zeros((n, n))
    table[:, 0] = values
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (nodes[i + j] - nodes[i])
    return table[0, :]


def newton_mult(x, nodes, k):
    result = 1.0
    for i in range(k):
        result *= (x - nodes[i])
    return result


def newton_polynomial(x, nodes, div_diff):
    return sum(div_diff[k] * newton_mult(x, nodes, k) for k in range(len(div_diff)))


def newton_polynomial_vector(nodes, div_diff, x_values):
    return np.array([newton_polynomial(x, nodes, div_diff) for x in x_values])
