import numpy as np
from tools import generate_equidistant_nodes, generate_optimal_nodes, evaluate_function


def lagrange_mult(x, i, nodes):
    result = 1.0
    for j, node in enumerate(nodes):
        if j != i:
            result *= (x - node) / (nodes[i] - node)
    return result


def lagrange_polynomial(x, nodes, values):
    return sum(values[i] * lagrange_mult(x, i, nodes) for i in range(len(nodes)))


def lagrange_polynomial_vector(nodes, values, x_values):
    return np.array([lagrange_polynomial(x, nodes, values) for x in x_values])
