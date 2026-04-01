import numpy as np


def generate_equidistant_nodes(a, b, n):
    return np.linspace(a, b, n)


def generate_optimal_nodes(a, b, n):
    i = np.arange(n)
    nodes = ((b - a) * np.cos((2 * i + 1) * np.pi / (2 * n + 2)) + b + a) / 2
    return np.sort(nodes)


def evaluate_function(f, nodes):
    return np.array([f(x) for x in nodes])


def calc_max_deviation(f, interp_values, test_points):
    f_values = np.array([f(x) for x in test_points])
    return np.max(np.abs(f_values - interp_values))

def common_deviation(f, interp_values, test_points):
    f_values = np.array([f(x) for x in test_points])
    return np.abs(f_values - interp_values)


def f(x):
    return x**2 + 1 - np.arccos(x)
