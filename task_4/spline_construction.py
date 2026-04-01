import numpy as np
from tools import generate_equidistant_nodes, evaluate_function


def linear_spline_coefficients(nodes, values):
    n = len(nodes) - 1
    coeffs = []
    for i in range(n):
        h = nodes[i + 1] - nodes[i]
        a1 = (values[i + 1] - values[i]) / h
        a0 = values[i] - a1 * nodes[i]
        coeffs.append([a0, a1])
    return coeffs


def quadratic_spline_coefficients(nodes, values):
    n = len(nodes) - 1
    # система уравнений для коэффициентов
    # 3n коэффициентов, 2n условий интерполяции + (n-1) условий непрерывности производной + 1 граничное
    A = np.zeros((3 * n, 3 * n))
    b = np.zeros(3 * n)

    for i in range(n):
        A[i, 3 * i] = 1
        A[i, 3 * i + 1] = nodes[i]
        A[i, 3 * i + 2] = nodes[i] ** 2
        b[i] = values[i]

        A[n + i, 3 * i] = 1
        A[n + i, 3 * i + 1] = nodes[i + 1]
        A[n + i, 3 * i + 2] = nodes[i + 1] ** 2
        b[n + i] = values[i + 1]

    # непрерывность первой производной в узлах
    for i in range(n - 1):
        A[2 * n + i, 3 * i + 1] = 1
        A[2 * n + i, 3 * i + 2] = 2 * nodes[i + 1]
        A[2 * n + i, 3 * (i + 1) + 1] = -1
        A[2 * n + i, 3 * (i + 1) + 2] = -2 * nodes[i + 1]
        b[2 * n + i] = 0

    # граничное условие: S''(x[1]) = 0 (естественный сплайн)
    A[3 * n - 1, 3 * 0 + 2] = 2
    b[3 * n - 1] = 0

    # решаем системку
    all_coeffs = np.linalg.solve(A, b)

    # складываем в одномерный массивчик по тройкам
    coeffs = []
    for i in range(n):
        coeffs.append([all_coeffs[3 * i], all_coeffs[3 * i + 1], all_coeffs[3 * i + 2]])

    return coeffs


def cubic_spline_coefficients(nodes, values):
    n = len(nodes) - 1
    h = np.diff(nodes)

    # построение трёхдиагональной системы для вторых производных
    A = np.zeros((n - 1, n - 1))
    gamma = np.zeros(n - 1)

    for i in range(n - 1):
        if i > 0:
            A[i, i - 1] = h[i]
        A[i, i] = 2 * (h[i] + h[i + 1])
        if i < n - 2:
            A[i, i + 1] = h[i + 1]

        gamma[i] = 6 * ((values[i + 2] - values[i + 1]) / h[i + 1] - 
                        (values[i + 1] - values[i]) / h[i])
    
    # hешение системы Hy = gamma
    y_second = np.zeros(n + 1)
    y_second[1:-1] = np.linalg.solve(A, gamma)
    # граничные условия: y''_1 = y''_n = 0 (естественный сплайн)

    # вычисление первых производных в узлах
    y_first = np.zeros(n)
    for i in range(n):
        y_first[i] = (values[i + 1] - values[i]) / h[i] - h[i] / 6 * (y_second[i + 1] + 2 * y_second[i])

    # коэффы для каждого интервала
    coeffs = []
    for i in range(n):
        a0 = values[i]
        a1 = y_first[i]
        a2 = y_second[i] / 2
        a3 = (y_second[i + 1] - y_second[i]) / (6 * h[i])
        coeffs.append([a0, a1, a2, a3])

    return coeffs


def evaluate_spline(x, nodes, coeffs, spline_type='linear'):
    n = len(nodes) - 1
    i = 0
    for j in range(n):
        if nodes[j] <= x <= nodes[j + 1]:
            i = j
            break
        elif x < nodes[0]:
            i = 0
            break
        elif x > nodes[-1]:
            i = n - 1
            break

    c = coeffs[i]
    dx = x - nodes[i]    
    if spline_type == 'linear':
        return c[0] + c[1] * x
    elif spline_type == 'quadratic':
        return c[0] + c[1] * x + c[2] * x ** 2
    elif spline_type == 'cubic':
        return c[0] + c[1] * dx + c[2] * dx ** 2 + c[3] * dx ** 3

    return 0


def spline_vector(x_values, nodes, coeffs, spline_type='linear'):
    return np.array([evaluate_spline(x, nodes, coeffs, spline_type) for x in x_values])


def build_spline(nodes, values, spline_type='linear'):
    if spline_type == 'linear':
        return linear_spline_coefficients(nodes, values)
    elif spline_type == 'quadratic':
        return quadratic_spline_coefficients(nodes, values)
    elif spline_type == 'cubic':
        return cubic_spline_coefficients(nodes, values)
    return None
