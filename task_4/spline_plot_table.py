import numpy as np
import matplotlib.pyplot as plt

from tools import generate_equidistant_nodes, evaluate_function, calc_max_deviation, common_deviation, f
from plot_tools import print_table, plot_error_vs_n, get_table_data
from spline_construction import build_spline, spline_vector


def spline_wrapper(nodes, values, x_values, spline_type='linear'):
    coeffs = build_spline(nodes, values, spline_type)
    return spline_vector(x_values, nodes, coeffs, spline_type)


def get_spline_table_data(f, a, b, n_values, k, spline_type):
    test_points = generate_equidistant_nodes(a, b, k)
    data = {'n': [], 'k': [], 'R': []}

    for n in n_values:
        nodes = generate_equidistant_nodes(a, b, n)
        values = evaluate_function(f, nodes)
        coeffs = build_spline(nodes, values, spline_type)
        interp_values = spline_vector(test_points, nodes, coeffs, spline_type)
        R = calc_max_deviation(f, interp_values, test_points)

        data['n'].append(n)
        data['k'].append(k)
        data['R'].append(R)

    return data


def print_spline_table(data, title):
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    print(f"{'n (количество узлов)':<15} | {'k (количество тестовых точек)':<15} | {'R (максимальное отклонение)':<20}\n{'-'*100}")
    for i in range(len(data['n'])):
        print(f"{data['n'][i]:<15} | {data['k'][i]:<15} | {data['R'][i]:<20.6e}")
    print('='*100)


def plot_spline(f, a, b, n, spline_type, save_path=None):
    from spline_construction import build_spline, spline_vector

    m = 1000
    x = generate_equidistant_nodes(a, b, m)
    y = np.array([f(xi) for xi in x])

    nodes = generate_equidistant_nodes(a, b, n)
    values = evaluate_function(f, nodes)
    coeffs = build_spline(nodes, values, spline_type)
    y_spline = spline_vector(x, nodes, coeffs, spline_type)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'k-', lw=2, label='f(x)')
    ax.plot(x, y_spline, 'r--', lw=1.5, label=f'S(x), n={n}')
    ax.plot(nodes, values, 'bo', ms=6, label='Узлы')
    ax.set(xlabel='x', ylabel='y', title=f'Сплайн {spline_type} (n={n})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'график тут: {save_path}')
    plt.show()


def plot_all_splines(f, a, b, n, save_path=None):
    m = 1000
    x = generate_equidistant_nodes(a, b, m)
    y = np.array([f(xi) for xi in x])

    nodes = generate_equidistant_nodes(a, b, n)
    values = evaluate_function(f, nodes)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    spline_types = ['linear', 'quadratic', 'cubic']
    titles = ['S[1,0]', 'S[2,1]', 'S[3,2]']

    for ax, stype, title in zip(axes, spline_types, titles):
        ax.plot(x, y, 'k-', lw=2, label='f(x)')
        coeffs = build_spline(nodes, values, stype)
        y_spline = spline_vector(x, nodes, coeffs, stype)
        ax.plot(x, y_spline, 'r--', lw=1.5, label=f'spline(x)')
        ax.plot(nodes, values, 'bo', ms=5, label='узлы')
        ax.set(xlabel='x', ylabel='y', title=title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'сравнение сплайнов ({n=})')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'график тут: {save_path}')
    plt.show()


def plot_error_comparison(f, a, b, n_values, save_path=None):
    from lagrange_construction import lagrange_polynomial_vector

    k = 100
    test_points = np.linspace(a, b, k)
    f_values = np.array([f(x) for x in test_points])

#     R_spline = []
#     R_lagrange = []
# 
#     for n in n_values:
#         nodes = generate_equidistant_nodes(a, b, n)
#         values = evaluate_function(f, nodes)
# 
#         coeffs = build_spline(nodes, values, 'cubic')
#         spline_vals = spline_vector(test_points, nodes, coeffs, 'cubic')
#         R_spline.append(calc_max_deviation(f, spline_vals, test_points))
# 
#         lagrange_vals = lagrange_polynomial_vector(nodes, values, test_points)
#         R_lagrange.append(calc_max_deviation(f, lagrange_vals, test_points))

    nodes = generate_equidistant_nodes(a, b, n_values)
    coeffs = build_spline(nodes, f_values, 'cubic')

    spline_vals = spline_vector(test_points, nodes, coeffs, 'cubic')
    R_spline = common_deviation(f, spline_vals, test_points)
    lagrange_vals = lagrange_polynomial_vector(nodes, f_values, test_points)
    R_lagrange = common_deviation(f, lagrange_vals, test_points)

    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(n_values, R_spline, 'go-', lw=2, ms=6, label='сплайн S[3,2]')
    # ax.plot(n_values, R_lagrange, 'rs-', lw=2, ms=6, label='полином Лагранжа')
    ax.plot(test_points, R_spline, 'go-', lw=2, ms=6, label='сплайн S[3,2]')
    ax.plot(test_points, R_lagrange, 'rs-', lw=2, ms=6, label='полином Лагранжа')
    ax.set(xlabel='n (количество узлов)', ylabel='R (максимальное отклонение)', 
           title='сравнение погрешностей: сплайн vs лагранж')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'график тут: {save_path}')
    plt.show()


def main():
    a, b = -1, 1

    n_values = [5, 10, 15, 20, 25, 30, 50, 100, 150, 200]
    k = 1000

    print(f'функция: f(x) = x**2 + 1 - arccos(x)')
    print(f'отрезок: [{a}, {b}], n = {n_values}, k = {k}\n')

    spline_types = ['linear', 'quadratic', 'cubic']
    spline_names = ['S[1,0]', 'S[2,1]', 'S[3,2]']

    for stype, sname in zip(spline_types, spline_names):
        print(f"\n{'='*100}")
        print(f'считаю {sname}................')
        print('='*100)

        data = get_spline_table_data(f, a, b, n_values, k, stype)
        print_spline_table(data, f'cплайн {sname}')

    plot_all_splines(f, a, b, 10, save_path='all_splines.png')

    for n in [5, 10, 20]:
        plot_spline(f, a, b, n, 'cubic', save_path=f'cubic_spline_n{n}.png')

    plot_error_comparison(f, a, b, 20, save_path='spline_vs_lagrange_error.png')

    print('УРАААААААА!!!!!!!! ВСЕ ПОЛУЧИЛОСЬ!!!!!!!!!')


if __name__ == '__main__':
    main()
