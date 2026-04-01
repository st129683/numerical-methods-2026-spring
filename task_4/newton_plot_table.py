import numpy as np

from tools import f
from plot_tools import print_table, plot_interpolation, plot_error_vs_n, get_table_data
from newton_construction import divided_differences, newton_polynomial_vector


def newton_poly_wrapper(nodes, values, x_values):
    div_diff = divided_differences(nodes, values)
    return newton_polynomial_vector(nodes, div_diff, x_values)


def main():
    a, b = -1, 1

    n_values = [5, 10, 15, 20, 25, 30, 50, 100, 150, 200]
    m = 1000

    print(f'функция: f(x) = x**2 + 1 - arccos(x)')
    print(f'отрезок: [{a}, {b}], n = {n_values}, m = {m}\n')

    data = get_table_data(f, a, b, n_values, m, newton_poly_wrapper)

    print_table(data, 'интерп. полином ньютона')
    plot_interpolation(f, a, b, [5, 10, 20], m, newton_poly_wrapper,
                      'полином ньютона', save_path='newton_interp.png')

    plot_error_vs_n(f, a, b, n_values, m, newton_poly_wrapper,
                   'зависимость погрешности от n (ньютон)', save_path='newton_error.png')

    print('УРАААААААА!!!!!!!! ВСЕ ПОЛУЧИЛОСЬ!!!!!!!!!')


if __name__ == '__main__':
    main()
