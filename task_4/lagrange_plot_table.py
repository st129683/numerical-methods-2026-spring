import numpy as np

from tools import f
from plot_tools import print_table, plot_interpolation, plot_error_vs_n, get_table_data
from lagrange_construction import lagrange_polynomial_vector


def main():
    a, b = -1, 1

    n_values = [5, 10, 15, 20, 25, 30, 50, 100, 150, 200]
    m = 1000

    print(f'функция: f(x) = x**2 + 1 - arccos(x)')
    print(f'отрезок: [{a}, {b}], n = {n_values}, m = {m}\n')

    data = get_table_data(f, a, b, n_values, m, lagrange_polynomial_vector)

    print_table(data, 'интерп. полином лагранжа')

    plot_interpolation(f, a, b, [100], m, lagrange_polynomial_vector, 
                      'полином лагранжа', save_path='lagrange_interp.png')

    plot_error_vs_n(f, a, b, n_values, m, lagrange_polynomial_vector,
                   'зависимость погрешности от n (лагранж)', save_path='lagrange_error.png')

    print('УРАААААААА!!!!!!!! ВСЕ ПОЛУЧИЛОСЬ!!!!!!!!!')


if __name__ == '__main__':
    main()
