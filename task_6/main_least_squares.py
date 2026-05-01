import numpy as np
from tools import generate_data, plot_degree, print_table
import least_squares_normal as lsn
import least_squares_orthogonal as lso


def main():
    x_data, y_data = generate_data(60)
    degrees = [1, 2, 3, 4, 5]
    sse_normal = []
    sse_orthogonal = []

    for n in degrees:
        coeffs_n = lsn.solve_normal(x_data, y_data, n)
        y_pred_n = lsn.eval_normal(x_data, coeffs_n)
        sn = np.sum((y_data - y_pred_n)**2)
        sse_normal.append(sn)

        coeffs_o, alpha_o, beta_o = lso.solve_orthogonal(x_data, y_data, n)
        y_pred_o = lso.eval_orthogonal(x_data, coeffs_o, alpha_o, beta_o, n)
        so = np.sum((y_data - y_pred_o)**2)
        sse_orthogonal.append(so)

        x_plot = np.linspace(-1, 1, 200)
        y_plot_n = lsn.eval_normal(x_plot, coeffs_n)
        y_plot_o = lso.eval_orthogonal(x_plot, coeffs_o, alpha_o, beta_o, n)
        plot_degree(x_data, y_data, y_plot_n, y_plot_o, n)

    print_table(degrees, sse_normal, sse_orthogonal)


if __name__ == '__main__':
    main()
