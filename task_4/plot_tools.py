import numpy as np
import matplotlib.pyplot as plt

from tools import generate_equidistant_nodes, generate_optimal_nodes, evaluate_function, calc_max_deviation


def print_table(data, title):
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    print(f"{'n':<10} | {'m':<10} | {'R (равноотст.)':<20} | {'R_opt (оптим.)':<20}\n{'-'*100}")
    for i in range(len(data['n'])):
        print(f"{data['n'][i]:<10} | {data['m'][i]:<10} | {data['R'][i]:<20.6e} | {data['R_opt'][i]:<20.6e}")
    print('='*100)


def plot_interpolation(f, a, b, n_values, m, poly_func, title, save_path=None):    
    x = generate_equidistant_nodes(a, b, m)
    y = evaluate_function(f, x)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_values)))

    for ax, nodes_func, subplot_title in [(ax1, generate_equidistant_nodes, 'равноотстоящие узлы'), 
                                           (ax2, generate_optimal_nodes, 'оптимальные узлы')
                                         ]:
        ax.plot(x, y, 'k-', lw=2, label='f(x)')
        for i, n in enumerate(n_values):
            nodes = nodes_func(a, b, n)
            values = evaluate_function(f, nodes)
            y_interp = poly_func(nodes, values, x)
            ax.plot(x, y_interp, '--', color=colors[i], lw=1.5, label=f'n={n}')
            ax.plot(nodes, values, 'o', color=colors[i], ms=4)
        ax.set(xlabel='x', ylabel='y', title=subplot_title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'график тут: {save_path}')
    plt.show()


def plot_error_vs_n(f, a, b, n_values, m, poly_func, title, save_path=None):    
    test_points = generate_equidistant_nodes(a, b, m)
    R_eq, R_opt = [], []

    for n in n_values:
        nodes_eq = generate_equidistant_nodes(a, b, n)
        values_eq = evaluate_function(f, nodes_eq)
        interp_eq = poly_func(nodes_eq, values_eq, test_points)
        R_eq.append(calc_max_deviation(f, interp_eq, test_points))

        nodes_opt = generate_optimal_nodes(a, b, n)
        values_opt = evaluate_function(f, nodes_opt)
        interp_opt = poly_func(nodes_opt, values_opt, test_points)
        R_opt.append(calc_max_deviation(f, interp_opt, test_points))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_values, R_eq, 'ro-', lw=2, ms=6, label='равноотстоящие узлы')
    ax.plot(n_values, R_opt, 'bs-', lw=2, ms=6, label='оптимальные узлы')
    ax.set(xlabel='n (количество узлов)', ylabel='R (максимальное отклонение)', title=title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'график тут: {save_path}')
    plt.show()


def get_table_data(f, a, b, n_values, m, poly_func):    
    test_points = generate_equidistant_nodes(a, b, m)
    data = {'n': [], 'm': [], 'R': [], 'R_opt': []}

    for n in n_values:
        nodes_eq = generate_equidistant_nodes(a, b, n)
        values_eq = evaluate_function(f, nodes_eq)
        R_eq = calc_max_deviation(f, poly_func(nodes_eq, values_eq, test_points), test_points)

        nodes_opt = generate_optimal_nodes(a, b, n)
        values_opt = evaluate_function(f, nodes_opt)
        R_opt = calc_max_deviation(f, poly_func(nodes_opt, values_opt, test_points), test_points)

        data['n'].append(n)
        data['m'].append(m)
        data['R'].append(R_eq)
        data['R_opt'].append(R_opt)

    return data
