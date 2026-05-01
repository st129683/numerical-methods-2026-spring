import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**2 * np.cos(x)


def generate_data(n):
    x_base = np.linspace(-1, 1, n)
    x_data = []
    y_data = []
    for x in x_base:
        y = f(x)
        for _ in range(3):
            x_data.append(x)
            y_data.append(y + np.random.uniform(-0.1, 0.1))
    return np.array(x_data), np.array(y_data)


def plot_degree(x_data, y_data, y_pred_n, y_pred_o, n):
    x_plot = np.linspace(-1, 1, 200)
    plt.figure()
    plt.scatter(x_data, y_data, s=8, alpha=0.5, label='experimental')
    plt.plot(x_plot, y_pred_n, 'b-', linewidth=2, label='normal eq')
    plt.plot(x_plot, y_pred_o, 'r--', linewidth=2, label='orthogonal')
    plt.title(f'degree n={n}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()
    plt.savefig(f'plot_n{n}.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_table(degrees, sse_n, sse_o):
    print(f"{'n':<3} | {'sse normal':<15} | {'sse orthogonal':<15} | {'error':<10}")
    print("-" * 50)
    for d, sn, so in zip(degrees, sse_n, sse_o):
        print(f"{d:<3} | {sn:<15.6f} | {so:<15.6f} | {abs(sn - so):<10.6f}")
