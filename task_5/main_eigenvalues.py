import numpy as np
from tools import EPSILON
from power_method import power_method
from inverse_power_method import inverse_power_method_all
from qr_algorithm import qr_algorithm


def generate_matrices(n):
    Lambda = np.diag(np.random.uniform(-10, 10, n))
    while True:
        C = np.random.uniform(-1, 1, (n, n))
        if np.abs(np.linalg.det(C)) > 1e-5:
            break
    A = np.linalg.inv(C) @ Lambda @ C
    return Lambda, C, A


def main():
    n = 5
    Lambda, C, A = generate_matrices(n)
    true_eigenvalues = np.diag(Lambda)

    print('\n\nmatrix Lambda')
    print(Lambda)

    print('\n\nmatrix C')
    print(C)

    print('\n\nmatrix A')
    print(A)

    print('\n\n\n=== power method ===')
    lam_max, _ = power_method(A)

    print(f'max abs eigenvalue: {lam_max}')
    max_abs_eigenvalue = np.max(np.abs(true_eigenvalues))
    for eigenvalue in true_eigenvalues:
        if max_abs_eigenvalue == np.abs(eigenvalue):
            max_abs_eigenvalue = eigenvalue
            break
    print(f'error: {np.abs(lam_max - max_abs_eigenvalue)}')
    
    print('\n\n\n=== inverse power method ===')
    shifts = np.diag(A)
    pairs = inverse_power_method_all(A, shifts)
    inv_eigs = [p[0] for p in pairs]
    errors_inv = sorted(np.abs(np.array(inv_eigs)[:, None] - true_eigenvalues).min(axis=1))
    for i, pair in enumerate(pairs):
        eigenvalue, eigenvector = pair[0], pair[1]
        print(f'\neigenvalue = {eigenvalue}, eigenvector = {eigenvector}')
        print(f'error: {errors_inv[i]}')

    print('\n\n\n=== qr algorithm ===')
    qr_eigs = qr_algorithm(A)
    errors_qr = sorted(np.abs(np.array(qr_eigs)[:, None] - true_eigenvalues).min(axis=1))
    for i, eigenvalue in enumerate(qr_eigs):
        print(f'\neigenvalue = {eigenvalue}')
        print(f'error = {errors_qr[i]}')

if __name__ == '__main__':
    main()
