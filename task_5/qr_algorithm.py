import numpy as np
from tools import EPSILON, reduce_to_hessenberg, qr_decomposition_hessenberg

def qr_algorithm(A):
    H = reduce_to_hessenberg(A)
    n = H.shape[0]
    eigenvalues = []
    m = n
    while m > 0:
        if m == 1:
            eigenvalues.append(H[0, 0])
            break
        if np.abs(H[m - 1, m - 2]) < EPSILON * (np.abs(H[m - 1, m - 1]) + np.abs(H[m - 2, m - 2])):
            eigenvalues.append(H[m - 1, m - 1])
            m -= 1
            continue
        sigma = H[m - 1, m - 1]
        H_shifted = H[:m, :m] - sigma * np.eye(m)
        Q, R = qr_decomposition_hessenberg(H_shifted)
        H[:m, :m] = R @ Q + sigma * np.eye(m)
    return eigenvalues
