import numpy as np

DELTA = 1e-8
RTOL = 1e-6
EPSILON = 1e-8

def givens_params(a, b):
    if b == 0:
        return 1.0, 0.0
    r = np.hypot(a, b)
    return a / r, -b / r

def apply_givens_left(A, c, s, i, k):
    n = A.shape[0]
    for j in range(n):
        a_ij = A[i, j]
        a_kj = A[k, j]
        A[i, j] = c * a_ij - s * a_kj
        A[k, j] = s * a_ij + c * a_kj

def apply_givens_right(A, c, s, i, k):
    n = A.shape[0]
    for j in range(n):
        a_ji = A[j, i]
        a_jk = A[j, k]
        A[j, i] = c * a_ji - s * a_jk
        A[j, k] = s * a_ji + c * a_jk

def reduce_to_hessenberg(A):
    n = A.shape[0]
    H = A.copy()
    for k in range(n - 2):
        for i in range(k + 2, n):
            if abs(H[i, k]) > 1e-15:
                c, s = givens_params(H[k + 1, k], H[i, k])
                apply_givens_left(H, c, s, k + 1, i)
                apply_givens_right(H, c, s, k + 1, i)
    return H

def qr_decomposition_hessenberg(H):
    n = H.shape[0]
    Q = np.eye(n)
    R = H.copy()
    for i in range(n - 1):
        c, s = givens_params(R[i, i], R[i + 1, i])
        apply_givens_left(R, c, s, i, i + 1)
        apply_givens_right(Q, c, s, i, i + 1)
    return Q, R
