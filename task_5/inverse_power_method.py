import numpy as np
from tools import DELTA, RTOL

def inverse_power_step(A, sigma, z):
    n = A.shape[0]
    M = A - sigma * np.eye(n)
    try:
        y = np.linalg.solve(M, z)
    except np.linalg.LinAlgError:
        return sigma, z, True
    norm_y = np.linalg.norm(y)
    if norm_y == 0:
        return sigma, z, True
    z_new = y / norm_y
    mask = np.abs(y) > DELTA
    if not np.any(mask):
        return sigma, z_new, True
    mus = z[mask] / y[mask]
    sigma_new = sigma + np.mean(mus)
    return sigma_new, z_new, False

def inverse_power_method_all(A, shifts):
    eigenpairs = []
    n = A.shape[0]
    for sigma0 in shifts:
        z = np.random.rand(n)
        z = z / np.linalg.norm(z)
        sigma = sigma0
        for _ in range(1000):
            sigma_new, z_new, converged = inverse_power_step(A, sigma, z)
            if converged or np.abs(sigma_new - sigma) < RTOL * max(np.abs(sigma_new), 1e-15):
                eigenpairs.append((sigma_new, z_new))
                break
            sigma = sigma_new
            z = z_new
    return eigenpairs
