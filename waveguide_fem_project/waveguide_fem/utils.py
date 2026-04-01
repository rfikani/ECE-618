import math
import numpy as np

from .fem import m_norm, normalize_mode_vector


def group_eigenvalues(vals, rel_tol=5e-3):
    groups = []
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals):
            ref = 0.5 * (vals[i] + vals[j])
            if abs(vals[j] - vals[i]) <= rel_tol * max(ref, 1.0):
                j += 1
            else:
                break
        groups.append((i, j))
        i = j
    return groups

def mass_orthonormalize(V, M):
    basis = []
    for k in range(V.shape[1]):
        v = V[:, k].copy()
        for q in basis:
            v = v - q * float(q @ (M @ v))
        nrm = m_norm(v, M)
        if nrm > 1e-10:
            basis.append(v / nrm)
    if not basis:
        raise ValueError("could not build M-orthonormal basis")
    return np.column_stack(basis)



def project_onto_subspace(space, target, M):
    coeffs = space.T @ (M @ target)
    v = space @ coeffs
    return normalize_mode_vector(v, M)



def pick_representative_mode(vec_block, M):
    B = mass_orthonormalize(vec_block, M)
    if B.shape[1] == 1:
        return normalize_mode_vector(B[:, 0], M)
    weights = np.asarray([1.0 / (k + 1) for k in range(B.shape[1])], dtype=float)
    return normalize_mode_vector(B @ weights, M)



def beta_from_kc(k0, kc):
    beta = np.empty_like(k0, dtype=complex)
    propagating = k0 >= kc
    beta[propagating] = np.sqrt(k0[propagating] ** 2 - kc**2)
    beta[~propagating] = 1j * np.sqrt(kc**2 - k0[~propagating] ** 2)
    return beta
