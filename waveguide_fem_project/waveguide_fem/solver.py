import numpy as np
import scipy.sparse.linalg as spla

from .fem import assemble_system, normalize_mode_vector


def solve_mode_family(mesh, family, kmax=30):
    K, M = assemble_system(mesh)
    npts = mesh["points"].shape[0]

    if family == "TM":
        keep = np.ones(npts, dtype=bool)
        keep[np.unique(mesh["boundary_nodes"])] = False
        interior = np.where(keep)[0]
        Kr = K[interior][:, interior]
        Mr = M[interior][:, interior]
        k = min(kmax, max(1, Kr.shape[0] - 2))
        eigvals, eigvecs_reduced = spla.eigsh(Kr, k=k, M=Mr, sigma=0.0, which="LM")
        order = np.argsort(np.real(eigvals))
        eigvals = np.real(eigvals[order])
        eigvecs_reduced = np.real(eigvecs_reduced[:, order])
        good = eigvals > 1e-10
        eigvals = eigvals[good]
        eigvecs_reduced = eigvecs_reduced[:, good]
        eigvecs = np.zeros((npts, eigvecs_reduced.shape[1]))
        eigvecs[interior, :] = eigvecs_reduced
        eigvecs = np.column_stack([normalize_mode_vector(eigvecs[:, i], M) for i in range(eigvecs.shape[1])])
        return eigvals, eigvecs, K, M

    if family == "TE":
        k = min(kmax + 8, max(2, npts - 2))
        eigvals, eigvecs = spla.eigsh(K, k=k, M=M, which="SM")
        order = np.argsort(np.real(eigvals))
        eigvals = np.real(eigvals[order])
        eigvecs = np.real(eigvecs[:, order])
        good = eigvals > 1e-8
        eigvals = eigvals[good]
        eigvecs = eigvecs[:, good]
        if len(eigvals) == 0:
            raise RuntimeError("did not get any nonzero TE eigenvalues")
        eigvecs = np.column_stack([normalize_mode_vector(eigvecs[:, i], M) for i in range(eigvecs.shape[1])])
        return eigvals, eigvecs, K, M

    raise ValueError("family must be TE or TM")
