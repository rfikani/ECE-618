import numpy as np
from constants import epszero, czero, pmlpow, pmlreflect


def sigama_side(n, n_npml, smax_v_xx):
    # 1D graded conductivity profile used on each side of the PML.
    s = np.zeros(n)
    for i in range(n):
        if i < n_npml:
            x = (n_npml - i) / n_npml
            s[i] = smax_v_xx * x**pmlpow
        elif i >= n - n_npml:
            x = (i - (n - n_npml - 1)) / n_npml
            s[i] = smax_v_xx * x**pmlpow
    return s


def compute_smax(n_npml, dx):
    return -(pmlpow + 1) * epszero * czero * np.log(pmlreflect) / (2.0 * n_npml * dx)
