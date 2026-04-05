import numpy as np
from constants import czero, freqzero, S, epszero
from profiles import sigama_side, compute_smax


def make_grid(n_xxx, nyy, ppww, tmax):
    lam = czero / freqzero
    dx = lam / ppww
    dy = dx
    dt = S / (czero * np.sqrt((1.0 / dx**2) + (1.0 / dy**2)))
    nt = int(np.ceil(tmax / dt))
    taw = tmax / 7.0
    return {
        "n_xxx": n_xxx,
        "nyy": nyy,
        "ppww": ppww,
        "tmax": tmax,
        "lam": lam,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "nt": nt,
        "taw": taw,
    }


def make_medium(grid, n_npml, pec_pol=None):
    n_xxx = grid["n_xxx"]
    nyy = grid["nyy"]
    dx = grid["dx"]
    dt = grid["dt"]

    eps = epszero * np.ones((n_xxx, nyy))
    siga = np.zeros((n_xxx, nyy))

    if pec_pol is None:
        pec_pol = np.zeros((n_xxx, nyy), dtype=bool)

    smax_v_xx = compute_smax(n_npml, dx)
    sx1 = sigama_side(n_xxx, n_npml, smax_v_xx)
    syy1 = sigama_side(nyy, n_npml, smax_v_xx)

    sx = sx1[:, None] * np.ones((1, nyy))
    syy = np.ones((n_xxx, 1)) * syy1[None, :]

    perf_match = np.zeros((n_xxx, nyy), dtype=bool)
    perf_match[:n_npml, :] = True
    perf_match[-n_npml:, :] = True
    perf_match[:, :n_npml] = True
    perf_match[:, -n_npml:] = True

    ax_y = (eps / dt) - sx / 2.0
    bx = (eps / dt) + sx / 2.0
    ay = (eps / dt) - syy / 2.0
    byy = (eps / dt) + syy / 2.0

    return {
        "n_npml": n_npml,
        "eps": eps,
        "siga": siga,
        "pec_pol": pec_pol,
        "perf_match": perf_match,
        "ax_y": ax_y,
        "bx": bx,
        "ay": ay,
        "byy": byy,
    }


def make_state(n_xxx, nyy):
    # Array shapes are kept exactly the same as in the original file.
    return {
        "Ez": np.zeros((n_xxx, nyy)),
        "Hx": np.zeros((n_xxx, nyy - 1)),
        "Hy": np.zeros((n_xxx - 1, nyy)),
        "Esxx_test": np.zeros((n_xxx, nyy)),
        "Esyy": np.zeros((n_xxx, nyy)),
    }
