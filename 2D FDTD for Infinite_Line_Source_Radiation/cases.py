import numpy as np
from scipy.special import hankel2

from constants import muzero, epszero, freqzero, src_amp
from mesh import make_grid, make_medium, make_state
from solver import steptmz


def run_validation_case():
    n_xxx = 240
    nyy = 240
    n_npml = 20
    ppww = 40.0
    tmax = 2.167e-9

    grid = make_grid(n_xxx, nyy, ppww, tmax)
    medium = make_medium(grid, n_npml)
    state = make_state(n_xxx, nyy)

    Ez = state["Ez"]
    Hx = state["Hx"]
    Hy = state["Hy"]
    Esxx_test = state["Esxx_test"]
    Esyy = state["Esyy"]

    eps = medium["eps"]
    siga = medium["siga"]
    pec_pol = medium["pec_pol"]
    perf_match = medium["perf_match"]
    ax_y = medium["ax_y"]
    bx = medium["bx"]
    ay = medium["ay"]
    byy = medium["byy"]

    dx = grid["dx"]
    dy = grid["dy"]
    dt = grid["dt"]
    nt = grid["nt"]
    taw = grid["taw"]

    srci = n_xxx // 2
    src_j = nyy // 2

    for n in range(nt):
        th = (n + 0.5) * dt
        steptmz(Ez, Hx, Hy, Esxx_test, Esyy, pec_pol, perf_match, eps, siga, ax_y, bx, ay, byy, dx, dy, dt, (srci, src_j), th, taw)

    Ezval = Ez.copy()
    xrel = (np.arange(n_xxx) - srci) * dx
    rho = np.abs(xrel) + 1e-12
    w = 2.0 * np.pi * freqzero
    k = w * np.sqrt(muzero * epszero)
    Ez_w = -(k**2 * src_amp) / (4.0 * w * epszero) * hankel2(0, k * rho)
    tf = (nt - 1) * dt
    Ezrref = np.real(Ez_w * np.exp(1j * (w * tf - np.pi / 2.0)))
    Ez_num = Ezval[:, src_j].copy()
    g00d = rho > 5.0 * dx
    scale = np.max(np.abs(Ezrref[g00d])) / (np.max(np.abs(Ez_num[g00d])) + 1e-30)
    xabs = np.arange(n_xxx) * dx

    return {
        "grid": grid,
        "medium": medium,
        "state": state,
        "Ezval": Ezval,
        "srci": srci,
        "src_j": src_j,
        "Ezrref": Ezrref,
        "Ez_num": Ez_num,
        "scale": scale,
        "xabs": xabs,
    }


def run_slot_cases():
    n_xxx = 320
    nyy = 320
    n_npml = 20
    ppww = 25.0
    tmax = 4.167e-9

    grid = make_grid(n_xxx, nyy, ppww, tmax)
    medium1 = make_medium(grid, n_npml, np.zeros((n_xxx, nyy), dtype=bool))
    medium2 = make_medium(grid, n_npml, np.zeros((n_xxx, nyy), dtype=bool))
    state1 = make_state(n_xxx, nyy)
    state2 = make_state(n_xxx, nyy)

    dx = grid["dx"]
    dy = grid["dy"]
    dt = grid["dt"]
    nt = grid["nt"]
    taw = grid["taw"]

    sheetjj = nyy // 2
    medium1["pec_pol"][:, sheetjj:sheetjj + 1] = True
    medium2["pec_pol"][:, sheetjj:sheetjj + 1] = True

    slot_len = 0.07
    half_oo = max(1, int(round((slot_len / 2.0) / dx)))
    c = n_xxx // 2
    medium1["pec_pol"][max(0, c - half_oo):min(n_xxx, c + half_oo + 1), sheetjj:sheetjj + 1] = False

    cA = n_xxx // 3
    cbee = 2 * n_xxx // 3
    medium2["pec_pol"][max(0, cA - half_oo):min(n_xxx, cA + half_oo + 1), sheetjj:sheetjj + 1] = False
    medium2["pec_pol"][max(0, cbee - half_oo):min(n_xxx, cbee + half_oo + 1), sheetjj:sheetjj + 1] = False

    srci = n_xxx // 2
    src_j = nyy // 4
    y_obs = int(round(0.75 * nyy))
    steps_per_period = max(1, int(round((1.0 / freqzero) / dt)))
    keep = 10 * steps_per_period
    grabone = []
    grab2 = []

    for n in range(nt):
        th = (n + 0.5) * dt
        steptmz(state1["Ez"], state1["Hx"], state1["Hy"], state1["Esxx_test"], state1["Esyy"], medium1["pec_pol"], medium1["perf_match"], medium1["eps"], medium1["siga"], medium1["ax_y"], medium1["bx"], medium1["ay"], medium1["byy"], dx, dy, dt, (srci, src_j), th, taw)
        steptmz(state2["Ez"], state2["Hx"], state2["Hy"], state2["Esxx_test"], state2["Esyy"], medium2["pec_pol"], medium2["perf_match"], medium2["eps"], medium2["siga"], medium2["ax_y"], medium2["bx"], medium2["ay"], medium2["byy"], dx, dy, dt, (srci, src_j), th, taw)

        if n >= nt - keep:
            grabone.append(state1["Ez"][:, y_obs].copy())
            grab2.append(state2["Ez"][:, y_obs].copy())

    Aone = np.stack(grabone, axis=0)
    A2 = np.stack(grab2, axis=0)
    I0ne = np.mean(Aone**2, axis=0)
    I2 = np.mean(A2**2, axis=0)
    x = np.arange(n_xxx) * dx

    return {
        "grid": grid,
        "single": state1,
        "double": state2,
        "I0ne": I0ne,
        "I2": I2,
        "x": x,
    }


def run_scattering_cases():
    n_xxx = 320
    nyy = 320
    n_npml = 20
    ppww = 25.0
    tmax = 4.167e-9

    grid = make_grid(n_xxx, nyy, ppww, tmax)
    dx = grid["dx"]
    dy = grid["dy"]
    dt = grid["dt"]
    nt = grid["nt"]
    taw = grid["taw"]

    srci = n_xxx // 2
    src_j = nyy // 4
    cxx = n_xxx // 2
    cy = nyy // 2 + nyy // 10
    r_r = 0.06 / dx

    i_i = np.arange(n_xxx)[:, None]
    jj = np.arange(nyy)[None, :]
    pec_c = (i_i - cxx) ** 2 + (jj - cy) ** 2 <= r_r**2

    pec_r = np.zeros((n_xxx, nyy), dtype=bool)
    sxr = int(round(0.10 / dx))
    syyr = int(round(0.06 / dy))
    xnau = cxx - sxr // 2
    yano = cy - syyr // 2
    pec_r[xnau:xnau + sxr, yano:yano + syyr] = True

    medium_c = make_medium(grid, n_npml, pec_c)
    medium_r = make_medium(grid, n_npml, pec_r)
    state_c = make_state(n_xxx, nyy)
    state_r = make_state(n_xxx, nyy)

    for n in range(nt):
        th = (n + 0.5) * dt
        steptmz(state_c["Ez"], state_c["Hx"], state_c["Hy"], state_c["Esxx_test"], state_c["Esyy"], medium_c["pec_pol"], medium_c["perf_match"], medium_c["eps"], medium_c["siga"], medium_c["ax_y"], medium_c["bx"], medium_c["ay"], medium_c["byy"], dx, dy, dt, (srci, src_j), th, taw)
        steptmz(state_r["Ez"], state_r["Hx"], state_r["Hy"], state_r["Esxx_test"], state_r["Esyy"], medium_r["pec_pol"], medium_r["perf_match"], medium_r["eps"], medium_r["siga"], medium_r["ax_y"], medium_r["bx"], medium_r["ay"], medium_r["byy"], dx, dy, dt, (srci, src_j), th, taw)

    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    ring_r = 0.18 / dx
    x_s = cxx + ring_r * np.cos(theta)
    ys = cy + ring_r * np.sin(theta)

    xnau = np.floor(x_s).astype(int)
    yano = np.floor(ys).astype(int)
    x1 = np.clip(xnau + 1, 0, n_xxx - 1)
    y1 = np.clip(yano + 1, 0, nyy - 1)
    xnau = np.clip(xnau, 0, n_xxx - 1)
    yano = np.clip(yano, 0, nyy - 1)

    wx_x = x_s - xnau
    wy = ys - yano

    v0o = state_c["Ez"][xnau, yano]
    v10 = state_c["Ez"][x1, yano]
    v01 = state_c["Ez"][xnau, y1]
    v_11 = state_c["Ez"][x1, y1]
    Ic = np.abs((1 - wx_x) * (1 - wy) * v0o + wx_x * (1 - wy) * v10 + (1 - wx_x) * wy * v01 + wx_x * wy * v_11) ** 2

    v0o = state_r["Ez"][xnau, yano]
    v10 = state_r["Ez"][x1, yano]
    v01 = state_r["Ez"][xnau, y1]
    v_11 = state_r["Ez"][x1, y1]
    Ir = np.abs((1 - wx_x) * (1 - wy) * v0o + wx_x * (1 - wy) * v10 + (1 - wx_x) * wy * v01 + wx_x * wy * v_11) ** 2

    return {
        "grid": grid,
        "circular": state_c,
        "rectangular": state_r,
        "theta": theta,
        "Ic": Ic,
        "Ir": Ir,
    }
