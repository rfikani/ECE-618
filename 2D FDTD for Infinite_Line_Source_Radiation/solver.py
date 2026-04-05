import numpy as np
from constants import muzero
from source import drive_f


def steptmz(Ez, Hx, Hy, Esxx_test, Esyy, pec_pol, perf_match, eps, siga, ax_y, bx, ay, byy, dx, dy, dt, srcij, t_half, taw):
   
    n_xxx, nyy = Ez.shape

    # Update Hx
    d1 = Ez[:, 1:] - Ez[:, :-1]
    pmlhx = perf_match[:, :-1] | perf_match[:, 1:]
    Hx[~pmlhx] -= (dt / (muzero * dy)) * d1[~pmlhx]

    ayh = 0.5 * (ay[:, :-1] + ay[:, 1:])
    byyh = 0.5 * (byy[:, :-1] + byy[:, 1:])
    epsh = 0.5 * (eps[:, :-1] + eps[:, 1:])
    Hx[pmlhx] = (ayh[pmlhx] * Hx[pmlhx] - (epsh[pmlhx] / (muzero * dy)) * d1[pmlhx]) / byyh[pmlhx]

    # Update Hy
    dtwo = Ez[1:, :] - Ez[:-1, :]
    pmlhhy = perf_match[:-1, :] | perf_match[1:, :]
    Hy[~pmlhhy] += (dt / (muzero * dx)) * dtwo[~pmlhhy]

    axhh = 0.5 * (ax_y[:-1, :] + ax_y[1:, :])
    bxh = 0.5 * (bx[:-1, :] + bx[1:, :])
    epsh2 = 0.5 * (eps[:-1, :] + eps[1:, :])
    Hy[pmlhhy] = (axhh[pmlhhy] * Hy[pmlhhy] + (epsh2[pmlhhy] / (muzero * dx)) * dtwo[pmlhhy]) / bxh[pmlhhy]

    # Build the discrete curl pieces at Ez locations
    dHy_dx = np.zeros((n_xxx, nyy))
    dHx_dy = np.zeros((n_xxx, nyy))
    dHy_dx[1:-1, :] = (Hy[1:, :] - Hy[:-1, :]) / dx
    dHx_dy[:, 1:-1] = (Hx[:, 1:] - Hx[:, :-1]) / dy

    # Split-field PML update
    Esxx_test[perf_match] = (ax_y[perf_match] * Esxx_test[perf_match] + dHy_dx[perf_match]) / bx[perf_match]
    Esyy[perf_match] = (ay[perf_match] * Esyy[perf_match] - dHx_dy[perf_match]) / byy[perf_match]
    Ez[perf_match] = Esxx_test[perf_match] + Esyy[perf_match]

    # Interior Ez update with source term.
    inside_r = ~perf_match
    a_zero = (eps / dt) - siga / 2.0
    b0 = (eps / dt) + siga / 2.0

    J = np.zeros((n_xxx, nyy))
    if srcij is not None:
        J[srcij[0], srcij[1]] = drive_f(t_half, dx, dy, taw)

    curll = dHy_dx - dHx_dy
    Ez[inside_r] = (a_zero[inside_r] * Ez[inside_r] + curll[inside_r] - J[inside_r]) / b0[inside_r]

    # PEC enforcement after the electric update
    Ez[pec_pol] = 0.0
    Esxx_test[pec_pol] = 0.0
    Esyy[pec_pol] = 0.0
