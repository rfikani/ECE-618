import numpy as np

from basis import basis_derivative, basis_value
from greens import green_from_distance, psi_jin
from geometry import segment_point


def pair_integral_gauss(nodes, dl, t, p, q, m, n, k, xg, wg):
    lm = np.array([basis_value(m, p, x) for x in xg])
    ln = np.array([basis_value(n, q, y) for y in xg])

    dm = basis_derivative(m, p, dl)
    dn = basis_derivative(n, q, dl)

    rp = nodes[p][None, :] + xg[:, None] * (nodes[p + 1] - nodes[p])[None, :]
    rq = nodes[q][None, :] + xg[:, None] * (nodes[q + 1] - nodes[q])[None, :]

    r = np.linalg.norm(rp[:, None, :] - rq[None, :, :], axis=2)
    g = green_from_distance(r, k)

    w2 = wg[:, None] * wg[None, :]
    tdot = np.dot(t[p], t[q])

    aval = dl[p] * dl[q] * np.sum(w2 * lm[:, None] * ln[None, :] * tdot * g)
    bval = dl[p] * dl[q] * dm * dn * np.sum(w2 * g)

    return aval, bval


def pair_integral_self_subsegments(nodes, dl, p, m, n, radius, k, nsub):
    dm = basis_derivative(m, p, dl)
    dn = basis_derivative(n, p, dl)

    dsub = dl[p] / nsub

    aval = 0.0 + 0.0j
    bval = 0.0 + 0.0j

    for i in range(nsub):
        xi = (i + 0.5) / nsub
        lmi = basis_value(m, p, xi)
        rpi = segment_point(nodes, p, xi)

        for j in range(nsub):
            eta = (j + 0.5) / nsub
            lnj = basis_value(n, p, eta)

            if i == j:
                gint = psi_jin(dsub, radius, k)
            else:
                rqj = segment_point(nodes, p, eta)
                rij = np.linalg.norm(rpi - rqj)
                gint = dsub * dsub * green_from_distance(rij, k)

            aval += lmi * lnj * gint
            bval += dm * dn * gint

    return aval, bval


def is_near_pair(nodes, dl, p, q, near_factor=2.5):
    if p == q:
        return True

    mid = 0.5 * (nodes[:-1] + nodes[1:])
    dmid = np.linalg.norm(mid[p] - mid[q])

    if abs(p - q) <= 1:
        return True

    return dmid < near_factor * max(dl[p], dl[q])
