import numpy as np
from numpy.polynomial.legendre import leggauss

from constants import c0, Z0
from integrals import is_near_pair, pair_integral_gauss, pair_integral_self_subsegments
from geometry import segment_data


def assemble_matrix(nodes, radius, freq, ng_far=8, ng_near=18, nsub_self=24):
    k = 2.0 * np.pi * freq / c0
    dl, t, s = segment_data(nodes)

    ns = len(dl)
    nu = ns - 1

    if nu <= 0:
        raise ValueError("At least two physical segments are required.")

    zmat = np.zeros((nu, nu), dtype=complex)

    xf, wf = leggauss(ng_far)
    xg_far = 0.5 * (xf + 1.0)
    wg_far = 0.5 * wf

    xn, wn = leggauss(ng_near)
    xg_near = 0.5 * (xn + 1.0)
    wg_near = 0.5 * wn

    for mi, m in enumerate(range(1, ns)):
        for ni, n in enumerate(range(1, ns)):
            asum = 0.0 + 0.0j
            bsum = 0.0 + 0.0j

            for p in (m - 1, m):
                for q in (n - 1, n):
                    if p == q:
                        av, bv = pair_integral_self_subsegments(
                            nodes, dl, p, m, n, radius, k, nsub_self
                        )
                    elif is_near_pair(nodes, dl, p, q):
                        av, bv = pair_integral_gauss(
                            nodes, dl, t, p, q, m, n, k, xg_near, wg_near
                        )
                    else:
                        av, bv = pair_integral_gauss(
                            nodes, dl, t, p, q, m, n, k, xg_far, wg_far
                        )

                    asum += av
                    bsum += bv

            zmat[mi, ni] = 1j * k * Z0 * asum - 1j * Z0 * bsum / k

    return zmat, dl, t, s


def rhs_delta_gap(ns, feed_node, voltage=1.0):
    if feed_node < 1 or feed_node > ns - 1:
        raise ValueError("feed_node must be an interior node.")

    rhs = np.zeros(ns - 1, dtype=complex)
    rhs[feed_node - 1] = voltage

    return rhs


def solve_wire(
    nodes,
    radius,
    freq,
    feed_node,
    voltage=1.0,
    ng_far=8,
    ng_near=18,
    nsub_self=24,
):
    zmat, dl, t, s = assemble_matrix(
        nodes=nodes,
        radius=radius,
        freq=freq,
        ng_far=ng_far,
        ng_near=ng_near,
        nsub_self=nsub_self,
    )

    ns = len(dl)
    rhs = rhs_delta_gap(ns, feed_node, voltage)
    sol = np.linalg.solve(zmat, rhs)

    current = np.zeros(ns + 1, dtype=complex)
    current[1:ns] = sol

    zin = voltage / current[feed_node]

    symmetry = np.linalg.norm(zmat - zmat.T, ord="fro") / np.linalg.norm(zmat, ord="fro")
    residual = np.linalg.norm(zmat @ sol - rhs) / np.linalg.norm(rhs)

    return {
        "Zin": zin,
        "current": current,
        "dl": dl,
        "t": t,
        "s": s,
        "zmat": zmat,
        "rhs": rhs,
        "symmetry": symmetry,
        "residual": residual,
    }
