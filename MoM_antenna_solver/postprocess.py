import numpy as np

from geometry import segment_data, segment_point


def reconstruct_current(nodes, current, samples_per_segment=12):
    dl, t, s = segment_data(nodes)
    ns = len(dl)

    vals = []
    points = []

    for p in range(ns):
        if p == ns - 1:
            xs = np.linspace(0.0, 1.0, samples_per_segment + 1)
        else:
            xs = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)

        for x in xs:
            vals.append((1.0 - x) * current[p] + x * current[p + 1])
            points.append(segment_point(nodes, p, x))

    return np.array(vals), np.array(points)


def reconstruct_bent_current(nodes, current, u_nodes, samples_per_segment=12):
    ns = len(current) - 1

    u = []
    vals = []

    for p in range(ns):
        if p == ns - 1:
            xs = np.linspace(0.0, 1.0, samples_per_segment + 1)
        else:
            xs = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)

        for x in xs:
            u.append((1.0 - x) * u_nodes[p] + x * u_nodes[p + 1])
            vals.append((1.0 - x) * current[p] + x * current[p + 1])

    return np.array(u), np.array(vals)


def relative_phase_deg(u, normalized_current):
    y = normalized_current.copy()
    mag = np.abs(y)

    phase = np.full(y.shape, np.nan, dtype=float)

    if np.nanmax(mag) <= 0.0:
        return phase

    mask = mag > 0.04 * np.nanmax(mag)

    if np.count_nonzero(mask) < 2:
        return phase

    ph = np.unwrap(np.angle(y[mask]))
    um = u[mask]

    ref = np.argmin(np.abs(um))
    ph = np.rad2deg(ph - ph[ref])

    while np.nanmedian(ph) > 180.0:
        ph -= 360.0

    while np.nanmedian(ph) < -180.0:
        ph += 360.0

    phase[mask] = ph

    return phase
