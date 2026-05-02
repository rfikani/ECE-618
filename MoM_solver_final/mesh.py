import numpy as np


def segment_data(nodes):
    d = nodes[1:] - nodes[:-1]
    dl = np.linalg.norm(d, axis=1)

    if np.any(dl <= 0.0):
        raise ValueError("Zero-length segment detected.")

    t = d / dl[:, None]

    s = np.zeros(nodes.shape[0])
    s[1:] = np.cumsum(dl)

    return dl, t, s


def segment_point(nodes, seg, x):
    return nodes[seg] + x * (nodes[seg + 1] - nodes[seg])


def straight_nodes(ns, wavelength=1.0):
    if ns % 2 != 0:
        raise ValueError("Use an even number of segments so the feed is exactly at the center node.")

    length = 0.5 * wavelength
    z = np.linspace(-0.5 * length, 0.5 * length, ns + 1)

    nodes = np.zeros((ns + 1, 3))
    nodes[:, 2] = z

    return nodes


def bent_nodes(n1=50, n2=49, l1=1.0, l2=1.0):
    if n1 < 1 or n2 < 1:
        raise ValueError("Both bent-wire arms need at least one segment.")

    u_left = np.linspace(-l1, 0.0, n1 + 1)
    u_right = np.linspace(l2 / n2, l2, n2)
    u = np.concatenate([u_left, u_right])

    nodes = np.zeros((n1 + n2 + 1, 3))

    for i, value in enumerate(u):
        if value <= 0.0:
            nodes[i, 2] = -value
        else:
            nodes[i, 0] = value

    return nodes, u
