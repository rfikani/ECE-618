import math
import numpy as np
from scipy.spatial import Delaunay

from .models import make_mesh


def prune_mesh(points, triangles, boundary_nodes):
    used_nodes = np.unique(triangles.ravel())
    old_to_new = -np.ones(points.shape[0], dtype=int)
    old_to_new[used_nodes] = np.arange(len(used_nodes))
    new_points = points[used_nodes]
    new_triangles = old_to_new[triangles]
    boundary_nodes = np.unique(boundary_nodes)
    boundary_nodes = np.intersect1d(boundary_nodes, used_nodes)
    new_boundary_nodes = old_to_new[boundary_nodes]
    return new_points, new_triangles, new_boundary_nodes


def rectangular_mesh(a=2.2, b=1.0, nx=61, ny=29):
    x = np.linspace(0.0, a, nx)
    y = np.linspace(0.0, b, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    triangles = Delaunay(points).simplices.copy()
    px = points[:, 0]
    py = points[:, 1]
    tol = 1e-12
    boundary_nodes = np.where(
        (np.abs(px - 0.0) < tol)
        | (np.abs(px - a) < tol)
        | (np.abs(py - 0.0) < tol)
        | (np.abs(py - b) < tol)
    )[0]
    points, triangles, boundary_nodes = prune_mesh(points, triangles, boundary_nodes)
    return make_mesh(points, triangles, boundary_nodes, "rectangle", a=a, b=b)


def circular_mesh(radius=1.0, nr=26, ntheta=140):
    points = [[0.0, 0.0]]
    outer_ring = None
    for ir in range(1, nr + 1):
        r = radius * ir / nr
        nphi = max(12, int(round(ntheta * ir / nr)))
        ring_ids = []
        for iphi in range(nphi):
            phi = 2.0 * math.pi * iphi / nphi
            points.append([r * math.cos(phi), r * math.sin(phi)])
            ring_ids.append(len(points) - 1)
        outer_ring = ring_ids
    points = np.asarray(points, dtype=float)
    triangles_all = Delaunay(points).simplices.copy()
    centers = points[triangles_all].mean(axis=1)
    keep = np.sum(centers**2, axis=1) <= (1.000001 * radius) ** 2
    triangles = triangles_all[keep]
    boundary_nodes = np.asarray(outer_ring, dtype=int)
    points, triangles, boundary_nodes = prune_mesh(points, triangles, boundary_nodes)
    return make_mesh(points, triangles, boundary_nodes, "circle", radius=radius)



def double_ridged_mesh(a=2.2, b=1.0, wr=0.9, hr=0.32, nx=81, ny=49):
    if not (0.0 < wr < a):
        raise ValueError("need 0 < wr < a")
    if not (0.0 < hr < 0.5 * b):
        raise ValueError("need 0 < hr < b/2")

    s = 0.5 * (a - wr)
    x = np.linspace(0.0, a, nx)
    y = np.linspace(0.0, b, ny)
    x = np.unique(np.concatenate([x, [s, s + wr]]))
    y = np.unique(np.concatenate([y, [hr, b - hr]]))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])

    triangles_all = Delaunay(points).simplices.copy()
    centers = points[triangles_all].mean(axis=1)
    cx = centers[:, 0]
    cy = centers[:, 1]
    tol = 1e-12

    inside_lower_ridge = (
        (cx >= s - tol)
        & (cx <= s + wr + tol)
        & (cy >= 0.0 - tol)
        & (cy <= hr + tol)
    )
    inside_upper_ridge = (
        (cx >= s - tol)
        & (cx <= s + wr + tol)
        & (cy >= b - hr - tol)
        & (cy <= b + tol)
    )

    triangles = triangles_all[~(inside_lower_ridge | inside_upper_ridge)]
    px = points[:, 0]
    py = points[:, 1]

    outer_boundary = (
        (np.abs(px - 0.0) < tol)
        | (np.abs(px - a) < tol)
        | (np.abs(py - 0.0) < tol)
        | (np.abs(py - b) < tol)
    )

    lower_ridge_boundary = (
        (
            ((np.abs(px - s) < tol) | (np.abs(px - (s + wr)) < tol))
            & (py >= 0.0 - tol)
            & (py <= hr + tol)
        )
        | (
            ((np.abs(py - 0.0) < tol) | (np.abs(py - hr) < tol))
            & (px >= s - tol)
            & (px <= s + wr + tol)
        )
    )

    upper_ridge_boundary = (
        (
            ((np.abs(px - s) < tol) | (np.abs(px - (s + wr)) < tol))
            & (py >= b - hr - tol)
            & (py <= b + tol)
        )
        | (
            ((np.abs(py - (b - hr)) < tol) | (np.abs(py - b) < tol))
            & (px >= s - tol)
            & (px <= s + wr + tol)
        )
    )

    boundary_nodes = np.where(outer_boundary | lower_ridge_boundary | upper_ridge_boundary)[0]
    points, triangles, boundary_nodes = prune_mesh(points, triangles, boundary_nodes)

    return make_mesh(
        points,
        triangles,
        boundary_nodes,
        "double_ridged",
        a=a,
        b=b,
        wr=wr,
        hr=hr,
        s=s,
        g=b - 2.0 * hr,
    )
