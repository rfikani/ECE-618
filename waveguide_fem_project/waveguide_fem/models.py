import numpy as np


def make_mesh(points, triangles, boundary_nodes, kind, **params):
    return {
        "points": points,
        "triangles": triangles,
        "boundary_nodes": boundary_nodes,
        "kind": kind,
        "params": params,
    }


def make_mode(family, label, kc, eigval, vec, mult=1, rel_err=np.nan):
    return {
        "family": family,
        "label": label,
        "kc": kc,
        "eigval": eigval,
        "vec": vec,
        "mult": mult,
        "rel_err": rel_err,
    }
