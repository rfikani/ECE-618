import math
import numpy as np
import scipy.sparse as sp

def local_element_matrices(xy):
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]
    det_j = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = abs(det_j) / 2.0
    if area <= 0.0:
        raise ValueError("bad triangle with zero area")

    b = np.asarray([y2 - y3, y3 - y1, y1 - y2], dtype=float)
    c = np.asarray([x3 - x2, x1 - x3, x2 - x1], dtype=float)

    ke = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
    me = (area / 12.0) * np.asarray(
        [
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=float,
    )
    return ke, me

def assemble_system(mesh):
    points = mesh["points"]
    triangles = mesh["triangles"]
    npts = points.shape[0]

    rows = []
    cols = []
    kdata = []
    mdata = []

    for tri in triangles:
        ke, me = local_element_matrices(points[tri])
        for i_local in range(3):
            i_global = tri[i_local]
            for j_local in range(3):
                j_global = tri[j_local]
                rows.append(i_global)
                cols.append(j_global)
                kdata.append(ke[i_local, j_local])
                mdata.append(me[i_local, j_local])

    K = sp.coo_matrix((kdata, (rows, cols)), shape=(npts, npts)).tocsr()
    M = sp.coo_matrix((mdata, (rows, cols)), shape=(npts, npts)).tocsr()
    return K, M

def m_norm(v, M):
    return math.sqrt(abs(float(np.real(v @ (M @ v)))))

def normalize_mode_vector(v, M):
    v = np.array(v, dtype=float, copy=True)
    nrm = m_norm(v, M)
    if nrm > 0.0:
        v /= nrm
    idx = np.argmax(np.abs(v))
    if v[idx] < 0.0:
        v *= -1.0
    return v
