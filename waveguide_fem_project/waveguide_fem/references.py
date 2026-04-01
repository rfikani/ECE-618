import math
from scipy.special import jn_zeros, jnp_zeros, jv
import numpy as np


def rectangular_mode_refs(a, b, family, max_m=8, max_n=8):
    refs = []
    for m in range(max_m + 1):
        for n in range(max_n + 1):
            if family == "TM":
                if m == 0 or n == 0:
                    continue
                kc = math.pi * math.sqrt((m / a) ** 2 + (n / b) ** 2)
                refs.append({"label": f"TM{m}{n}", "m": m, "n": n, "kc": kc})
            else:
                if m == 0 and n == 0:
                    continue
                kc = math.pi * math.sqrt((m / a) ** 2 + (n / b) ** 2)
                refs.append({"label": f"TE{m}{n}", "m": m, "n": n, "kc": kc})
    refs.sort(key=lambda x: x["kc"])
    return refs



def circular_mode_refs(radius, family, mmax=8, nmax=5):
    refs = []
    for m in range(mmax + 1):
        for n in range(1, nmax + 1):
            if family == "TM":
                chi = float(jn_zeros(m, n)[-1])
            else:
                chi = float(jnp_zeros(m, n)[-1])
            mult = 1 if m == 0 else 2
            refs.append(
                {
                    "label": f"{family}{m}{n}",
                    "m": m,
                    "n": n,
                    "chi": chi,
                    "kc": chi / radius,
                    "mult": mult,
                }
            )
    refs.sort(key=lambda x: x["kc"])
    return refs



def rectangular_target_mode(mesh, family, m, n):
    a = mesh["params"]["a"]
    b = mesh["params"]["b"]
    x = mesh["points"][:, 0]
    y = mesh["points"][:, 1]
    if family == "TM":
        return np.sin(m * math.pi * x / a) * np.sin(n * math.pi * y / b)
    return np.cos(m * math.pi * x / a) * np.cos(n * math.pi * y / b)



def circular_target_mode(mesh, m, chi, use_sin=False):
    radius = mesh["params"]["radius"]
    x = mesh["points"][:, 0]
    y = mesh["points"][:, 1]
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    radial = jv(m, chi * r / radius)
    angular = np.sin(m * phi) if use_sin else np.cos(m * phi)
    return radial * angular



def exact_kc_for_mode(mesh, mode):
    if mesh["kind"] == "rectangle":
        refs = rectangular_mode_refs(mesh["params"]["a"], mesh["params"]["b"], mode["family"], max_m=12, max_n=12)
        for ref in refs:
            if ref["label"] == mode["label"]:
                return ref["kc"]
        return None

    if mesh["kind"] == "circle":
        refs = circular_mode_refs(mesh["params"]["radius"], mode["family"], mmax=12, nmax=8)
        for ref in refs:
            if ref["label"] == mode["label"]:
                return ref["kc"]
        return None
    return None
