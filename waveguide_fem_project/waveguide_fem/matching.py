import math
import numpy as np

from .models import make_mode
from .references import (
    rectangular_mode_refs,
    circular_mode_refs,
    rectangular_target_mode,
    circular_target_mode,
)
from .utils import (
    group_eigenvalues,
    mass_orthonormalize,
    project_onto_subspace,
    pick_representative_mode,
)
from .fem import normalize_mode_vector


def match_rectangular_modes(mesh, family, eigvals, eigvecs, M, n_modes=3):
    refs = rectangular_mode_refs(mesh["params"]["a"], mesh["params"]["b"], family)
    groups = group_eigenvalues(eigvals)
    used_groups = set()
    modes = []

    for ref in refs:
        kc_ref = ref["kc"]
        best_group = None
        best_err = 1e99
        for ig, (i0, i1) in enumerate(groups):
            if ig in used_groups:
                continue
            kc_here = math.sqrt(float(np.mean(eigvals[i0:i1])))
            err = abs(kc_here - kc_ref) / kc_ref
            if err < best_err:
                best_err = err
                best_group = ig
        if best_group is None or best_err > 0.08:
            continue
        i0, i1 = groups[best_group]
        target = rectangular_target_mode(mesh, family, ref["m"], ref["n"])
        subspace = mass_orthonormalize(eigvecs[:, i0:i1], M)
        vec = project_onto_subspace(subspace, target, M)
        modes.append(
            make_mode(
                family=family,
                label=ref["label"],
                kc=math.sqrt(float(np.mean(eigvals[i0:i1]))),
                eigval=float(np.mean(eigvals[i0:i1])),
                vec=vec,
                mult=i1 - i0,
                rel_err=best_err,
            )
        )
        used_groups.add(best_group)
        if len(modes) >= n_modes:
            break
    return modes

def match_circular_modes(mesh, family, eigvals, eigvecs, M, n_modes=3):
    refs = circular_mode_refs(mesh["params"]["radius"], family)
    groups = group_eigenvalues(eigvals)
    used_groups = set()
    modes = []

    for ref in refs:
        kc_ref = ref["kc"]
        best_group = None
        best_err = 1e99
        for ig, (i0, i1) in enumerate(groups):
            if ig in used_groups:
                continue
            kc_here = math.sqrt(float(np.mean(eigvals[i0:i1])))
            err = abs(kc_here - kc_ref) / kc_ref
            if err < best_err:
                best_err = err
                best_group = ig
        if best_group is None or best_err > 0.08:
            continue
        i0, i1 = groups[best_group]
        subspace = mass_orthonormalize(eigvecs[:, i0:i1], M)
        if ref["m"] == 0:
            vec = normalize_mode_vector(subspace[:, 0], M)
        else:
            target = circular_target_mode(mesh, ref["m"], ref["chi"], use_sin=False)
            vec = project_onto_subspace(subspace, target, M)
        modes.append(
            make_mode(
                family=family,
                label=ref["label"],
                kc=math.sqrt(float(np.mean(eigvals[i0:i1]))),
                eigval=float(np.mean(eigvals[i0:i1])),
                vec=vec,
                mult=i1 - i0,
                rel_err=best_err,
            )
        )
        used_groups.add(best_group)
        if len(modes) >= n_modes:
            break
    return modes

def extract_ridged_modes(mesh, family, eigvals, eigvecs, M, n_modes=3, rel_tol=5e-3):
    groups = group_eigenvalues(eigvals, rel_tol=rel_tol)
    modes = []
    for k, (i0, i1) in enumerate(groups, start=1):
        lam = float(np.mean(eigvals[i0:i1]))
        vec = pick_representative_mode(eigvecs[:, i0:i1], M)
        modes.append(
            make_mode(
                family=family,
                label=f"{family} mode {k}",
                kc=math.sqrt(lam),
                eigval=lam,
                vec=vec,
                mult=i1 - i0,
                rel_err=np.nan,
            )
        )
        if len(modes) >= n_modes:
            break
    return modes
