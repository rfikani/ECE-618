from .solver import solve_mode_family
from .matching import match_rectangular_modes, match_circular_modes, extract_ridged_modes


def solve_case(mesh):
    te_vals, te_vecs, _, M_te = solve_mode_family(mesh, "TE", kmax=30)
    tm_vals, tm_vecs, _, M_tm = solve_mode_family(mesh, "TM", kmax=30)

    if mesh["kind"] == "rectangle":
        te_modes = match_rectangular_modes(mesh, "TE", te_vals, te_vecs, M_te, n_modes=3)
        tm_modes = match_rectangular_modes(mesh, "TM", tm_vals, tm_vecs, M_tm, n_modes=3)
    elif mesh["kind"] == "circle":
        te_modes = match_circular_modes(mesh, "TE", te_vals, te_vecs, M_te, n_modes=3)
        tm_modes = match_circular_modes(mesh, "TM", tm_vals, tm_vecs, M_tm, n_modes=3)
    elif mesh["kind"] == "double_ridged":
        te_modes = extract_ridged_modes(mesh, "TE", te_vals, te_vecs, M_te, n_modes=3)
        tm_modes = extract_ridged_modes(mesh, "TM", tm_vals, tm_vecs, M_tm, n_modes=3)
    else:
        raise ValueError(f"unknown mesh kind: {mesh['kind']}")

    if len(te_modes) < 3 or len(tm_modes) < 3:
        raise RuntimeError("couldn't get 3 clean TE and TM modes from this mesh")
    return te_modes, tm_modes
