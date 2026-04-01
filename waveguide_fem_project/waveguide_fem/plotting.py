import math
import numpy as np
import matplotlib.pyplot as plt

from .constants import cnau
from .references import exact_kc_for_mode
from .utils import beta_from_kc


def draw_ridges(ax, mesh):
    if mesh["kind"] != "double_ridged":
        return
    b = mesh["params"]["b"]
    s = mesh["params"]["s"]
    wr = mesh["params"]["wr"]
    hr = mesh["params"]["hr"]
    ax.plot([s, s + wr, s + wr, s, s], [0, 0, hr, hr, 0], "k-", lw=1.2)
    ax.plot([s, s + wr, s + wr, s, s], [b - hr, b - hr, b, b, b - hr], "k-", lw=1.2)



def plot_mode_set(mesh, te_modes, tm_modes, filename):
    points = mesh["points"]
    triangles = mesh["triangles"]
    x = points[:, 0]
    y = points[:, 1]

    fig, ax = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)

    for j in range(3):
        mode = te_modes[j]
        a = ax[0, j]
        cf = a.tricontourf(x, y, triangles, mode["vec"], levels=40)
        a.triplot(x, y, triangles, color="k", lw=0.12, alpha=0.18)
        draw_ridges(a, mesh)
        a.set_title(f"{mode['label']}  kc={mode['kc']:.4f}")
        a.set_aspect("equal")
        cb = fig.colorbar(cf, ax=a)
        cb.set_label("Normalised $H_z$")

    for j in range(3):
        mode = tm_modes[j]
        a = ax[1, j]
        cf = a.tricontourf(x, y, triangles, mode["vec"], levels=40)
        a.triplot(x, y, triangles, color="k", lw=0.12, alpha=0.18)
        draw_ridges(a, mesh)
        a.set_title(f"{mode['label']}  kc={mode['kc']:.4f}")
        a.set_aspect("equal")
        cb = fig.colorbar(cf, ax=a)
        cb.set_label("Normalised $E_z$")

    ax[0, 0].set_ylabel("TE longitudinal field $H_z$")
    ax[1, 0].set_ylabel("TM longitudinal field $E_z$")
    fig.suptitle(f"{mesh['kind'].replace('_', ' ').capitalize()} waveguide FEM modes")
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    fig.canvas.draw()
    return fig



def plot_dispersion(mesh, te_modes, tm_modes, filename, title):
    all_kc = [m["kc"] for m in te_modes + tm_modes]
    fmin = 0.02 * cnau * min(all_kc) / (2.0 * math.pi)
    fmax = 1.8 * cnau * max(all_kc) / (2.0 * math.pi)

    freqs = np.linspace(fmin, fmax, 500)
    k0 = 2.0 * math.pi * freqs / cnau

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    for mode in te_modes:
        beta = beta_from_kc(k0, mode["kc"])
        ax.plot(freqs / 1e9, np.real(beta / k0), label=f"{mode['label']} FEM")
        kc_exact = exact_kc_for_mode(mesh, mode)
        if kc_exact is not None:
            beta_exact = beta_from_kc(k0, kc_exact)
            ax.plot(
                freqs / 1e9,
                np.real(beta_exact / k0),
                ":",
                lw=2.0,
                color="black",
                alpha=0.85,
                label=f"{mode['label']} exact",
            )

    for mode in tm_modes:
        beta = beta_from_kc(k0, mode["kc"])
        ax.plot(freqs / 1e9, np.real(beta / k0), "--", label=f"{mode['label']} FEM")
        kc_exact = exact_kc_for_mode(mesh, mode)
        if kc_exact is not None:
            beta_exact = beta_from_kc(k0, kc_exact)
            ax.plot(
                freqs / 1e9,
                np.real(beta_exact / k0),
                ":",
                lw=2.0,
                color="black",
                alpha=0.85,
                label=f"{mode['label']} exact",
            )

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(r"Re$(\beta/k_0)$")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    ax.set_title(title)
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    fig.canvas.draw()
    return fig
