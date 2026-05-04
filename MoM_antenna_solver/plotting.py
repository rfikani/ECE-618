import numpy as np
import matplotlib.pyplot as plt

from postprocess import reconstruct_bent_current, reconstruct_current, relative_phase_deg


def setup_plots():
    plt.rcParams.update({
        "figure.figsize": (3.45, 2.35),
        "figure.dpi": 150,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "legend.fontsize": 6.2,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.15,
        "lines.markersize": 3.4,
        "grid.linewidth": 0.35,
        "grid.alpha": 0.25,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.borderpad": 0.28,
        "legend.handlelength": 1.95,
        "legend.labelspacing": 0.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def finish_axis(ax):
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.16)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)


def finish_figure(fig):
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.22, top=0.96)
    return fig


def plot_straight_current(straight_results):
    fig, ax = plt.subplots()
    colors = ["black", "#1f4e79", "#7f7f7f", "#b60020"]

    straight_results = sorted(straight_results, key=lambda item: item["Ns"])

    for idx, item in enumerate(straight_results):
        vals, points = reconstruct_current(item["nodes"], item["current"])
        feed = item["current"][item["feed_node"]]
        lam = item["wavelength"]

        ax.plot(
            points[:, 2] / lam,
            np.abs(vals / feed),
            color=colors[idx % len(colors)],
            label=rf"$N_s={item['Ns']}$",
        )

    zref = np.linspace(-0.25, 0.25, 500)

    ax.plot(
        zref,
        np.abs(np.cos(2.0 * np.pi * zref)),
        "--",
        color="#606060",
        linewidth=0.95,
        label=r"$|\cos(k_0 z)|$",
    )

    ax.set_xlabel(r"$z/\lambda$")
    ax.set_ylabel(r"$|I(z)/I_\mathrm{feed}|$")
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(0.0, 1.08)
    ax.legend(loc="lower center")
    finish_axis(ax)

    return finish_figure(fig)


def plot_straight_impedance(straight_results):
    straight_results = sorted(straight_results, key=lambda item: item["Ns"])

    ns = np.array([item["Ns"] for item in straight_results])
    zr = np.array([item["Zin"].real for item in straight_results])
    zi = np.array([item["Zin"].imag for item in straight_results])

    fig, ax = plt.subplots()

    ax.plot(ns, zr, "-o", color="black", label=r"$\Re\{Z_\mathrm{in}\}$")
    ax.plot(ns, zi, "-s", color="#b60020", label=r"$\Im\{Z_\mathrm{in}\}$")

    ax.axhline(73.0, color="black", linestyle="--", linewidth=0.9, label=r"$73~\Omega$")
    ax.axhline(42.5, color="#b60020", linestyle="--", linewidth=0.9, label=r"$42.5~\Omega$")

    ax.set_xlabel(r"Number of segments $N_s$")
    ax.set_ylabel(r"Input impedance [$\Omega$]")
    ax.set_xlim(ns.min() - 8, ns.max() + 8)
    ax.legend(loc="center right")
    finish_axis(ax)

    return finish_figure(fig)


def plot_straight_error(straight_results):
    straight_results = sorted(straight_results, key=lambda item: item["Ns"])

    ns = np.array([item["Ns"] for item in straight_results], dtype=float)
    err = np.array([item["relative_error"] for item in straight_results], dtype=float)

    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    ax.plot(
        ns,
        err,
        "-o",
        color="#1f4e79",
        linewidth=1.25,
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=0.85,
    )

    for xval, yval in zip(ns, err):
        ax.annotate(
            f"{100.0 * yval:.1f}%",
            xy=(xval, yval),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )

    ax.set_xlabel(r"Number of segments $N_s$")
    ax.set_ylabel(r"Relative impedance error")
    ax.set_xlim(ns.min() - 8, ns.max() + 8)

    pad = 0.08 * (err.max() - err.min()) if err.max() > err.min() else 0.005
    ax.set_ylim(err.min() - pad, err.max() + 1.8 * pad)

    ax.set_xticks(ns.astype(int))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    ax.text(
        0.04,
        0.08,
        r"$Z_\mathrm{ref}=73+j42.5~\Omega$",
        transform=ax.transAxes,
        fontsize=6.8,
        bbox=dict(facecolor="white", edgecolor="0.75", linewidth=0.5, alpha=0.92),
    )

    finish_axis(ax)

    return finish_figure(fig)


def plot_bent_current(bent_results, u_nodes):
    fig, ax = plt.subplots()
    colors = ["black", "#1f4e79", "#b60020"]

    bent_results = sorted(bent_results, key=lambda item: item["frequency_MHz"])

    for idx, item in enumerate(bent_results):
        u, vals = reconstruct_bent_current(item["nodes"], item["current"], u_nodes)
        feed = item["current"][item["feed_node"]]

        ax.plot(
            100.0 * u,
            np.abs(vals / feed),
            color=colors[idx % len(colors)],
            label=rf"${item['frequency_MHz']:.0f}~\mathrm{{MHz}}$",
        )

    ax.axvline(0.0, color="#505050", linestyle="--", linewidth=0.9)
    ax.set_xlabel(r"Signed position from feed $u$ [cm]")
    ax.set_ylabel(r"$|I(u)/I_\mathrm{feed}|$")
    ax.set_xlim(-100.0, 100.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper center")
    finish_axis(ax)

    return finish_figure(fig)


def plot_bent_phase(bent_results, u_nodes):
    fig, ax = plt.subplots()
    colors = ["black", "#1f4e79", "#b60020"]

    bent_results = sorted(bent_results, key=lambda item: item["frequency_MHz"])

    for idx, item in enumerate(bent_results):
        u, vals = reconstruct_bent_current(item["nodes"], item["current"], u_nodes)
        feed = item["current"][item["feed_node"]]
        phase = relative_phase_deg(u, vals / feed)

        ax.plot(
            100.0 * u,
            phase,
            color=colors[idx % len(colors)],
            label=rf"${item['frequency_MHz']:.0f}~\mathrm{{MHz}}$",
        )

    ax.axvline(0.0, color="#505050", linestyle="--", linewidth=0.9)
    ax.set_xlabel(r"Signed position from feed $u$ [cm]")
    ax.set_ylabel(r"$\angle[I(u)/I_\mathrm{feed}]$ [deg]")
    ax.set_xlim(-100.0, 100.0)
    ax.legend(loc="lower left")
    finish_axis(ax)

    return finish_figure(fig)


def plot_bent_impedance(bent_results):
    bent_results = sorted(bent_results, key=lambda item: item["frequency_MHz"])

    f = np.array([item["frequency_MHz"] for item in bent_results])
    zr = np.array([item["Zin"].real for item in bent_results])
    zi = np.array([item["Zin"].imag for item in bent_results])

    fig, ax = plt.subplots()

    ax.plot(f, zr, "-o", color="black", label=r"$\Re\{Z_\mathrm{in}\}$")
    ax.plot(f, zi, "-s", color="#b60020", label=r"$\Im\{Z_\mathrm{in}\}$")

    ax.plot([75.0], [41.2], "D", color="black", label=r"Ref. $\Re$")
    ax.plot([75.0], [-6.6], "D", color="#b60020", label=r"Ref. $\Im$")

    ax.set_xlabel(r"Frequency [MHz]")
    ax.set_ylabel(r"Input impedance [$\Omega$]")
    ax.set_xlim(68.0, 232.0)
    ax.legend(loc="lower left")
    finish_axis(ax)

    return finish_figure(fig)