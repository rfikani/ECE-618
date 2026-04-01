import matplotlib.pyplot as plt

from waveguide_fem.mesh import rectangular_mesh, circular_mesh, double_ridged_mesh
from waveguide_fem.pipeline import solve_case
from waveguide_fem.plotting import plot_mode_set, plot_dispersion


def main():
    plt.ion()

    rect = rectangular_mesh(a=2.2, b=1.0, nx=61, ny=29)
    circ = circular_mesh(radius=1.0, nr=26, ntheta=140)
    ridged = double_ridged_mesh(a=2.2, b=1.0, wr=0.9, hr=0.32, nx=81, ny=49)

    rect_te, rect_tm = solve_case(rect)
    circ_te, circ_tm = solve_case(circ)
    ridged_te, ridged_tm = solve_case(ridged)

    figs = []
    figs.append(plot_mode_set(rect, rect_te, rect_tm, "rect_waveguide_modes_fixed.png"))
    figs.append(plot_mode_set(circ, circ_te, circ_tm, "circ_waveguide_modes_fixed.png"))
    figs.append(plot_mode_set(ridged, ridged_te, ridged_tm, "double_ridged_waveguide_modes_fixed.png"))

    figs.append(
        plot_dispersion(
            rect,
            rect_te,
            rect_tm,
            "rect_waveguide_dispersion_fixed.png",
            "Rectangular waveguide dispersion: FEM + analytical",
        )
    )
    figs.append(
        plot_dispersion(
            circ,
            circ_te,
            circ_tm,
            "circ_waveguide_dispersion_fixed.png",
            "Circular waveguide dispersion: FEM + analytical",
        )
    )
    figs.append(
        plot_dispersion(
            ridged,
            ridged_te,
            ridged_tm,
            "double_ridged_waveguide_dispersion_fixed.png",
            "Double-ridged waveguide dispersion: FEM only",
        )
    )

    plt.show(block=True)
    for fig in figs:
        plt.close(fig)


if __name__ == "__main__":
    main()
