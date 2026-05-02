import matplotlib.pyplot as plt

from cases import run_bent, run_straight
from plotting import (
    plot_bent_current,
    plot_bent_impedance,
    plot_bent_phase,
    plot_straight_current,
    plot_straight_error,
    plot_straight_impedance,
    plot_z_matrix_structure,
    setup_plots,
)


def main():
    setup_plots()

    straight_results = run_straight()
    bent_results, u_nodes = run_bent()

    plot_straight_current(straight_results)
    plot_straight_impedance(straight_results)
    plot_straight_error(straight_results)

    plot_bent_current(bent_results, u_nodes)
    plot_bent_phase(bent_results, u_nodes)
    plot_bent_impedance(bent_results)

    plot_z_matrix_structure(straight_results, bent_results)

    plt.show()


if __name__ == "__main__":
    main()
