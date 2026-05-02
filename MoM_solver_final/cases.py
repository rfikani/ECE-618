from constants import c0
from mesh import bent_nodes, straight_nodes
from solver import solve_wire


def run_straight():
    results = []

    ns_values = [20, 40, 80, 160]

    wavelength = 1.0
    radius = 0.001 * wavelength
    freq = c0 / wavelength
    zref = 73.0 + 1j * 42.5

    for ns in ns_values:
        if ns % 2 != 0:
            raise ValueError(
                f"Straight dipole requires even Ns so the feed is centered. Got Ns={ns}."
            )

        print(f"solving straight dipole: Ns={ns}")

        nodes = straight_nodes(ns, wavelength)
        feed_node = ns // 2

        sol = solve_wire(
            nodes=nodes,
            radius=radius,
            freq=freq,
            feed_node=feed_node,
            voltage=1.0,
            ng_far=8,
            ng_near=18,
            nsub_self=24,
        )

        zin = sol["Zin"]
        rel = abs(zin - zref) / abs(zref)

        print(
            f"  Zin = {zin.real:.6f} {zin.imag:+.6f}j ohm, "
            f"rel err = {rel:.6e}, "
            f"sym = {sol['symmetry']:.3e}, "
            f"res = {sol['residual']:.3e}"
        )

        results.append({
            "Ns": ns,
            "Zin": zin,
            "relative_error": rel,
            "symmetry": sol["symmetry"],
            "residual": sol["residual"],
            "nodes": nodes,
            "current": sol["current"],
            "feed_node": feed_node,
            "frequency_Hz": freq,
            "wavelength": wavelength,
            "zmat": sol["zmat"],
        })

    return results


def run_bent():
    results = []

    n1 = 80
    n2 = 80

    radius = 0.005
    feed_node = n1
    freqs = [75.0e6, 150.0e6, 225.0e6]
    zref_75 = 41.2 - 1j * 6.6

    nodes, u_nodes = bent_nodes(n1=n1, n2=n2, l1=1.0, l2=1.0)

    for freq in freqs:
        print(
            f"solving bent wire: "
            f"Ns={n1 + n2}, Nu={n1 + n2 - 1}, f={freq / 1.0e6:.0f} MHz"
        )

        sol = solve_wire(
            nodes=nodes,
            radius=radius,
            freq=freq,
            feed_node=feed_node,
            voltage=1.0,
            ng_far=8,
            ng_near=20,
            nsub_self=28,
        )

        zin = sol["Zin"]
        rel = abs(zin - zref_75) / abs(zref_75)

        print(
            f"  Zin = {zin.real:.6f} {zin.imag:+.6f}j ohm, "
            f"rel err vs 75 MHz ref = {rel:.6e}, "
            f"sym = {sol['symmetry']:.3e}, "
            f"res = {sol['residual']:.3e}"
        )

        results.append({
            "frequency_MHz": freq / 1.0e6,
            "frequency_Hz": freq,
            "Zin": zin,
            "relative_error_vs_75MHz_reference": rel,
            "symmetry": sol["symmetry"],
            "residual": sol["residual"],
            "nodes": nodes,
            "current": sol["current"],
            "feed_node": feed_node,
            "zmat": sol["zmat"],
        })

    return results, u_nodes