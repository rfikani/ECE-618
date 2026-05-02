# Thin-Wire Antenna EFIE MoM Solver


## Files

- `constants.py` — physical constants: `mu0`, `eps0`, `c0`, and `Z0`.
- `mesh.py` — wire geometry and segment utilities, including straight and bent-wire node generation.
- `basis.py` — piecewise-linear basis-function values and derivatives.
- `greens.py` — scalar Green's function and the self-term function `psi_jin`.
- `integrals.py` — Gaussian and self-subsegment pair integrals used in the matrix assembly.
- `solver.py` — EFIE impedance matrix assembly, delta-gap RHS, and linear solve.
- `postprocess.py` — current reconstruction and phase postprocessing.
- `plotting.py` — all plotting functions, including current, impedance, error, and matrix-magnitude plots.
- `cases.py` — straight-dipole and bent-wire simulation cases.
- `main.py` — main driver script.



```bash
pip3 install numpy matplotlib
```

## Run

From inside this folder, run:

```bash
python3 main.py
```

## Notes

The matrix-magnitude plots are computed directly from the assembled impedance matrices stored in `sol["zmat"]`:

```python
z_straight = np.log10(np.abs(straight_item["zmat"]) + 1.0e-16)
z_bent = np.log10(np.abs(bent_item["zmat"]) + 1.0e-16)
```




