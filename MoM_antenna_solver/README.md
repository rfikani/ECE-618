# Thin-Wire EFIE MoM Solver


## Files

- `constants.py` — physical constants
- `geometry.py` — wire geometry and segment utilities, including straight and bent-wire node generation.
- `basis.py` — piecewise-linear basis-function values and derivatives.
- `greens.py` — scalar Green's function and the self-term function 
- `integrals.py` — Gaussian and self-subsegment pair integrals used in the matrix assembly.
- `solver.py` — EFIE impedance matrix assembly, delta-gap RHS, and linear solve
- `postprocess.py` — current reconstruction and phase postprocessing.
- `plotting.py` — all plotting functions, including current, impedance, error
- `cases.py` —straight-dipole and bent-wire simulation cases.
- `main.py` — main driver script.

## Requirements

Install NumPy and Matplotlib:

```bash
pip3 install numpy matplotlib
```

## Run

From inside this folder, run:

```bash
python3 main.py
```


