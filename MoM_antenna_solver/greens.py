import numpy as np


def green_from_distance(r, k):
    r = np.maximum(r, 1.0e-15)
    return np.exp(-1j * k * r) / (4.0 * np.pi * r)


def psi_jin(delta, radius, k):
    if delta <= 0.0:
        raise ValueError("delta must be positive.")
    if radius <= 0.0:
        raise ValueError("radius must be positive")

    x = delta / radius

    return (
        delta / (2.0 * np.pi)
        * (
            np.log(x + np.sqrt(1.0 + x * x))
            - np.sqrt(1.0 + 1.0 / (x * x))
            + 1.0 / x
        )
        - 1j * k * delta * delta / (4.0 * np.pi)
    )
