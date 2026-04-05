import numpy as np
import matplotlib.pyplot as plt


def plot_field(Ezval, dx, dy, title):
    plt.figure(figsize=(7, 5))
    plt.imshow(Ezval, origin="lower", extent=[0, Ezval.shape[1] * dy, 0, Ezval.shape[0] * dx], aspect="auto")
    cbee = plt.colorbar()
    cbee.set_label("Ez (V/m)")
    plt.xlabel("y (m)")
    plt.ylabel("x (m)")
    plt.title(title)
    plt.tight_layout()

def plot_diffraction(x, I0ne, I2):
    plt.figure(figsize=(7, 5))
    plt.plot(x, I0ne, linewidth=1.6, label="Single slot")
    plt.plot(x, I2, linewidth=1.6, label="Two slots")
    plt.xlabel("x (m)")
    plt.ylabel("|Ez|^2 (V^2/m^2)")
    plt.title("Diffraction comparison")
    plt.legend()
    plt.tight_layout()


def plot_scattering(theta, Ic, Ir):
    plt.figure(figsize=(7, 5))
    plt.plot(theta * 180.0 / np.pi, Ic, linewidth=1.6, label="Circular PEC")
    plt.plot(theta * 180.0 / np.pi, Ir, linewidth=1.6, label="Rectangular PEC")
    plt.xlabel("Angle (deg)")
    plt.ylabel("|Ez|^2 (V^2/m^2)")
    plt.title("Angular scattering")
    plt.legend()
    plt.tight_layout()
