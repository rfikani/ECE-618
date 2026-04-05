import numpy as np
from constants import src_amp, freqzero

def drive_f(t, dx, dy, taw):

    J0 = src_amp / (dx * dy)
    return J0 * (1.0 - np.exp(-t / taw)) * np.sin(2.0 * np.pi * freqzero * t)
