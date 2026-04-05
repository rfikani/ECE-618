import numpy as np

epszero = 8.854187817e-12
muzero = 4.0 * np.pi * 1e-7
czero = 1.0 / np.sqrt(muzero * epszero)
freqzero = 2.4e9
S = 0.95
src_amp = 1e-3
pmlpow = 3
pmlreflect = 1e-8
