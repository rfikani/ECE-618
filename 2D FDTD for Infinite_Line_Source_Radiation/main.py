import matplotlib.pyplot as plt
from cases import run_validation_case, run_slot_cases, run_scattering_cases
from plotting import plot_field, plot_diffraction, plot_scattering


validation = run_validation_case()
plot_field(validation["Ezval"], validation["grid"]["dx"], validation["grid"]["dy"], "Validation field")

slots = run_slot_cases()
plot_field(slots["single"]["Ez"], slots["grid"]["dx"], slots["grid"]["dy"], "Single slot")
plot_field(slots["double"]["Ez"], slots["grid"]["dx"], slots["grid"]["dy"], "Two slots")
plot_diffraction(slots["x"], slots["I0ne"], slots["I2"])

scattering = run_scattering_cases()
plot_field(scattering["circular"]["Ez"], scattering["grid"]["dx"], scattering["grid"]["dy"], "PEC circle")
plot_field(scattering["rectangular"]["Ez"], scattering["grid"]["dx"], scattering["grid"]["dy"], "PEC rectangle")
plot_scattering(scattering["theta"], scattering["Ic"], scattering["Ir"])

plt.show()
