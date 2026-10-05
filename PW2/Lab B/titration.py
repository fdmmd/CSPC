"""Find the equivalence point of a titration.

Read titration.csv (volume base, pH), compute the slope d(pH)/dV,
and find the volume where the slope is largest -> equivalence point.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---- Read data ----
V, pH = np.loadtxt("titration.csv", delimiter=",", skiprows=1, unpack=True)

# ---- Slope ----
slope = np.gradient(pH, V)

# ---- Equivalence point: max slope ----
i_max = np.argmax(slope)
V_eq = V[i_max]
print(f"Equivalence point at V = {V_eq:.2f} mL")

# ---- Plot ----
fig, (ax_ph, ax_slope) = plt.subplots(1, 2, figsize=(12, 5))

# pH curve
ax_ph.plot(V, pH, "-o", markersize=3, color="C0")
ax_ph.axvline(V_eq, color="red", linestyle=":", label=f"V_eq = {V_eq:.2f} mL")
ax_ph.set_xlabel("volume base V (mL)")
ax_ph.set_ylabel("pH")
ax_ph.set_title("Titration curve")
ax_ph.legend()
ax_ph.grid(True, alpha=0.3)

# Slope
ax_slope.plot(V, slope, "-o", markersize=3, color="C1")
ax_slope.axvline(V_eq, color="red", linestyle=":", label=f"V_eq = {V_eq:.2f} mL")
ax_slope.set_xlabel("volume base V (mL)")
ax_slope.set_ylabel("d(pH)/dV (per mL)")
ax_slope.set_title("Slope — peaks at equivalence point")
ax_slope.legend()
ax_slope.grid(True, alpha=0.3)

fig.suptitle("Titration: pH curve and its slope")
fig.tight_layout()
fig.savefig("titration.png", dpi=150)
print("Saved titration.png")