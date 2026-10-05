"""Fit a first-order reaction rate constant.

Model: C(t) = C0 * exp(-k * t)
Find k that minimises sum((C_measured - C0*exp(-k*t))**2).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---- Read data ----
t, C = np.loadtxt("kinetics.csv", delimiter=",", skiprows=1, unpack=True)
C0 = C[0]
print(f"C0 = {C0}")

# ---- Error function ----
def total_error(k):
    model = C0 * np.exp(-k * t)
    return np.sum((C - model)**2)

# ---- Minimise ----
result = minimize(total_error, x0=0.5, method="SLSQP", bounds=[(0, 5)])
k_fit = result.x[0]
print(f"Fitted k = {k_fit:.4f}")
print(f"Final error = {result.fun:.4f}")

# ---- Plot ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(t, C, s=15, color="C0", label="measured")
t_smooth = np.linspace(t.min(), t.max(), 200)
ax.plot(t_smooth, C0 * np.exp(-k_fit * t_smooth), "-", color="C1",
        label=f"fit: C0·exp(-{k_fit:.3f}·t)")
ax.set_xlabel("time t (s)")
ax.set_ylabel("concentration C")
ax.set_title("First-order reaction: measured vs fitted")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("kinetics.png", dpi=150)
print("Saved kinetics.png")