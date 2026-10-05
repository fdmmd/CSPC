"""Find chemical equilibrium extent x for H2 + I2 <=> 2 HI."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 54.0
print(f"K = {K}")

def k_imbalance(x):
    """Zero at equilibrium. Multiply through by (1-x)^2 to avoid singularity."""
    return (2*x)**2 - K * (1 - x)**2

# ---- Method 1: Newton ----
x_newton = newton(k_imbalance, x0=0.5)
print(f"Newton: x_eq = {x_newton:.6f}")

# ---- Method 2: SLSQP ----
def error(x):
    return k_imbalance(x)**2

result = minimize(error, x0=0.5, method="SLSQP", bounds=[(0, 1 - 1e-9)])
x_slsqp = result.x[0]
print(f"SLSQP:  x_eq = {x_slsqp:.6f}")
print(f"Both agree: {np.isclose(x_newton, x_slsqp)}")

# ---- Equilibrium amounts ----
x_eq = x_newton
print(f"\nAt equilibrium:")
print(f"  n_H2 = {1 - x_eq:.4f} mol")
print(f"  n_I2 = {1 - x_eq:.4f} mol")
print(f"  n_HI = {2 * x_eq:.4f} mol")

# ---- Plot ----
x = np.linspace(0, 0.999, 200)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, 1 - x, label="H2 = 1 - x")
ax.plot(x, 1 - x, "--", label="I2 = 1 - x")
ax.plot(x, 2 * x, label="HI = 2x")
ax.axvline(x_eq, color="red", linestyle=":", label=f"equilibrium x = {x_eq:.3f}")
ax.set_xlabel("extent of reaction x")
ax.set_ylabel("amount (mol)")
ax.set_title(f"H2 + I2 ⇌ 2 HI  (K = {K})")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("equilibrium.png", dpi=150)
print("Saved equilibrium.png")