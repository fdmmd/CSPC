"""
PW2 Lab A -- Motion from tracking data.

Read noisy free-fall position measurements, then:
  - differentiate once  -> velocity
  - differentiate twice -> acceleration (should be ~ constant -g, but noisy!)
  - integrate the acceleration back up -> recover velocity and position

Complete the TODOs. Run:  python analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# TODO 1: read freefall.csv into arrays t and y
t, y = np.loadtxt("freefall.csv", delimiter=",", skiprows=1, unpack=True)

# TODO 2: compute velocity v = derivative of y w.r.t. t   (np.gradient)
#         and acceleration a = derivative of v w.r.t. t    (np.gradient again)
#         Print the mean acceleration. Is it close to -9.81? Is it noisy?
v = np.gradient(y, t)
a = np.gradient(v, t)

print(f"Mean acceleration: {a.mean():.3f} m/s²")

# TODO 3: integrate a back up to recover velocity and position
#         (hint: cumulative_trapezoid(a, t, initial=0) + v[0], then again)
v_rec = cumulative_trapezoid(a, t, initial=0) + v[0]
y_rec = cumulative_trapezoid(v_rec, t, initial=0) + y[0]

# TODO 4: make a figure with 3 stacked panels: position, velocity, acceleration
#         vs time. Mark the true -9.81 line on the acceleration panel.
#         Save it as motion.png
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

axes[0].plot(t, y, label="measured")
axes[0].plot(t, y_rec, "--", label="recovered")
axes[0].set_ylabel("position y (m)")
axes[0].legend()

axes[1].plot(t, v, label="measured")
axes[1].plot(t, v_rec, "--", label="recovered")
axes[1].set_ylabel("velocity v (m/s)")
axes[1].legend()

axes[2].plot(t, a, label="acceleration")
axes[2].axhline(-9.81, color="red", linestyle=":", label="−g = −9.81")
axes[2].set_ylabel("acceleration a (m/s²)")
axes[2].set_xlabel("time t (s)")
axes[2].legend()

fig.suptitle("Free fall: position, velocity, acceleration")
fig.tight_layout()
fig.savefig("motion.png", dpi=150)
print("Saved motion.png")
print(f"Mean acceleration: {a.mean():.3f} m/s²")
print("  (noisy — numerical differentiation amplifies measurement noise)")