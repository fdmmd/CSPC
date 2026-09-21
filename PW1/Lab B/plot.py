"""Plot observed radioactive decay vs the analytical law."""
import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 0.3

# TODO 1: read decay_observed.csv into arrays t and observed
t, observed = np.loadtxt("decay_observed.csv", delimiter=",", skiprows=1, unpack=True)

# TODO 2: set N0 to the first observed value and build analytical
N0 = observed[0]
analytical = N0 * np.exp(-LAMBDA * t)

# TODO 3: make the 1x2 subplot with shared axes
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

ax_left.scatter(t, observed, s=15, color="C0", label="observed")
ax_left.set_title("Observed data")
ax_left.set_xlabel("time")
ax_left.set_ylabel("count")
ax_left.legend()

ax_right.plot(t, analytical, color="C1", label="analytical")
ax_right.set_title(f"Analytical: $N_0 e^{{-\\lambda t}}$, λ={LAMBDA}")
ax_right.set_xlabel("time")
ax_right.legend()

fig.suptitle("Radioactive decay: observed vs analytical")
fig.tight_layout()

# TODO 4: save the figure as figure.png
fig.savefig("figure.png", dpi=200)
print("Saved figure.png")