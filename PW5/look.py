import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
Z = d["Z"].to_numpy()
N = d["N"].to_numpy()
E = d["E"].to_numpy()

fs = 1.0 / (t[1] - t[0])
print(f"fs = {fs:.3f} Hz, N = {len(t)}, duration = {t[-1]:.2f} s")

mm = 1e3

# full record
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
ax[0].plot(t, Z * mm, lw=0.6, color="k");  ax[0].set_ylabel("Z (mm/s)")
ax[1].plot(t, N * mm, lw=0.6, color="C0"); ax[1].set_ylabel("N (mm/s)")
ax[2].plot(t, E * mm, lw=0.6, color="C3"); ax[2].set_ylabel("E (mm/s)")
ax[2].set_xlabel("time (s)")
for a in ax: a.grid(alpha=0.3)
fig.suptitle("Seismogram A — raw, all channels")
fig.tight_layout()
fig.savefig("figures/step1_full.png", dpi=130)
plt.close(fig)

# zoom 60 s
m = t <= 60
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
ax[0].plot(t[m], Z[m] * mm, lw=0.6, color="k");  ax[0].set_ylabel("Z (mm/s)")
ax[1].plot(t[m], N[m] * mm, lw=0.6, color="C0"); ax[1].set_ylabel("N (mm/s)")
ax[2].plot(t[m], E[m] * mm, lw=0.6, color="C3"); ax[2].set_ylabel("E (mm/s)")
ax[2].set_xlabel("time (s)")
for a in ax: a.grid(alpha=0.3)
fig.suptitle("Seismogram A — first 60 s (zoom)")
fig.tight_layout()
fig.savefig("figures/step1_zoom60.png", dpi=130)
plt.close(fig)

print("saved figures/step1_full.png, figures/step1_zoom60.png")