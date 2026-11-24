import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# use RAW Z (not filtered) — spectrogram shows everything
d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
Z = d["Z"].to_numpy()
fs = 1.0 / (t[1] - t[0])

f, ts, S = spectrogram(Z, fs=fs, nperseg=256)
print(f"spectrogram: f up to {f[-1]:.1f} Hz, {len(ts)} time bins")

# limit to 10 Hz
m = f <= 10
fig, ax = plt.subplots(figsize=(12, 5))
pc = ax.pcolormesh(ts, f[m], 10*np.log10(S[m] + 1e-30),
                   shading="auto", cmap="magma")
ax.set_xlabel("time (s)")
ax.set_ylabel("frequency (Hz)")
ax.set_title("Spectrogram of raw Z (0–10 Hz)")
fig.colorbar(pc, ax=ax, label="10*log10(S)")
fig.tight_layout()
fig.savefig("figures/step7_spectrogram.png", dpi=130)
plt.close(fig)

# annotations of phases from Part 5 / 6
arr = np.load("results/arrivals_A.npz")
tP = float(arr["tP"]); tS = float(arr["tS"])
fig, ax = plt.subplots(figsize=(12, 5))
pc = ax.pcolormesh(ts, f[m], 10*np.log10(S[m] + 1e-30),
                   shading="auto", cmap="magma")
ax.axvline(tP, color="cyan",   ls="--", lw=1.5, label=f"t_P = {tP:.1f} s")
ax.axvline(tS, color="yellow", ls="--", lw=1.5, label=f"t_S = {tS:.1f} s")
ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
ax.set_title("Spectrogram of raw Z with P and S")
ax.legend(loc="upper right")
fig.colorbar(pc, ax=ax, label="10*log10(S)")
fig.tight_layout()
fig.savefig("figures/step7_spectrogram_annot.png", dpi=130)
plt.close(fig)

print("saved figures/step7_spectrogram.png, step7_spectrogram_annot.png")