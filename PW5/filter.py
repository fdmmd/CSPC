import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- load ----------
d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
Z = d["Z"].to_numpy()
N = d["N"].to_numpy()
E = d["E"].to_numpy()

fs = 1.0 / (t[1] - t[0])
Npts = len(Z)
print(f"fs = {fs:.3f} Hz, N = {Npts}")

# ---------- 3(a) ifft(fft(x)) == x ----------
X = np.fft.fft(Z)
back = np.fft.ifft(X).real
print("[3a] np.allclose(ifft(fft(Z)).real, Z) =", np.allclose(back, Z))

# ---------- 3(b) bandpass function ----------
def bandpass(x, fs, a, b):
    """Keep only frequencies a..b Hz (both sides of 0), return real signal."""
    n = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(n, 1.0/fs)
    X[(np.abs(f) < a) | (np.abs(f) > b)] = 0
    return np.fft.ifft(X).real

A_LOW, B_HIGH = 0.5, 10.0

Zf = bandpass(Z, fs, A_LOW, B_HIGH)
Nf = bandpass(N, fs, A_LOW, B_HIGH)
Ef = bandpass(E, fs, A_LOW, B_HIGH)
print(f"[3b] bandpass {A_LOW}..{B_HIGH} Hz applied to Z, N, E")

# plot filtered 3 channels
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
ax[0].plot(t, Zf * 1e3, lw=0.6, color="k");  ax[0].set_ylabel("Z filt (mm/s)")
ax[1].plot(t, Nf * 1e3, lw=0.6, color="C0"); ax[1].set_ylabel("N filt (mm/s)")
ax[2].plot(t, Ef * 1e3, lw=0.6, color="C3"); ax[2].set_ylabel("E filt (mm/s)")
ax[2].set_xlabel("time (s)")
for a in ax: a.grid(alpha=0.3)
fig.suptitle(f"Filtered {A_LOW}–{B_HIGH} Hz")
fig.tight_layout()
fig.savefig("figures/step3_filtered.png", dpi=130)
plt.close(fig)

# ---------- 3(c) what was removed: Z - Zf ----------
removed = Z - Zf
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t, removed * 1e3, lw=0.6, color="purple")
ax.set_xlabel("time (s)")
ax.set_ylabel("Z - Zfilt (mm/s)")
ax.set_title("What the filter removed from Z")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/step3_removed.png", dpi=130)
plt.close(fig)

# zoom on removed, first 60 s, to see microseism
m = t <= 60
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t[m], removed[m] * 1e3, lw=0.6, color="purple")
ax.set_xlabel("time (s)")
ax.set_ylabel("Z - Zfilt (mm/s)")
ax.set_title("Removed from Z — zoom first 60 s (should be smooth, low-freq noise)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/step3_removed_zoom.png", dpi=130)
plt.close(fig)

print("saved figures/step3_filtered.png, step3_removed.png, step3_removed_zoom.png")

# save filtered arrays for later steps
np.savez("results/filtered_A.npz",
         t=t, Zf=Zf, Nf=Nf, Ef=Ef, fs=fs)
print("[ok] results/filtered_A.npz saved (for next steps)")