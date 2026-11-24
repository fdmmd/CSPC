import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.load("results/filtered_A.npz")
t = z["t"]; Zf = z["Zf"]; Nf = z["Nf"]; Ef = z["Ef"]
fs = float(z["fs"])
print(f"fs = {fs:.3f} Hz, N = {len(t)}, duration = {t[-1]:.2f} s")

def envelope(x, fs, win_s):
    w = max(1, int(round(win_s * fs)))
    return np.convolve(np.abs(x), np.ones(w) / w, mode="same")

Hf = np.sqrt(Nf**2 + Ef**2)

wins = [0.05, 0.5, 5.0]
fig, ax = plt.subplots(figsize=(12, 5))
for win, c in zip(wins, ["C0", "C1", "C2"]):
    ax.plot(t, envelope(Zf, fs, win) * 1e3, lw=0.8, color=c, label=f"window = {win} s")
ax.set_xlabel("time (s)"); ax.set_ylabel("Z envelope (mm/s)")
ax.set_title("(a) Envelope of filtered Z — effect of window length")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig("figures/step4a_envelope_windows.png", dpi=130); plt.close(fig)

WIN = 0.5
envZ = envelope(Zf, fs, WIN)
envH = envelope(Hf, fs, WIN)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
ax[0].plot(t, envZ * 1e3, color="k", lw=0.8); ax[0].set_ylabel("Z envelope (mm/s)")
ax[0].set_title(f"Envelopes, window = {WIN} s"); ax[0].grid(alpha=0.3)
ax[1].plot(t, envH * 1e3, color="C3", lw=0.8); ax[1].set_ylabel("horizontal envelope (mm/s)")
ax[1].set_xlabel("time (s)"); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig("figures/step4b_envelopes_Z_H.png", dpi=130); plt.close(fig)

print("saved figures/step4a_envelope_windows.png, step4b_envelopes_Z_H.png")

np.savez("results/envelopes_A.npz",
         t=t, Zf=Zf, Nf=Nf, Ef=Ef, Hf=Hf,
         envZ=envZ, envH=envH, fs=fs, win=WIN)
print("[ok] results/envelopes_A.npz saved")

iZ = np.argmax(envZ); iH = np.argmax(envH)
print(f"[hint] max Z envelope at t = {t[iZ]:.2f} s")
print(f"[hint] max horizontal envelope at t = {t[iH]:.2f} s")