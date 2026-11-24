import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

z = np.load("results/envelopes_A.npz")
t = z["t"]; Zf = z["Zf"]; Nf = z["Nf"]; Ef = z["Ef"]
Hf = z["Hf"]; envZ = z["envZ"]; envH = z["envH"]
fs = float(z["fs"]); WIN = float(z["win"])
print(f"fs = {fs:.3f} Hz, N = {len(t)}, duration = {t[-1]:.2f} s")

# ---------- t_P: first time Z envelope > 5% of its max ----------
levP = 0.05 * envZ.max()
iP = np.argmax(envZ > levP)          # argmax on boolean = first True
tP = t[iP]
print(f"[P] level = {levP*1e3:.3f} mm/s, t_P = {tP:.2f} s")

# ---------- t_S: first time after t_P+2 s that horizontal envelope > 10% of its max ----------
levS = 0.10 * envH.max()
i_start = np.searchsorted(t, tP + 2.0)   # index corresponding to t_P + 2 s
iS_rel = np.argmax(envH[i_start:] > levS)
iS = i_start + iS_rel
tS = t[iS]
print(f"[S] level = {levS*1e3:.3f} mm/s, t_S = {tS:.2f} s")
print(f"[dt] t_S - t_P = {tS - tP:.2f} s")

# ---------- time of horizontal envelope maximum (via find_peaks) ----------
peaks, props = find_peaks(envH, height=0.5 * envH.max())
i_peak = peaks[np.argmax(envH[peaks])]
t_peak = t[i_peak]
print(f"[peak] horizontal envelope max at t = {t_peak:.2f} s  (t_S = {tS:.2f} s)")
print(f"[diff] t_peak - t_S = {t_peak - tS:.2f} s")

# ---------- plot: raw 3 channels + filtered envelopes + markers ----------
mm = 1e3
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(13, 11))

ax[0].plot(t, Zf * mm, lw=0.6, color="k");  ax[0].set_ylabel("Z filt (mm/s)")
ax[1].plot(t, Nf * mm, lw=0.6, color="C0"); ax[1].set_ylabel("N filt (mm/s)")
ax[2].plot(t, Ef * mm, lw=0.6, color="C3"); ax[2].set_ylabel("E filt (mm/s)")
ax[3].plot(t, envZ * mm, lw=0.8, color="k",  label="env Z")
ax[3].plot(t, envH * mm, lw=0.8, color="C3", label="env H")
ax[3].set_ylabel("envelopes (mm/s)"); ax[3].set_xlabel("time (s)")
ax[3].legend(loc="upper right")

for a in ax:
    a.axvline(tP, color="green", ls="--", lw=1.2)
    a.axvline(tS, color="orange", ls="--", lw=1.2)
    a.grid(alpha=0.3)
ax[0].set_title(f"t_P = {tP:.2f} s, t_S = {tS:.2f} s, dt = {tS-tP:.2f} s")
fig.tight_layout()
fig.savefig("figures/step5_arrivals.png", dpi=130)
plt.close(fig)

# zoom around first arrival
t_lo = max(0, tP - 20)
t_hi = tS + 40
m = (t >= t_lo) & (t <= t_hi)
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(13, 11))
ax[0].plot(t[m], Zf[m] * mm, lw=0.8, color="k");  ax[0].set_ylabel("Z filt (mm/s)")
ax[1].plot(t[m], Nf[m] * mm, lw=0.8, color="C0"); ax[1].set_ylabel("N filt (mm/s)")
ax[2].plot(t[m], Ef[m] * mm, lw=0.8, color="C3"); ax[2].set_ylabel("E filt (mm/s)")
ax[3].plot(t[m], envZ[m] * mm, lw=0.8, color="k",  label="env Z")
ax[3].plot(t[m], envH[m] * mm, lw=0.8, color="C3", label="env H")
ax[3].set_ylabel("envelopes (mm/s)"); ax[3].set_xlabel("time (s)")
ax[3].legend(loc="upper right")
for a in ax:
    a.axvline(tP, color="green", ls="--", lw=1.5)
    a.axvline(tS, color="orange", ls="--", lw=1.5)
    a.grid(alpha=0.3)
ax[0].set_title(f"Zoom on arrivals: t_P = {tP:.2f} s, t_S = {tS:.2f} s")
fig.tight_layout()
fig.savefig("figures/step5_arrivals_zoom.png", dpi=130)
plt.close(fig)

print("saved figures/step5_arrivals.png, step5_arrivals_zoom.png")

# save for Part 6
np.savez("results/arrivals_A.npz",
         tP=tP, tS=tS, t_peak=t_peak,
         envZ=envZ, envH=envH, t=t, fs=fs)
print("[ok] results/arrivals_A.npz saved")
print(f"[summary] t_P = {tP:.2f} s, t_S = {tS:.2f} s, dt = {tS-tP:.2f} s")