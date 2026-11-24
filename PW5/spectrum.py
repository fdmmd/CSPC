import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
Z = d["Z"].to_numpy()

N = len(Z)
fs = 1.0 / (t[1] - t[0])
print(f"fs = {fs:.3f} Hz, N = {N}")

# ---------- FFT ----------
X = np.fft.fft(Z)
f = np.fft.fftfreq(N, d=1/fs)
amp = np.abs(X) / N * 2  # one-sided amplitude (for positive freqs)

# ---------- (a) amplitude vs index ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(np.arange(N), np.abs(X))
ax[0].set_xlabel("array index")
ax[0].set_ylabel("|X|")
ax[0].set_title("(a) amplitude vs index")
ax[0].grid(alpha=0.3)

ax[1].plot(f, np.abs(X))
ax[1].set_xlabel("frequency (Hz)")
ax[1].set_ylabel("|X|")
ax[1].set_title("(a) amplitude vs fftfreq")
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/step2a_index_vs_freq.png", dpi=130)
plt.close(fig)

# ---------- (b) log-log spectrum: quiet vs whole ----------
# quiet part = first 60 s
mask_q = t <= 60
Zq = Z[mask_q]
Nq = len(Zq)
Xq = np.fft.fft(Zq)
fq = np.fft.fftfreq(Nq, d=1/fs)
ampq = np.abs(Xq) / Nq * 2

# keep only positive frequencies
pos = f > 0
posq = fq > 0

fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(f[pos], amp[pos], label="whole record", lw=0.7)
ax.loglog(fq[posq], ampq[posq], label="first 60 s (quiet)", lw=0.7)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("amplitude (m/s)")
ax.set_title("(b) spectrum of Z — whole vs quiet")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("figures/step2b_loglog.png", dpi=130)
plt.close(fig)

print("saved figures/step2a_index_vs_freq.png, figures/step2b_loglog.png")

# ---------- печать подсказок для README ----------
# где шум сильнее всего (в тихой части)
q_band = (fq > 0.05) & (fq < 2.0)
i_noise = np.argmax(ampq[q_band])
print(f"[hint] quiet-part peak in 0.05..2 Hz at f = {fq[q_band][i_noise]:.3f} Hz")

# где землетрясение добавляет больше всего энергии
ratio = amp / np.maximum(ampq_interp:=np.interp(f, fq, ampq), 1e-30)
earth_band = (f > 0.1) & (f < 20)
i_eq = np.argmax(ratio[earth_band])
print(f"[hint] max earthquake/quiet ratio at f = {f[earth_band][i_eq]:.3f} Hz")