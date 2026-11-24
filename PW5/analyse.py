"""
analyse.py — выполняет шаги 3–8 (фильтр, огибающие, t_P, t_S, R, PGV, M)
для одного CSV-файла. Возвращает словарь со всеми величинами.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ---------- bandpass (Part 3) ----------
def bandpass(x, fs, a, b):
    n = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(n, 1.0 / fs)
    X[(np.abs(f) < a) | (np.abs(f) > b)] = 0
    return np.fft.ifft(X).real


# ---------- envelope (Part 4) ----------
def envelope(x, fs, win_s):
    w = max(1, int(round(win_s * fs)))
    return np.convolve(np.abs(x), np.ones(w) / w, mode="same")


# ---------- full pipeline ----------
def analyse(csv_file, make_figs=True):
    d = pd.read_csv(csv_file)
    t = d["time_s"].to_numpy()
    Z = d["Z"].to_numpy()
    N = d["N"].to_numpy()
    E = d["E"].to_numpy()
    fs = 1.0 / (t[1] - t[0])

    # --- Part 3: filter 0.5..10 Hz ---
    A_LOW, B_HIGH = 0.5, 10.0
    Zf = bandpass(Z, fs, A_LOW, B_HIGH)
    Nf = bandpass(N, fs, A_LOW, B_HIGH)
    Ef = bandpass(E, fs, A_LOW, B_HIGH)
    Hf = np.sqrt(Nf**2 + Ef**2)

    # --- Part 4: envelopes, 0.5 s window ---
    WIN = 0.5
    envZ = envelope(Zf, fs, WIN)
    envH = envelope(Hf, fs, WIN)

    # --- Part 5: arrivals ---
    levP = 0.05 * envZ.max()
    iP = int(np.argmax(envZ > levP))
    tP = float(t[iP])

    levS = 0.10 * envH.max()
    i_start = int(np.searchsorted(t, tP + 2.0))
    iS = i_start + int(np.argmax(envH[i_start:] > levS))
    tS = float(t[iS])

    peaks, _ = find_peaks(envH, height=0.5 * envH.max())
    i_peak = peaks[int(np.argmax(envH[peaks]))]
    t_peak = float(t[i_peak])

    # --- Part 6: distance and origin time ---
    vP, vS = 6.2, 3.6
    dt = tS - tP
    k = 1.0 / vS - 1.0 / vP
    R = dt / k
    t0 = tP - R / vP

    # --- Part 8: PGV and magnitude ---
    H = np.sqrt(N**2 + E**2)
    pgv_cms = float(H.max()) * 100.0
    M = (np.log10(pgv_cms) + np.log10(R) + 1.0) / 0.55

    out = dict(
        file=os.path.basename(csv_file),
        t_P=tP, t_S=tS, dt=dt, R_km=R, t0=t0,
        PGV_cms=pgv_cms, M=M,
    )

    # --- optional figure for the report ---
    if make_figs:
        fig_dir = "figures"
        os.makedirs(fig_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(csv_file))[0]
        mm = 1e3
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
        ax[0].plot(t, Zf * mm, lw=0.6, color="k", label="Z filt")
        ax[0].axvline(tP, color="green", ls="--", label=f"t_P = {tP:.1f}s")
        ax[0].axvline(tS, color="orange", ls="--", label=f"t_S = {tS:.1f}s")
        ax[0].set_ylabel("Z filt (mm/s)")
        ax[0].grid(alpha=0.3); ax[0].legend(loc="upper right")
        ax[1].plot(t, envZ * mm, lw=0.8, color="k",  label="env Z")
        ax[1].plot(t, envH * mm, lw=0.8, color="C3", label="env H")
        ax[1].axvline(tP, color="green", ls="--")
        ax[1].axvline(tS, color="orange", ls="--")
        ax[1].set_ylabel("envelopes (mm/s)")
        ax[1].set_xlabel("time (s)")
        ax[1].grid(alpha=0.3); ax[1].legend(loc="upper right")
        ax[0].set_title(f"{stem}: R={R:.1f} km, M={M:.2f}")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"analyse_{stem}.png"), dpi=130)
        plt.close(fig)

    return out


# ---------- run on both files, save one table ----------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    rows = []
    for f in ["data/seismogram_A.csv", "data/seismogram_B.csv"]:
        if not os.path.exists(f):
            print(f"[skip] {f} not found")
            continue
        r = analyse(f)
        print(f"[{r['file']}] t_P={r['t_P']:.2f}s  t_S={r['t_S']:.2f}s  "
              f"dt={r['dt']:.2f}s  R={r['R_km']:.1f}km  "
              f"t0={r['t0']:.2f}s  PGV={r['PGV_cms']:.4f}cm/s  M={r['M']:.2f}")
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv("results/events.csv", index=False)
    print("[ok] results/events.csv saved")
    print(df.to_string(index=False))