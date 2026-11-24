import numpy as np
import pandas as pd

# ---------- load raw data (PGV from raw, not filtered!) ----------
d = pd.read_csv("data/seismogram_A.csv")
N = d["N"].to_numpy()
E = d["E"].to_numpy()

# PGV = max of sqrt(N^2+E^2) over whole record
H = np.sqrt(N**2 + E**2)
pgv_ms = H.max()              # m/s
pgv_cms = pgv_ms * 100.0      # cm/s
print(f"PGV = {pgv_ms:.6e} m/s = {pgv_cms:.4f} cm/s")

# ---------- distance from Part 6 ----------
dist = np.load("results/distance_A.npz")
R = float(dist["R"])
print(f"R = {R:.1f} km")

# ---------- magnitude formula (PDF) ----------
M = (np.log10(pgv_cms) + np.log10(R) + 1.0) / 0.55
print(f"M = {M:.2f}")

# ---------- what-if: same EQ at 2*R ----------
R2 = 2.0 * R
M_wrong_if_noR = (np.log10(pgv_cms) + 1.0) / 0.55
print(f"[what-if] ignoring log10(R) gives M = {M_wrong_if_noR:.2f}")
print(f"[what-if] at 2R with correct formula M(R') = {(np.log10(pgv_cms)+np.log10(R2)+1.0)/0.55:.2f}")

# PGV needed for M+1 at same R
pgv_Mplus1 = 10 ** (((M + 1) * 0.55) - np.log10(R) - 1.0)
print(f"[what-if] PGV for M+1 at same R = {pgv_Mplus1:.3f} cm/s  (x{pgv_Mplus1/pgv_cms:.2f})")

np.savez("results/magnitude_A.npz", pgv_cms=pgv_cms, R=R, M=M)
print("[ok] results/magnitude_A.npz saved")
print(f"[summary] R = {R:.1f} km, PGV = {pgv_cms:.4f} cm/s, M = {M:.2f}")