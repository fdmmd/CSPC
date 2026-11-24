import numpy as np

arr = np.load("results/arrivals_A.npz")
tP = float(arr["tP"]); tS = float(arr["tS"])
print(f"t_P = {tP:.2f} s, t_S = {tS:.2f} s")

# wave speeds (PDF)
vP = 6.2   # km/s
vS = 3.6   # km/s

dt = tS - tP
k  = 1.0/vS - 1.0/vP       # s/km
R  = dt / k                # km

print(f"dt = {dt:.2f} s")
print(f"1/vS - 1/vP = {k:.6f} s/km")
print(f"R = {R:.1f} km")

# uncertainty: t_S ±1 s
dt_lo = (tS - 1.0) - tP
dt_hi = (tS + 1.0) - tP
R_lo  = dt_lo / k
R_hi  = dt_hi / k
print(f"R with t_S-1 s = {R_lo:.1f} km")
print(f"R with t_S+1 s = {R_hi:.1f} km")
print(f"dR per 1 s of t_S = {(R_hi-R_lo)/2:.1f} km")

# origin time
t0 = tP - R / vP
print(f"origin time t0 = t_P - R/v_P = {t0:.2f} s")

# save
np.savez("results/distance_A.npz",
         tP=tP, tS=tS, dt=dt, R=R, R_lo=R_lo, R_hi=R_hi, t0=t0,
         vP=vP, vS=vS)
print("[ok] results/distance_A.npz saved")