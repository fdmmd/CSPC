import numpy as np, pandas as pd
d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy(); Z = d["Z"].to_numpy()
fs = 1.0/(t[1]-t[0]); N = len(Z)

X  = np.fft.fft(Z);           f  = np.fft.fftfreq(N, 1/fs)
Zq = Z[t<=60]; Nq=len(Zq)
Xq = np.fft.fft(Zq);          fq = np.fft.fftfreq(Nq, 1/fs)

amp  = np.abs(X) /N *2
ampq = np.abs(Xq)/Nq*2
ampq_on_f = np.interp(f, fq, ampq)   # quiet spectrum on whole-record freqs

pos = f > 0
diff = amp - ampq_on_f               # absolute extra amplitude due to EQ
band = (f > 0.05) & (f < 20)

i_abs  = np.argmax((amp - ampq_on_f)[band])
i_full = np.argmax(amp[band])
print(f"[abs diff] max extra amplitude at f = {f[band][i_abs]:.3f} Hz")
print(f"[absolute] whole-record amplitude max at f = {f[band][i_full]:.3f} Hz")

# energy (P SD-ish) in decades
for lo, hi in [(0.05,0.5),(0.5,2),(2,5),(5,10),(10,20)]:
    m = (f>=lo)&(f<hi)
    E_eq  = np.sum(amp[m]**2)
    E_q   = np.sum(ampq_on_f[m]**2)
    print(f"  {lo:5.2f}-{hi:5.2f} Hz : EQ {E_eq:.2e}  quiet {E_q:.2e}  ratio {E_eq/E_q:6.2f}")
