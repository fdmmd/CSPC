"""
report.py — читает results/events.csv и пишет results/report.txt
с коротким summary по каждому землетрясению.
"""
import os
import pandas as pd

CSV = "results/events.csv"
OUT = "results/report.txt"

df = pd.read_csv(CSV)
os.makedirs("results", exist_ok=True)

lines = []
lines.append("PW5 — Earthquake summary")
lines.append("=" * 60)
lines.append(f"records analysed: {len(df)}")
lines.append("")

# per-event blocks
for _, r in df.iterrows():
    lines.append(f"Event {r['file']}")
    lines.append("-" * 60)
    lines.append(f"  P arrival      : {r['t_P']:7.2f} s")
    lines.append(f"  S arrival      : {r['t_S']:7.2f} s")
    lines.append(f"  S-P time       : {r['dt']:7.2f} s")
    lines.append(f"  distance       : {r['R_km']:7.1f} km")
    lines.append(f"  origin time    : {r['t0']:7.2f} s")
    lines.append(f"  PGV            : {r['PGV_cms']:7.4f} cm/s")
    lines.append(f"  magnitude M    : {r['M']:7.2f}")
    lines.append("")

# comparison
if len(df) >= 2:
    i_close = df["R_km"].idxmin()
    i_strong = df["M"].idxmax()
    close = df.loc[i_close, "file"]
    strong = df.loc[i_strong, "file"]

    lines.append("Comparison")
    lines.append("-" * 60)
    lines.append(f"  closer  earthquake : {close} (R = {df.loc[i_close,'R_km']:.1f} km)")
    lines.append(f"  stronger earthquake: {strong} (M = {df.loc[i_strong,'M']:.2f})")
    lines.append("")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"[ok] {OUT} written")
print("\n".join(lines))