"""PW3 Session 2 -- draw conclusions from the heart data.

Parts:
  4. Compare thalach between disease and healthy groups (Mann-Whitney)
  5. Correlation between age and thalach (Spearman)
  6. Naive analysis of chemicals_cancer.csv
  7. Account for a confounder
  Bonus. Entropy of a categorical column
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("heart.csv")

# ============================================================
# Part 4: Do heart-disease patients reach a different max heart rate?
# ============================================================
print("=" * 60)
print("Part 4: thalach -- disease vs healthy")
print("=" * 60)

healthy = df[df["target"] == 0]["thalach"].dropna()
disease = df[df["target"] == 1]["thalach"].dropna()

print(f"Healthy: n = {len(healthy)}, mean = {healthy.mean():.2f}, "
      f"std = {healthy.std():.2f}, sem = {healthy.sem():.2f}")
print(f"Disease: n = {len(disease)}, mean = {disease.mean():.2f}, "
      f"std = {disease.std():.2f}, sem = {disease.sem():.2f}")

# Normality already checked in Session 1 -> all variables are NOT normal
# So use Mann-Whitney U (non-parametric), not t-test
u_stat, p_value = stats.mannwhitneyu(healthy, disease, alternative="two-sided")
print(f"\nMann-Whitney U = {u_stat:.1f}, p = {p_value:.4g}")
if p_value < 0.05:
    print("-> The two groups DIFFER significantly (p < 0.05)")
else:
    print("-> No significant difference (p >= 0.05)")

# Plot means with 95% CI error bars
fig, ax = plt.subplots(figsize=(6, 5))
groups = ["Healthy", "Disease"]
means = [healthy.mean(), disease.mean()]
errors = [1.96 * healthy.sem(), 1.96 * disease.sem()]  # 95% CI

ax.bar(groups, means, yerr=errors, capsize=10,
       color=["C0", "C1"], alpha=0.8)
ax.set_ylabel("mean thalach (bpm)")
ax.set_title(f"Max heart rate: healthy vs disease\n(Mann-Whitney p = {p_value:.3g})")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("thalach_comparison.png", dpi=150)
print("Saved thalach_comparison.png")

# ============================================================
# Part 5: Do age and thalach move together?
# ============================================================
print("\n" + "=" * 60)
print("Part 5: correlation age vs thalach")
print("=" * 60)

# Spearman (non-parametric) because data is not normal
rho, p_corr = stats.spearmanr(df["age"], df["thalach"])
print(f"Spearman rho = {rho:.4f}, p = {p_corr:.4g}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(df["age"], df["thalach"], s=10, alpha=0.5, color="C0")
ax.set_xlabel("age")
ax.set_ylabel("thalach")
ax.set_title(f"age vs thalach (Spearman rho = {rho:.3f}, p = {p_corr:.3g})")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("age_vs_thalach.png", dpi=150)
print("Saved age_vs_thalach.png")