"""PW3 Session 1 -- load and characterise the heart dataset.

Parts:
  1. Load heart.csv with pandas
  2. Plot distributions of the four continuous variables
  3. Test normality (Shapiro-Wilk + Q-Q plots)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Part 1: Load the data
# ============================================================
df = pd.read_csv("heart.csv")

print(f"Shape: {df.shape}")
print(f"\nFirst rows:")
print(df.head())
print(f"\nColumns: {list(df.columns)}")
print(f"\nDescribe:")
print(df[["age", "chol", "trestbps", "thalach"]].describe())

# The four continuous columns
cont_cols = ["age", "chol", "trestbps", "thalach"]

# ============================================================
# Part 2: Distributions (one figure with 4 panels)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flat, cont_cols):
    ax.hist(df[col].dropna(), bins=20, color="C0", edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("count")
fig.suptitle("Distributions of the four continuous variables")
fig.tight_layout()
fig.savefig("distributions.png", dpi=150)
print("\nSaved distributions.png")

# ============================================================
# Part 3: Normality check (Shapiro-Wilk + Q-Q plots)
# ============================================================
print("\nShapiro-Wilk test (H0: data is normal):")
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes2.flat, cont_cols):
    data = df[col].dropna()
    stat, p = stats.shapiro(data)
    verdict = "normal" if p > 0.05 else "NOT normal"
    print(f"  {col:10s}  W = {stat:.4f}  p = {p:.4g}  -> {verdict}")
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f"{col}  (p = {p:.3g})")
fig2.suptitle("Q-Q plots vs normal distribution")
fig2.tight_layout()
fig2.savefig("qqplots.png", dpi=150)
print("Saved qqplots.png")