"""PW4 Session A -- Concrete strength: supervised learning."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# A2: Clean and look
# ============================================================
print("=" * 60)
print("A2: Clean and look")
print("=" * 60)

df = pd.read_csv("data/concrete.csv")
print(f"Shape: {df.shape}")

# Duplicates
n_dups = df.duplicated().sum()
print(f"Duplicate rows: {n_dups}")

# Missing values
print(f"Missing values per column:\n{df.isna().sum()}")

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape after dedup: {df.shape}")

# Rebuild X and y
X = df.drop(columns=["strength"])
y = df["strength"]

print("\ndf.describe():")
print(df.describe())

# Plots: strength vs cement, strength vs age
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(df["cement"], df["strength"], s=5, alpha=0.4)
axes[0].set_xlabel("cement (kg/m³)")
axes[0].set_ylabel("strength (MPa)")
axes[0].set_title("Strength vs cement")

axes[1].scatter(df["age"], df["strength"], s=5, alpha=0.4)
axes[1].set_xlabel("age (days)")
axes[1].set_ylabel("strength (MPa)")
axes[1].set_title("Strength vs age")
fig.tight_layout()
fig.savefig("strength_vs_cement_age.png", dpi=150)
print("\nSaved strength_vs_cement_age.png")

# ============================================================
# A3: Split
# ============================================================
print("\n" + "=" * 60)
print("A3: Train/test split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
# ============================================================
# A4: Linear Regression
# ============================================================
print("\n" + "=" * 60)
print("A4: Linear Regression")
print("=" * 60)

# Fit
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# Coefficients (equation)
print("\nFitted equation:")
terms = [f"{coef:+.4f}*{name}" for coef, name in zip(lr.coef_, X.columns)]
equation = f"strength = {lr.intercept_:.4f} " + " ".join(terms)
print(equation)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Adjusted R2
n = X_train.shape[0]
p = X_train.shape[1]
train_r2_adj = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)

print(f"\n{'Metric':<20} {'Value':>10}")
print(f"{'Train RMSE':<20} {train_rmse:>10.4f}")
print(f"{'Test RMSE':<20} {test_rmse:>10.4f}")
print(f"{'Test MAE':<20} {test_mae:>10.4f}")
print(f"{'Train R2':<20} {train_r2:>10.4f}")
print(f"{'Train R2 (adj)':<20} {train_r2_adj:>10.4f}")
print(f"{'Test R2':<20} {test_r2:>10.4f}")

# Residuals vs age
residuals = y_test - y_test_pred
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_test["age"], residuals, s=15, alpha=0.6)
ax.axhline(0, color="red", linestyle="--")
ax.set_xscale("log")
ax.set_xlabel("age (days, log scale)")
ax.set_ylabel("residual (true - predicted) MPa")
ax.set_title("Residuals vs age (test set)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("residuals_vs_age.png", dpi=150)
print("\nSaved residuals_vs_age.png")