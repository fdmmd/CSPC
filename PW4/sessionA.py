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
# ============================================================
# A5: Polynomial Regression
# ============================================================
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print("\n" + "=" * 60)
print("A5: Polynomial Regression")
print("=" * 60)

# Table for results
results = []

for degree in [1, 2, 3, 4]:
    # Create pipeline
    poly = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    
    # Fit
    poly.fit(X_train, y_train)
    
    # Number of features after expansion
    p = poly[0].n_output_features_
    
    # Predictions
    y_train_pred = poly.predict(X_train)
    y_test_pred = poly.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Adjusted R2 (on train)
    n = X_train.shape[0]
    train_r2_adj = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)
    
    results.append({
        "degree": degree,
        "p": p,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "train_r2_adj": train_r2_adj,
        "test_r2": test_r2,
    })

# Print table
print(f"\n{'degree':<8} {'p':>6} {'train_rmse':>12} {'test_rmse':>10} {'train_r2':>10} {'train_r2_adj':>14} {'test_r2':>10}")
for r in results:
    print(f"{r['degree']:<8} {r['p']:>6} {r['train_rmse']:>12.4f} {r['test_rmse']:>10.4f} {r['train_r2']:>10.4f} {r['train_r2_adj']:>14.4f} {r['test_r2']:>10.4f}")
# ============================================================
# A6: Classification (KNN and Decision Tree)
# ============================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

print("\n" + "=" * 60)
print("A6: Classification (KNN and Decision Tree)")
print("=" * 60)

# 1. Create classification target
y_class = (df["strength"] >= 40).astype(int)
print(f"Class proportions:\n{y_class.value_counts(normalize=True)}")

# 2. Split with stratify
Xc_train, Xc_test, c_train, c_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)
print(f"\nClassification split: Train {Xc_train.shape[0]}, Test {Xc_test.shape[0]}")

# --- KNN Sweep ---
print("\n--- KNN Sweep ---")
k_values = [1, 3, 5, 7, 15, 25, 51, 101]
knn_train_accs = []
knn_test_accs = []

print(f"{'K':>4} {'Train Acc':>10} {'Test Acc':>10}")
for k in k_values:
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    knn.fit(Xc_train, c_train)
    
    train_acc = knn.score(Xc_train, c_train)
    test_acc = knn.score(Xc_test, c_test)
    
    knn_train_accs.append(train_acc)
    knn_test_accs.append(test_acc)
    print(f"{k:>4} {train_acc:>10.4f} {test_acc:>10.4f}")

# Plot KNN sweep
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_values, knn_train_accs, marker='o', label='Train')
ax.plot(k_values, knn_test_accs, marker='s', label='Test')
ax.set_xscale("log")
ax.set_xlabel("K (log scale)")
ax.set_ylabel("Accuracy")
ax.set_title("KNN Accuracy vs K")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("knn_sweep.png", dpi=150)

# --- Decision Tree Sweep ---
print("\n--- Decision Tree Sweep ---")
depths = [1, 2, 3, 4, 5, 6, 8, None]
tree_train_accs = []
tree_test_accs = []
tree_leaves = []

print(f"{'Depth':>6} {'Leaves':>8} {'Train Acc':>10} {'Test Acc':>10}")
for depth in depths:
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=0)
    tree.fit(Xc_train, c_train)
    
    train_acc = tree.score(Xc_train, c_train)
    test_acc = tree.score(Xc_test, c_test)
    leaves = tree.get_n_leaves()
    
    tree_train_accs.append(train_acc)
    tree_test_accs.append(test_acc)
    tree_leaves.append(leaves)
    
    depth_str = str(depth) if depth is not None else "None"
    print(f"{depth_str:>6} {leaves:>8} {train_acc:>10.4f} {test_acc:>10.4f}")

# Plot Tree sweep (convert None to max depth + 1 for x-axis)
plot_depths = [d if d is not None else 10 for d in depths]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(plot_depths, tree_train_accs, marker='o', label='Train')
ax.plot(plot_depths, tree_test_accs, marker='s', label='Test')
ax.set_xlabel("Max Depth (10 = unlimited)")
ax.set_ylabel("Accuracy")
ax.set_title("Decision Tree Accuracy vs Depth")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("tree_sweep.png", dpi=150)

# --- Print top 2 levels of a chosen tree (e.g., depth=3) ---
print("\n--- Decision Tree (max_depth=3) top levels ---")
best_tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
best_tree.fit(Xc_train, c_train)
print(export_text(best_tree, feature_names=list(X.columns), max_depth=2))
# ============================================================
# A7: The metrics that matter
# ============================================================
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print("\n" + "=" * 60)
print("A7: Metrics and thresholds")
print("=" * 60)

# Use KNN with K=7
knn_best = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
knn_best.fit(Xc_train, c_train)
c_pred = knn_best.predict(Xc_test)
proba = knn_best.predict_proba(Xc_test)[:, 1]

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(c_test, c_pred).ravel()
print(f"KNN (K=5):")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"  Accuracy:  {accuracy_score(c_test, c_pred):.4f}")
print(f"  Precision: {precision_score(c_test, c_pred):.4f}")
print(f"  Recall:    {recall_score(c_test, c_pred):.4f}")
print(f"  F1:        {f1_score(c_test, c_pred):.4f}")

# Thresholds
print("\nThresholds for KNN (K=7):")
print(f"{'Threshold':>9} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} {'Precision':>10} {'Recall':>8} {'F1':>6}")
results_metrics = []
for t in [0.3, 0.5, 0.7, 0.9]:
    t_pred = (proba >= t).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(c_test, t_pred).ravel()
    p = precision_score(c_test, t_pred, zero_division=0)
    r = recall_score(c_test, t_pred, zero_division=0)
    f = f1_score(c_test, t_pred, zero_division=0)
    print(f"{t:>9.1f} {tp_t:>3} {fp_t:>3} {fn_t:>3} {tn_t:>3} {p:>10.4f} {r:>8.4f} {f:>6.4f}")
    results_metrics.append({"model": "KNN_K7", "threshold": t, "TP": tp_t, "FP": fp_t, "FN": fn_t, "TN": tn_t, "precision": p, "recall": r, "f1": f})

# Save metrics for B7
os.makedirs("results", exist_ok=True)
metrics_df = pd.DataFrame(results_metrics)
metrics_df.to_csv("results/A_metrics.csv", index=False)
print("\nSaved results/A_metrics.csv")
# ============================================================
# A8: Which ingredients matter? Feature importance
# ============================================================
from sklearn.inspection import permutation_importance

print("\n" + "=" * 60)
print("A8: Feature Importance")
print("=" * 60)

# 1. Raw coefficients (from linear regression A4)
raw_coefs = lr.coef_

# 2. Standardised coefficients
scaler_lr = make_pipeline(StandardScaler(), LinearRegression())
scaler_lr.fit(X_train, y_train)
std_coefs = scaler_lr[-1].coef_

# 3. Tree importances (from depth=3 or None)
best_tree_imp = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
best_tree_imp.fit(Xc_train, c_train)
tree_imps = best_tree_imp.feature_importances_

# 4. Permutation importance (KNN K=7)
knn_best = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
knn_best.fit(Xc_train, c_train)
perm_imp = permutation_importance(knn_best, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)

# Compile table
imp_df = pd.DataFrame({
    "feature": X.columns,
    "raw_coef": raw_coefs,
    "std_coef": std_coefs,
    "tree_importance": tree_imps,
    "knn_perm_importance": perm_imp.importances_mean,
})

print(imp_df.to_string(index=False))

# Plot (3 bar charts side by side)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].barh(imp_df["feature"], imp_df["std_coef"], color="C0")
axes[0].set_title("Standardized Coefficients")
axes[0].axvline(0, color="black", lw=0.8)

axes[1].barh(imp_df["feature"], imp_df["tree_importance"], color="C1")
axes[1].set_title("Tree Feature Importance")

axes[2].barh(imp_df["feature"], imp_df["knn_perm_importance"], color="C2")
axes[2].set_title("KNN Permutation Importance")

for ax in axes:
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("feature_importances.png", dpi=150)
print("\nSaved feature_importances.png")
# ============================================================
# Bonus: Let physics choose the features
# ============================================================
print("\n" + "=" * 60)
print("Bonus: Physics-based features (log(age) + w/b ratio)")
print("=" * 60)

# Create a copy of the data
X_phys = X.copy()
X_phys["log_age"] = np.log(X_phys["age"])
X_phys["w_b_ratio"] = X_phys["water"] / (X_phys["cement"] + X_phys["slag"] + X_phys["ash"] + 1e-9)

# Drop original age and water
X_phys = X_phys.drop(columns=["age", "water"])

print(f"Number of features: {X_phys.shape[1]} (was 8)")
print(f"New features: {list(X_phys.columns)}")

# Split
Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X_phys, y, test_size=0.2, random_state=42
)

# Fit linear regression
lr_phys = LinearRegression()
lr_phys.fit(Xp_train, yp_train)

# Evaluate
yp_pred = lr_phys.predict(Xp_test)
test_r2_phys = r2_score(yp_test, yp_pred)
test_rmse_phys = np.sqrt(mean_squared_error(yp_test, yp_pred))

print(f"\nLinear Regression with physics features:")
print(f"  Test RMSE: {test_rmse_phys:.4f} MPa")
print(f"  Test R²:   {test_r2_phys:.4f}")

print("\nComparison with polynomial table:")
print(f"  Linear (8 features):       Test R² = 0.5802")
print(f"  Physics (8 features):      Test R² = {test_r2_phys:.4f}")
print(f"  Polynomial degree=3 (164): Test R² = 0.8813")