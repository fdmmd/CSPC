"""Fetch UCI Concrete dataset and save as data/concrete.csv."""
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch
concrete = fetch_ucirepo(id=165)

print("=== Metadata ===")
print(concrete.metadata)

print("\n=== Variables ===")
print(concrete.variables)

X = concrete.data.features.copy()
y = concrete.data.targets.iloc[:, 0].rename("strength")

# Rename columns by position (from variables table)
X.columns = ["cement", "slag", "ash", "water", "superplastic",
             "coarse_agg", "fine_agg", "age"]

print("\n=== New names ===")
print(list(X.columns))

# Save
os.makedirs("data", exist_ok=True)
X.assign(strength=y).to_csv("data/concrete.csv", index=False)
print("\nSaved data/concrete.csv")