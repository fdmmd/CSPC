"""Convert processed.cleveland.data -> heart.csv."""
import pandas as pd

cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

df = pd.read_csv("processed.cleveland.data", names=cols, na_values="?")

# target: 1 if num > 0 (disease), 0 otherwise
df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])

df.to_csv("heart.csv", index=False)
print(f"Saved heart.csv: {df.shape}")
print(df[["age", "chol", "trestbps", "thalach", "target"]].head())