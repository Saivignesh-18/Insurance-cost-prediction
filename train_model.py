"""
train_model.py
--------------
Trains a Linear Regression model on the insurance dataset,
evaluates it, and saves the model + scaler to disk.

Run once before starting the Flask app:
    python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import os

# ── 1. Load or generate dataset ──────────────────────────────────────────────
DATASET_PATH = "insurance.csv"

if os.path.exists(DATASET_PATH):
    print(f"Loading dataset from {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)
else:
    print("insurance.csv not found — generating synthetic dataset ...")
    np.random.seed(42)
    n = 1338

    age        = np.random.randint(18, 65, n)
    sex        = np.random.choice(["male", "female"], n)
    bmi        = np.round(np.random.normal(30.7, 6.1, n).clip(15, 53.1), 1)
    children   = np.random.choice([0,1,2,3,4,5], n, p=[0.43,0.24,0.18,0.10,0.03,0.02])
    smoker     = np.random.choice(["yes","no"], n, p=[0.205, 0.795])
    region     = np.random.choice(["southwest","southeast","northwest","northeast"], n)

    charges = (
        250 * age
        + 330 * bmi
        + 480 * children
        + np.where(smoker=="yes", 23848 + np.where(bmi>=30, bmi*13.4, 0), 0)
        + np.where(sex=="female", -131, 0)
        + np.where(region=="southeast", -953, 0)
        + np.where(region=="southwest", -960, 0)
        + np.where(region=="northwest", -353, 0)
        - 11938
        + np.random.normal(0, 2000, n)
    ).clip(1122, 63770)

    df = pd.DataFrame({
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker,
        "region": region, "charges": np.round(charges, 2)
    })
    df.to_csv(DATASET_PATH, index=False)
    print(f"Saved {len(df)} rows to {DATASET_PATH}")

print(f"\nDataset shape : {df.shape}")
print(df.head())

# ── 2. Feature Engineering ────────────────────────────────────────────────────
df["sex_male"]         = (df["sex"] == "male").astype(int)
df["smoker_yes"]       = (df["smoker"] == "yes").astype(int)
df["region_northeast"] = (df["region"] == "northeast").astype(int)
df["region_northwest"] = (df["region"] == "northwest").astype(int)
df["region_southeast"] = (df["region"] == "southeast").astype(int)
df["bmi_obese"]        = (df["bmi"] >= 30).astype(int)
df["smoker_obese"]     = df["smoker_yes"] * df["bmi_obese"]

FEATURES = [
    "age", "bmi", "children",
    "sex_male", "smoker_yes",
    "region_northeast", "region_northwest", "region_southeast",
    "bmi_obese", "smoker_obese"
]

X = df[FEATURES]
y = df["charges"]

# ── 3. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Scale features ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. Train Model ────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_sc, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n── Model Performance ──────────────────────────────")
print(f"  R² Score : {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"  MAE      : ${mae:,.2f}")
print(f"  RMSE     : ${rmse:,.2f}")
print("───────────────────────────────────────────────────")

print("\nFeature Coefficients:")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"  {feat:<25} {coef:+.2f}")
print(f"  {'intercept':<25} {model.intercept_:+.2f}")

# ── 7. Save Model & Scaler ────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("features.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

print("\nSaved: model.pkl, scaler.pkl, features.pkl")
print("Training complete!")
