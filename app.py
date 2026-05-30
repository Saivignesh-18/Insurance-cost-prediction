"""
app.py
------
Flask REST API for Insurance Cost Prediction.

Endpoints:
  POST /predict  — returns predicted insurance cost
  GET  /health   — health check
  GET  /         — serves the frontend

Run:
    python app.py
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

# ── Load model artefacts ──────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

def load_pkl(name):
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found. Run 'python train_model.py' first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)

model    = load_pkl("model.pkl")
scaler   = load_pkl("scaler.pkl")
features = load_pkl("features.pkl")

print("Model loaded successfully.")

# ── Helper ────────────────────────────────────────────────────────────────────
def build_feature_vector(data: dict) -> np.ndarray:
    """Convert raw input dict to the model's feature vector."""
    age      = float(data["age"])
    bmi      = float(data["bmi"])
    children = float(data["children"])
    sex      = data["sex"].strip().lower()        # "male" | "female"
    smoker   = data["smoker"].strip().lower()     # "yes"  | "no"
    region   = data["region"].strip().lower()     # northeast/northwest/southeast/southwest

    sex_male         = 1 if sex == "male" else 0
    smoker_yes       = 1 if smoker == "yes" else 0
    region_northeast = 1 if region == "northeast" else 0
    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    bmi_obese        = 1 if bmi >= 30 else 0
    smoker_obese     = smoker_yes * bmi_obese

    return np.array([[
        age, bmi, children,
        sex_male, smoker_yes,
        region_northeast, region_northwest, region_southeast,
        bmi_obese, smoker_obese
    ]])

def get_risk_level(cost: float) -> str:
    if cost < 5000:   return "Low"
    if cost < 15000:  return "Medium"
    if cost < 30000:  return "High"
    return "Very High"

def get_bmi_category(bmi: float) -> str:
    if bmi < 18.5: return "Underweight"
    if bmi < 25:   return "Normal"
    if bmi < 30:   return "Overweight"
    return "Obese"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "LinearRegression"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate required fields
        required = ["age", "bmi", "children", "sex", "smoker", "region"]
        missing  = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Validate ranges
        age = float(data["age"])
        bmi = float(data["bmi"])
        children = int(data["children"])
        if not (18 <= age <= 100):
            return jsonify({"error": "Age must be between 18 and 100"}), 400
        if not (10 <= bmi <= 60):
            return jsonify({"error": "BMI must be between 10 and 60"}), 400
        if not (0 <= children <= 10):
            return jsonify({"error": "Children must be between 0 and 10"}), 400

        # Build vector, scale, predict
        X_raw    = build_feature_vector(data)
        X_scaled = scaler.transform(X_raw)
        cost     = float(model.predict(X_scaled)[0])
        cost     = max(1000, round(cost, 2))

        response = {
            "annual_cost":  cost,
            "monthly_cost": round(cost / 12, 2),
            "risk_level":   get_risk_level(cost),
            "bmi_category": get_bmi_category(bmi),
            "inputs": {
                "age": age, "bmi": bmi, "children": children,
                "sex": data["sex"], "smoker": data["smoker"],
                "region": data["region"]
            }
        }
        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n Insurance Cost Predictor API")
    print(" Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
