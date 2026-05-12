"""
Cancer Classification API — Hugging Face Spaces Backend
========================================================
Flask API that loads the trained SVM model, scaler, and feature
selector, and exposes a /predict endpoint for the Vercel frontend.
"""

import os, json, joblib, numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.datasets import load_breast_cancer

app = Flask(__name__)

# Allow the Vercel frontend (set via env var) + localhost for dev
FRONTEND_URL = os.environ.get("FRONTEND_URL", "*")
CORS(app, origins=["*"])  # HF Spaces needs broad CORS

# ── Load artefacts ────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
model    = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
selector = joblib.load(os.path.join(MODEL_DIR, "selector.pkl"))

# ── Feature metadata ─────────────────────────────────────────────
data = load_breast_cancer()
FEATURE_NAMES = list(data.feature_names)
X_all         = data.data

FEATURE_META = []
for i, name in enumerate(FEATURE_NAMES):
    col = X_all[:, i]
    FEATURE_META.append({
        "index": i,
        "name":  name,
        "min":   float(round(col.min(), 4)),
        "max":   float(round(col.max(), 4)),
        "mean":  float(round(col.mean(), 4)),
        "step":  float(round((col.max() - col.min()) / 200, 6)),
    })

SELECTED_MASK = selector.get_support().tolist()


@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Cancer Classification API is running"})


@app.route("/features", methods=["GET"])
def get_features():
    """Return feature metadata + selection mask for the frontend."""
    return jsonify({
        "features":      FEATURE_META,
        "selected_mask": SELECTED_MASK,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Receive 30 feature values, select → scale → predict."""
    payload = request.get_json(force=True)
    values  = payload.get("features", [])

    if len(values) != 30:
        return jsonify({"error": "Expected exactly 30 feature values"}), 400

    arr      = np.array(values, dtype=np.float64).reshape(1, -1)
    selected = selector.transform(arr)        # 30 → 20 features
    scaled   = scaler.transform(selected)     # scale the 20 selected

    pred     = int(model.predict(scaled)[0])
    label    = "Benign" if pred == 1 else "Malignant"

    try:
        proba = model.predict_proba(scaled)[0].tolist()
        confidence = float(max(proba)) * 100
    except AttributeError:
        dec = float(model.decision_function(scaled)[0])
        confidence = min(abs(dec) * 20, 100)
        proba = None

    return jsonify({
        "prediction": pred,
        "label":      label,
        "confidence": round(confidence, 2),
        "proba":      proba,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
