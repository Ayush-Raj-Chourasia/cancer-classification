"""
Cancer Classification – Interactive Testing UI
================================================
A Flask web app that loads the trained SVM model, scaler, and feature
selector, then lets users drag sliders for the 30 breast‑cancer features
and get a Benign / Malignant prediction in real time.
"""

import os, json, joblib, numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_breast_cancer

app = Flask(__name__)

# ── Load artefacts ────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")
model    = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
selector = joblib.load(os.path.join(MODEL_DIR, "selector.pkl"))

# ── Feature metadata from the dataset ─────────────────────────────
data = load_breast_cancer()
FEATURE_NAMES = list(data.feature_names)          # 30 names
X_all         = data.data                          # (569, 30)

# Pre‑compute per‑feature min / max / mean for the sliders
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

# Which features did SelectKBest keep?
SELECTED_MASK = selector.get_support().tolist()    # list of bool


@app.route("/")
def index():
    return render_template(
        "index.html",
        features=FEATURE_META,
        selected_mask=json.dumps(SELECTED_MASK),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Receive 30 feature values, scale → select → predict."""
    payload = request.get_json(force=True)
    values  = payload.get("features", [])

    if len(values) != 30:
        return jsonify({"error": "Expected exactly 30 feature values"}), 400

    arr      = np.array(values, dtype=np.float64).reshape(1, -1)
    selected = selector.transform(arr)        # 30 → 20 features
    scaled   = scaler.transform(selected)     # scale the 20 selected

    pred     = int(model.predict(scaled)[0])            # 0 or 1
    label    = "Benign" if pred == 1 else "Malignant"

    # probability (SVM with probability=True or decision_function)
    try:
        proba = model.predict_proba(scaled)[0].tolist()
        confidence = float(max(proba)) * 100
    except AttributeError:
        dec = float(model.decision_function(scaled)[0])
        confidence = min(abs(dec) * 20, 100)   # rough mapping
        proba = None

    return jsonify({
        "prediction": pred,
        "label":      label,
        "confidence": round(confidence, 2),
        "proba":      proba,
    })


if __name__ == "__main__":
    print("\n  Cancer Classification UI")
    print("  Open  http://127.0.0.1:5000  in your browser\n")
    app.run(debug=True, port=5000)
