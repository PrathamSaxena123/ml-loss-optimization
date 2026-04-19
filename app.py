"""
ML Loss Optimization — Flask Web Application
=============================================
REST API + Jinja2 template server for the loss optimization dashboard.

Routes:
    GET  /              → Landing page
    GET  /train         → Training dashboard
    GET  /predict       → Prediction interface
    POST /api/train     → Run optimization, return results JSON
    POST /api/predict   → Predict single sample
    GET  /api/dataset   → Dataset info & stats
    GET  /api/compare   → Run both optimizers and return side-by-side
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import traceback

from model.optimizer import GradientDescentOptimizer, NewtonMethodOptimizer
from model.utils import load_and_split, get_dataset_info, evaluate

app = Flask(__name__)
CORS(app)  # enable CORS for local JS fetch calls

# ── Pre-load dataset once at startup ─────────────────────────────────────────
X_train, X_test, y_train, y_test, scaler, feature_names = load_and_split()

# ── Train models once (global, reusable) ─────────────────────────────

gd_model = GradientDescentOptimizer(learning_rate=0.1, max_iter=500)
gd_model.fit(X_train, y_train)

nm_model = NewtonMethodOptimizer(max_iter=50)
nm_model.fit(X_train, y_train)


# ── Page Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", dataset_info=get_dataset_info())


@app.route("/train")
def train_page():
    return render_template("train.html", feature_names=feature_names)


@app.route("/predict")
def predict_page():
    return render_template("predict.html", feature_names=feature_names)


# ── API: Train a single optimizer ────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def api_train():
    """
    Body (JSON):
        method        : "gd" | "newton"
        learning_rate : float  (GD only, default 0.1)
        max_iter      : int    (default 500)
        lam           : float  (L2 regularization, default 0.0)

    Returns:
        Training history + evaluation metrics as JSON.
    """
    try:
        body = request.get_json(force=True) or {}
        method = body.get("method", "gd").lower()
        max_iter = int(body.get("max_iter", 500))
        lam = float(body.get("lam", 0.0))

        if method == "gd":
            lr = float(body.get("learning_rate", 0.1))
            model = GradientDescentOptimizer(
                learning_rate=lr, max_iter=max_iter, lam=lam
            )
        elif method == "newton":
            model = NewtonMethodOptimizer(max_iter=max_iter, lam=lam)
        else:
            return jsonify({"error": f"Unknown method: {method}"}), 400

        result = model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        return jsonify({
            "training": result.to_dict(),
            "evaluation": metrics,
            "hyperparams": {
                "method": method,
                "max_iter": max_iter,
                "lam": lam,
                **({"learning_rate": lr} if method == "gd" else {}),
            },
        })

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ── API: Compare both optimizers ─────────────────────────────────────────────

@app.route("/api/compare", methods=["GET"])
def api_compare():
    """
    Run Gradient Descent and Newton's Method with default params,
    return side-by-side results for the comparison chart.
    """
    try:
        gd = GradientDescentOptimizer(learning_rate=0.1, max_iter=500)
        gd_result = gd.fit(X_train, y_train)
        gd_metrics = evaluate(gd, X_test, y_test)

        nm = NewtonMethodOptimizer(max_iter=50)
        nm_result = nm.fit(X_train, y_train)
        nm_metrics = evaluate(nm, X_test, y_test)

        return jsonify({
            "gradient_descent": {
                "training": gd_result.to_dict(),
                "evaluation": gd_metrics,
            },
            "newton": {
                "training": nm_result.to_dict(),
                "evaluation": nm_metrics,
            },
        })

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ── API: Predict a single sample ──────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Body (JSON):
        method   : "gd" | "newton"
        features : array of 30 floats (raw, unscaled values)

    Returns:
        prediction (0=Benign, 1=Malignant), probability, confidence
    """
    try:
        body = request.get_json(force=True) or {}
        method = body.get("method", "gd").lower()
        features = body.get("features")

        if features is None or len(features) != 30:
            return jsonify({"error": "Provide exactly 30 feature values."}), 400

        X_raw = np.array(features, dtype=float).reshape(1, -1)
        X_scaled = scaler.transform(X_raw)

        # Re-train (stateless API — for production, cache trained weights)
        # Use pre-trained model (no retraining)
        model = gd_model if method == "gd" else nm_model

        prob = float(model.predict_proba(X_scaled)[0])
        pred = int(prob >= 0.5)
        label = "Malignant" if pred == 1 else "Benign"

        return jsonify({
            "prediction": pred,
            "label": label,
            "probability": round(prob, 4),
            "confidence": round(max(prob, 1 - prob) * 100, 1),
        })

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ── API: Dataset info ─────────────────────────────────────────────────────────

@app.route("/api/dataset", methods=["GET"])
def api_dataset():
    return jsonify(get_dataset_info())


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)