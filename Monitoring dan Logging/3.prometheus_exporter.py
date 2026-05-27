"""
prometheus_exporter.py - Flask-based ML Model Serving with Prometheus Metrics
Exposes 10+ custom metrics for Advance-level monitoring (Kriteria 4).

Metrics exposed:
 1. request_count          - Total prediction requests (Counter)
 2. prediction_latency     - Prediction latency in seconds (Histogram)
 3. prediction_result      - Count of predictions by class label (Counter with label)
 4. model_load_time        - Time taken to load the model (Gauge)
 5. model_version_info     - Model version identifier (Info/Gauge)
 6. cpu_usage_percent      - System CPU usage percentage (Gauge)
 7. memory_usage_mb        - System memory usage in MB (Gauge)
 8. disk_usage_percent     - Disk usage percentage (Gauge)
 9. active_requests        - Currently in-flight requests (Gauge)
10. error_count            - Total prediction errors (Counter)
11. input_feature_mean     - Mean of input features per request (Gauge)
12. request_payload_size   - Size of the incoming request payload in bytes (Summary)
"""

from flask import Flask, request, jsonify
import time
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import psutil
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    generate_latest, CONTENT_TYPE_LATEST
)

app = Flask(__name__)

# ──────────────────────────────────────────────
# Prometheus Metrics (12 total → Advance level)
# ──────────────────────────────────────────────

# 1. Total requests
REQUEST_COUNT = Counter(
    "ml_request_count_total",
    "Total number of prediction requests received"
)

# 2. Prediction latency histogram
PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Histogram of prediction latencies in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# 3. Prediction result by class
PREDICTION_RESULT = Counter(
    "ml_prediction_result_total",
    "Prediction results broken down by predicted class",
    ["predicted_class"]
)

# 4. Model load time
MODEL_LOAD_TIME = Gauge(
    "ml_model_load_time_seconds",
    "Time taken to load the ML model"
)

# 5. Model version
MODEL_VERSION = Gauge(
    "ml_model_version",
    "Current model version number"
)

# 6. CPU usage
CPU_USAGE = Gauge(
    "ml_cpu_usage_percent",
    "Current system CPU usage percentage"
)

# 7. Memory usage
MEMORY_USAGE = Gauge(
    "ml_memory_usage_mb",
    "Current system memory usage in megabytes"
)

# 8. Disk usage
DISK_USAGE = Gauge(
    "ml_disk_usage_percent",
    "Current system disk usage percentage"
)

# 9. Active (in-flight) requests
ACTIVE_REQUESTS = Gauge(
    "ml_active_requests",
    "Number of currently processing requests"
)

# 10. Error count
ERROR_COUNT = Counter(
    "ml_error_count_total",
    "Total number of prediction errors"
)

# 11. Input feature mean
INPUT_FEATURE_MEAN = Gauge(
    "ml_input_feature_mean",
    "Mean value of input features for the latest request"
)

# 12. Request payload size
REQUEST_PAYLOAD_SIZE = Summary(
    "ml_request_payload_bytes",
    "Size of the incoming request payload in bytes"
)

# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
model = None
FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol"
]

def load_model():
    """Load the trained model.joblib. Raise FileNotFoundError if missing (production-grade)."""
    global model
    start = time.time()

    # Search paths for the actual trained model
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "model.joblib"),
        os.path.join(os.path.dirname(__file__), "..", "Membangun_model", "model.joblib"),
        os.path.join(os.path.dirname(__file__), "model.joblib"),
        "model.joblib"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"Model loaded successfully from: {path}")
            break

    if model is None:
        raise FileNotFoundError(
            "CRITICAL: Real trained model.joblib was not found in expected paths. "
            "Exporter startup halted to prevent silent failure."
        )

    elapsed = time.time() - start
    MODEL_LOAD_TIME.set(elapsed)
    MODEL_VERSION.set(1.0)
    print(f"Model loaded in {elapsed:.4f}s")

load_model()


# ──────────────────────────────────────────────
# System metrics updater
# ──────────────────────────────────────────────
def update_system_metrics():
    """Refresh CPU / Memory / Disk gauges."""
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    MEMORY_USAGE.set(psutil.virtual_memory().used / (1024 * 1024))
    DISK_USAGE.set(psutil.disk_usage("/").percent)


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return jsonify({
        "service": "Wine Quality ML Prediction API",
        "endpoints": {
            "/predict": "POST - Send JSON with feature values",
            "/metrics": "GET  - Prometheus metrics endpoint",
            "/health":  "GET  - Health check",
        }
    })


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    ACTIVE_REQUESTS.inc()
    REQUEST_COUNT.inc()

    try:
        payload = request.get_data()
        REQUEST_PAYLOAD_SIZE.observe(len(payload))

        data = request.get_json(force=True)
        features = data.get("features", None)

        if features is None:
            ERROR_COUNT.inc()
            ACTIVE_REQUESTS.dec()
            return jsonify({"error": "Missing 'features' key in JSON body"}), 400

        arr = np.array(features).reshape(1, -1)
        INPUT_FEATURE_MEAN.set(float(np.mean(arr)))

        start = time.time()
        prediction = model.predict(arr)
        proba = model.predict_proba(arr)
        latency = time.time() - start

        PREDICTION_LATENCY.observe(latency)
        predicted_class = int(prediction[0])
        PREDICTION_RESULT.labels(predicted_class=str(predicted_class)).inc()

        ACTIVE_REQUESTS.dec()
        return jsonify({
            "prediction": predicted_class,
            "probability": proba[0].tolist(),
            "latency_seconds": round(latency, 6),
        })

    except Exception as e:
        ERROR_COUNT.inc()
        ACTIVE_REQUESTS.dec()
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
def metrics():
    update_system_metrics()
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting ML Serving on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
