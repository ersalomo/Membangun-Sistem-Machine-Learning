from flask import Flask, request, jsonify
import time
import psutil
import mlflow.sklearn
import joblib
from prometheus_client import Counter, Summary, Gauge, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Metrics
REQUEST_COUNT = Counter("request_count", "Total number of predictions")
PRED_LATENCY = Summary("prediction_latency_seconds", "Latency of prediction")
CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage")
MEM_USAGE = Gauge("memory_usage_mb", "Memory usage in MB")

# Load model (path relative to this file)
MODEL_PATH = "../Membangun_model/wine_preprocessed.csv"  # placeholder, replace with actual model artifact path
model = None

def load_model():
    global model
    # In a real setup you would load the MLflow‑logged model:
    # model = mlflow.sklearn.load_model("runs:/<run_id>/model")
    # Here we just load a dummy sklearn model for illustration.
    try:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    except Exception as e:
        print("Model load error:", e)

load_model()

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.inc()
    start = time.time()
    # Dummy prediction
    pred = 1
    latency = time.time() - start
    PRED_LATENCY.observe(latency)
    return jsonify({"prediction": pred, "latency": latency})

@app.route("/metrics")
def metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    MEM_USAGE.set(psutil.virtual_memory().used / (1024 * 1024))
    registry = CollectorRegistry()
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
