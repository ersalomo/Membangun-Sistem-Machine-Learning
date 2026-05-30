# 🍷 Wine Quality ML System - MLOps Pipeline

[![CI – Retrain Model](https://github.com/ersalomo/Membangun-Sistem-Machine-Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/ersalomo/Membangun-Sistem-Machine-Learning/actions/workflows/ci.yml)
[![Preprocessing Pipeline](https://github.com/ersalomo/Membangun-Sistem-Machine-Learning/actions/workflows/preprocess.yml/badge.svg)](https://github.com/ersalomo/Membangun-Sistem-Machine-Learning/actions/workflows/preprocess.yml)

Proyek akhir **Machine Learning Operations (MLOps)** yang membangun sistem ML end-to-end untuk prediksi kualitas anggur menggunakan **RandomForest Classifier** dengan pipeline lengkap dari eksperimen hingga monitoring.

---

## 📁 Struktur Repository

```
Membangun-Sistem-Machine-Learning/
│
├── 📂 Eksperimen_SML_siswa/           # Kriteria 1: Eksperimen & Preprocessing
│   ├── wine_quality_raw.csv           # Dataset mentah
│   ├── .github/workflows/
│   │   └── preprocess.yml             # GitHub Actions: otomasi preprocessing
│   └── preprocessing/
│       ├── Eksperimen_siswa.ipynb     # Notebook EDA + Preprocessing
│       ├── automate_siswa.py          # Script preprocessing otomatis
│       └── namadataset_preprocessing/
│           └── wine_preprocessed.csv  # Output preprocessing
│
├── 📂 Membangun_model/                # Kriteria 2: Membangun Model
│   ├── wine_preprocessed.csv          # Data siap latih
│   ├── modelling.py                   # Model dasar (autolog MLflow)
│   ├── modelling_tuning.py            # Model dengan GridSearchCV
│   ├── model.joblib                   # Model terlatih (untuk serving)
│   ├── DagsHub.txt                    # Link DagsHub MLflow Tracking
│   ├── requirements.txt               # Dependensi Python
│   └── screenshoot_*.png              # Bukti MLflow experiments
│
├── 📂 Workflow-CI/                    # Kriteria 3: CI/CD dengan MLflow Projects
│   ├── .github/workflows/
│   │   └── ci.yml                     # GitHub Actions: CI retraining
│   └── MLProject/
│       ├── MLProject                  # Konfigurasi MLflow Project
│       ├── conda.yaml                 # Conda environment
│       ├── modelling.py               # Entry point training CI
│       └── wine_preprocessed.csv      # Data untuk CI
│
└── 📂 Monitoring dan Logging/         # Kriteria 4: Monitoring & Logging
    ├── Dockerfile                      # Docker image ML serving
    ├── docker-compose.yml             # Orkestrasi: serving + prometheus + grafana
    ├── 2.prometheus.yml               # Konfigurasi Prometheus scraping
    ├── 3.prometheus_exporter.py       # Flask API + 12 Prometheus metrics
    ├── 7.inference.py                 # Script pengujian endpoint
    ├── requirements.txt               # Dependensi serving
    ├── model.joblib                   # Model untuk container
    ├── grafana/
    │   ├── dashboards/wine_dashboard.json
    │   └── provisioning/
    ├── 1.bukti_serving/               # Screenshot bukti serving
    ├── 4.bukti monitoring Prometheus/ # Screenshot 10+ metrics Prometheus
    ├── 5.bukti monitoring Grafana/    # Screenshot Grafana dashboard
    └── 6.bukti alerting Grafana/      # Screenshot Grafana alerts
```

---

## 🚀 Cara Menjalankan

### Prasyarat
- Python 3.9+
- Docker & Docker Compose
- Git

### 1️⃣ Clone Repository
```bash
git clone https://github.com/ersalomo/Membangun-Sistem-Machine-Learning.git
cd Membangun-Sistem-Machine-Learning
```

### 2️⃣ Install Dependensi Python
```bash
pip install -r Membangun_model/requirements.txt
```

### 3️⃣ Jalankan Preprocessing
```bash
cd Eksperimen_SML_siswa/preprocessing
python automate_siswa.py ../wine_quality_raw.csv ./namadataset_preprocessing/wine_preprocessed.csv
```

### 4️⃣ Jalankan Training Model
```bash
cd Membangun_model

# Training dasar (autolog)
python modelling.py

# Training dengan hyperparameter tuning (manual logging)
python modelling_tuning.py
```

### 5️⃣ Jalankan CI/CD dengan MLflow Projects
```bash
# Set DagsHub environment variables
export MLFLOW_TRACKING_URI="https://dagshub.com/ersalomo/Membangun-Sistem-Machine-Learning.mlflow"
export MLFLOW_TRACKING_USERNAME="ersalomo"
export MLFLOW_TRACKING_PASSWORD="<YOUR_DAGSHUB_TOKEN>"

# Run MLflow Project
mlflow run Workflow-CI/MLProject -P n_estimators=120 -P max_depth=6 --env-manager=local
```

### 6️⃣ Jalankan Monitoring Stack (Docker)
```bash
cd "Monitoring dan Logging"

# Build dan jalankan semua container
docker-compose up -d

# Verifikasi container berjalan
docker-compose ps
```

Setelah container berjalan:
- **ML Serving API**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (login tidak diperlukan)

### 7️⃣ Uji Endpoint Prediksi
```bash
# Uji health check
curl http://localhost:8000/health

# Uji prediksi (single request)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]}'

# Kirim 50 request otomatis untuk populate metrics
python "Monitoring dan Logging/7.inference.py"

# Lihat metrics Prometheus
curl http://localhost:8000/metrics
```

---

## 📊 Model Information

| Aspek | Detail |
|-------|--------|
| **Dataset** | Wine Quality (versi MLOps) |
| **Task** | Binary Classification (kualitas rendah/tinggi) |
| **Algorithm** | RandomForest Classifier |
| **Hyperparameters** | n_estimators=120, max_depth=6, class_weight=balanced |
| **Tuning** | GridSearchCV (5-fold CV, F1 scoring) |
| **MLflow Tracking** | DagsHub Remote Server |
| **Metrics** | Accuracy, Precision, Recall, F1-Score |

---

## 🔍 Prometheus Metrics (12 Custom Metrics)

| # | Metric Name | Tipe | Deskripsi |
|---|-------------|------|-----------|
| 1 | `ml_request_count_total` | Counter | Total request prediksi |
| 2 | `ml_prediction_latency_seconds` | Histogram | Latensi prediksi |
| 3 | `ml_prediction_result_total` | Counter | Hasil prediksi per kelas |
| 4 | `ml_model_load_time_seconds` | Gauge | Waktu load model |
| 5 | `ml_model_version` | Gauge | Versi model aktif |
| 6 | `ml_cpu_usage_percent` | Gauge | Penggunaan CPU |
| 7 | `ml_memory_usage_mb` | Gauge | Penggunaan memori (MB) |
| 8 | `ml_disk_usage_percent` | Gauge | Penggunaan disk |
| 9 | `ml_active_requests` | Gauge | Request aktif (in-flight) |
| 10 | `ml_error_count_total` | Counter | Total error prediksi |
| 11 | `ml_input_feature_mean` | Gauge | Rata-rata nilai fitur input |
| 12 | `ml_request_payload_bytes` | Summary | Ukuran payload request |

---

## ⚙️ GitHub Actions CI/CD

### Workflow 1: Preprocessing Pipeline
**File**: `.github/workflows/preprocess.yml` (di `Eksperimen_SML_siswa/`)  
**Trigger**: Push ke `main`  
**Tugas**: Jalankan `automate_siswa.py` dan commit hasil preprocessing

### Workflow 2: CI – Retrain Model
**File**: `.github/workflows/ci.yml` (di `Workflow-CI/`)  
**Trigger**: Push ke `main`, PR, atau manual dispatch  
**Tugas**: Retrain model dengan MLflow Projects, log ke DagsHub, upload artifacts

**Secrets yang diperlukan** (di GitHub → Settings → Secrets):
- `DAGSHUB_USERNAME` = `ersalomo`
- `DAGSHUB_TOKEN` = Token dari DagsHub Settings

---

## 🔗 Links

- **DagsHub MLflow**: https://dagshub.com/ersalomo/Membangun-Sistem-Machine-Learning.mlflow
- **GitHub Repository**: https://github.com/ersalomo/Membangun-Sistem-Machine-Learning
- **Docker Hub**: https://hub.docker.com/r/ersalomo/wine-quality-mlproject

---

## 👤 Author

**Ersalomo**  
Submission MLOps - Dicoding  