"""
modelling_tuning.py - Hyperparameter Tuning dengan GridSearchCV + MLflow Manual Logging & DagsHub Integration
Kriteria 2 (Advance):
- Menggunakan GridSearchCV untuk hyperparameter tuning.
- Koneksi online ke DagsHub dengan auto-parsing DagsHub.txt.
- Manual MLflow logging secara penuh (TIDAK MENGGUNAKAN autolog).
- Log parameter, metrik, dan minimal 5 artefak tambahan (autolog + minimal 2 artefak).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.sklearn
import joblib


def setup_dagshub_mlflow():
    """
    Mengatur environment variables untuk MLflow tracking menggunakan DagsHub secara otomatis.
    Membaca langsung dari DagsHub.txt jika environment variable belum di-set.
    """
    # 1. Cek apakah environment variables sudah di-set
    if os.environ.get("MLFLOW_TRACKING_URI") and os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        print("MLflow credentials detected in environment variables. Using them.")
        return True

    # 2. Cari file DagsHub.txt di direktori aktif atau direktori script berada
    dagshub_paths = [
        "DagsHub.txt",
        os.path.join(os.path.dirname(__file__), "DagsHub.txt"),
        "../DagsHub.txt"
    ]
    
    found_file = None
    for path in dagshub_paths:
        if os.path.exists(path):
            found_file = path
            break

    if found_file:
        print(f"Loading DagsHub configuration from {found_file}...")
        try:
            with open(found_file, "r") as f:
                lines = f.readlines()
            
            credentials_set = 0
            for line in lines:
                line = line.strip()
                if line.startswith("export "):
                    # Parsing format: export KEY="VALUE"
                    expr = line.replace("export ", "")
                    parts = expr.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().strip('"').strip("'")
                        os.environ[key] = value
                        credentials_set += 1
            
            if credentials_set >= 3:
                print("DagsHub environment variables initialized successfully!")
                return True
        except Exception as e:
            print(f"Warning: Failed to parse {found_file}: {e}")
    
    print("Warning: DagsHub.txt not found or incomplete. Falling back to local MLflow tracking.")
    return False


def train_tuned_model():
    print("=" * 65)
    print("  Starting Hyperparameter Tuned Model Training with DagsHub MLflow")
    print("=" * 65)

    # Setup tracking URI
    is_online = setup_dagshub_mlflow()
    if is_online:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        print(f"MLflow Tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")
    else:
        mlflow.set_tracking_uri("file:./mlruns")
        print("MLflow Tracking URI: (Local) ./mlruns")

    # Set nama eksperimen
    mlflow.set_experiment("Wine_Quality_Tuning_Experiment")

    # 1. Load preprocessed data
    data_path = os.path.join("namadataset_preprocessing", "wine_preprocessed.csv")
    if not os.path.exists(data_path):
        data_path = "wine_preprocessed.csv"
        if not os.path.exists(data_path):
            print("Error: Preprocessed data not found at namadataset_preprocessing/wine_preprocessed.csv or wine_preprocessed.csv")
            return

    print(f"Loading preprocessed dataset from: {data_path}")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['quality'])
    y = df['quality']

    # Cek jumlah kelas
    unique_classes = sorted(y.unique().tolist())
    n_classes = len(unique_classes)
    avg_method = 'binary' if n_classes == 2 else 'weighted'
    print(f"Dataset Shape   : {df.shape}")
    print(f"Target Classes  : {unique_classes} -> using average='{avg_method}'")

    # 2. Split train/test (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Setup Hyperparameter Grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    }

    base_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    scoring = 'f1' if n_classes == 2 else 'f1_weighted'

    print(f"\nRunning Grid Search CV (scoring='{scoring}', folds=5)...")
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model  = grid_search.best_estimator_
    best_cv_score = grid_search.best_score_

    print(f"\nBest Parameters: {best_params}")
    print(f"Best CV Score  : {best_cv_score:.4f}")

    # 4. Start MLflow Run (Manual Logging)
    # Kami TIDAK mengaktifkan autolog untuk memenuhi kriteria manual logging
    with mlflow.start_run(run_name="RandomForest_GridSearchCV_Manual"):
        
        print("\nLogging parameters and metrics to MLflow...")
        # --- LOG PARAMETERS ---
        for key, val in best_params.items():
            mlflow.log_param(key, val)
        
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", scoring)
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("avg_method", avg_method)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("model_type", "RandomForestClassifier")

        # --- EVALUATE BEST MODEL ---
        y_pred = best_model.predict(X_test)
        
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        recall    = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1        = f1_score(y_test, y_pred, average=avg_method, zero_division=0)

        print("\n--- Best Model Evaluation Metrics ---")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        # --- LOG METRICS ---
        mlflow.log_metric("best_cv_score", best_cv_score)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # --- GENERATE & LOG ARTIFACTS (5 ARTEFAK) ---
        os.makedirs("./temp_artifacts", exist_ok=True)
        print("\nGenerating and logging 5 manual artifacts...")

        # Artefak 1: Confusion Matrix Plot (PNG)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=ax)
        plt.title(f"Confusion Matrix - Tuned RF (F1={f1:.4f})", fontsize=12, fontweight='bold')
        plt.tight_layout()
        cm_path = "./temp_artifacts/confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path, "plots")

        # Artefak 2: Feature Importance Plot (PNG)
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = X.columns.tolist()

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(feature_names)))
        ax.bar(range(len(feature_names)), importances[indices], color=colors, alpha=0.85)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right', fontsize=10)
        ax.set_title("Feature Importance - Tuned RandomForest", fontsize=14, fontweight='bold')
        ax.set_xlabel("Features", fontsize=12)
        ax.set_ylabel("Importance Score", fontsize=12)
        plt.tight_layout()
        fi_path = "./temp_artifacts/feature_importance.png"
        plt.savefig(fi_path, dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(fi_path, "plots")

        # Artefak 3: CV Results (CSV)
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        cv_cols = [
            'param_n_estimators', 'param_max_depth', 'param_min_samples_split',
            'mean_test_score', 'std_test_score', 'rank_test_score'
        ]
        cv_cols_exist = [c for c in cv_cols if c in cv_results_df.columns]
        cv_results_path = "./temp_artifacts/cv_results.csv"
        cv_results_df[cv_cols_exist].sort_values('rank_test_score').to_csv(cv_results_path, index=False)
        mlflow.log_artifact(cv_results_path, "tuning")

        # Artefak 4: Classification Report (TXT)
        report_str = classification_report(y_test, y_pred, zero_division=0)
        report_path = "./temp_artifacts/classification_report.txt"
        with open(report_path, "w") as f:
            f.write("=== Classification Report - GridSearchCV Best Model ===\n\n")
            f.write(report_str)
        mlflow.log_artifact(report_path, "reports")

        # Artefak 5: Dataset Summary Metadata (JSON)
        dataset_meta = {
            "dataset_shape": list(df.shape),
            "target_column": "quality",
            "class_distribution": df['quality'].value_counts().to_dict(),
            "feature_names": feature_names,
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0]
        }
        meta_path = "./temp_artifacts/dataset_summary.json"
        with open(meta_path, "w") as f:
            json.dump(dataset_meta, f, indent=4)
        mlflow.log_artifact(meta_path, "metadata")

        # --- LOG MODEL WITH SIGNATURE ---
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            best_model,
            "model",
            signature=signature,
            input_example=X_test.iloc[:3]
        )

        # 5. Simpan model secara lokal untuk kebutuhan deployment kriteria selanjutnya
        joblib.dump(best_model, "model.joblib")
        print("\n[SUCCESS] Model saved locally to model.joblib")
        print("[SUCCESS] All metrics and 5 advanced artifacts logged to MLflow/DagsHub successfully!")

        run_id = mlflow.active_run().info.run_id
        print(f"DagsHub MLflow Run ID: {run_id}")

        # --- CLEANUP TEMP ARTIFACTS ---
        for f in [cm_path, fi_path, cv_results_path, report_path, meta_path]:
            if os.path.exists(f):
                os.remove(f)
        try:
            os.rmdir("./temp_artifacts")
        except OSError:
            pass


if __name__ == '__main__':
    train_tuned_model()
