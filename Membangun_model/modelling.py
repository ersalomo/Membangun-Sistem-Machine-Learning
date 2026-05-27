"""
modelling.py - Dasar Model Training dengan MLflow Autolog
Kriteria 2: Membangun Model Machine Learning
- Menggunakan mlflow.autolog() untuk logging otomatis
- Logging manual tambahan: metrics, artifacts (confusion matrix, feature importance)
- Model disimpan ke mlruns dan model.joblib lokal
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.sklearn
import joblib
import os


def train_basic_model():
    print("Starting basic model training (with MLflow autolog + manual artifacts)...")

    # Set tracking URI ke local mlruns
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Wine_Quality_Basic_Experiment")

    # 1. Load preprocessed data
    data_path = "wine_preprocessed.csv"
    if not os.path.exists(data_path):
        print(f"Error: Preprocessed data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=['quality'])
    y = df['quality']

    # 2. Split train/test (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Enable MLflow Autologging (logs params, metrics, model automatically)
    mlflow.sklearn.autolog(log_models=True, log_input_examples=True)

    with mlflow.start_run(run_name="RandomForest_Autolog"):
        # 4. Instantiate and train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 5. Predict and evaluate
        y_pred = model.predict(X_test)

        # Determine average method based on number of unique classes
        n_classes = len(np.unique(y))
        avg_method = 'binary' if n_classes == 2 else 'weighted'

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        recall    = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1        = f1_score(y_test, y_pred, average=avg_method, zero_division=0)

        print("\n--- Basic Model Evaluation ---")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"Classes  : {list(np.unique(y))}")

        # 6. Manual metric logging (selain autolog)
        mlflow.log_metric("manual_accuracy",  accuracy)
        mlflow.log_metric("manual_precision", precision)
        mlflow.log_metric("manual_recall",    recall)
        mlflow.log_metric("manual_f1_score",  f1)
        mlflow.log_param("avg_method", avg_method)

        # 7. Artifact: Confusion Matrix
        os.makedirs("./temp_artifacts", exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=ax)
        plt.title("Confusion Matrix - Basic RandomForest")
        plt.tight_layout()
        cm_path = "./temp_artifacts/confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path, "plots")

        # 8. Artifact: Feature Importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = X.columns.tolist()

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(feature_names)), importances[indices], color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_title("Feature Importance - Basic RandomForest", fontsize=14, fontweight='bold')
        ax.set_xlabel("Features", fontsize=12)
        ax.set_ylabel("Importance Score", fontsize=12)
        plt.tight_layout()
        fi_path = "./temp_artifacts/feature_importance.png"
        plt.savefig(fi_path, dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(fi_path, "plots")

        # 9. Log model with signature (manual, backup dari autolog)
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            model, "model_manual",
            signature=signature,
            input_example=X_test.iloc[:3]
        )

        # 10. Simpan model lokal untuk serving
        joblib.dump(model, "model.joblib")
        print("\nModel saved locally to model.joblib")
        print("All artifacts and metrics logged to MLflow successfully!\n")

        # Cleanup temp files
        for f in [cm_path, fi_path]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists("./temp_artifacts"):
            try:
                os.rmdir("./temp_artifacts")
            except OSError:
                pass


if __name__ == '__main__':
    train_basic_model()
