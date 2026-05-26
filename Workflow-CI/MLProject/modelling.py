"""
modelling.py - MLProject Entry Point for CI/CD Training Pipeline
This script is designed to be called by MLflow Projects (mlflow run .).
It accepts hyperparameters via argparse and logs everything to MLflow.
"""
import pandas as pd
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import mlflow
import mlflow.sklearn
import os


def train_model(n_estimators, max_depth):
    """Train a RandomForest model with given hyperparameters and log to MLflow."""
    print(f"Training with n_estimators={n_estimators}, max_depth={max_depth}")

    # Load preprocessed data
    data_path = os.path.join(os.path.dirname(__file__), "wine_preprocessed.csv")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['quality'])
    y = df['quality']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name=f"RF_n{n_estimators}_d{max_depth}"):
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            print(f"  {name}: {value:.4f}")

        # Artifact: Confusion Matrix
        os.makedirs("artifacts", exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap='Blues', ax=ax)
        plt.title("Confusion Matrix")
        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path, "plots")

        # Artifact: Feature Importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title("Feature Importance")
        plt.bar(range(X.shape[1]), importances[indices], color='skyblue')
        plt.xticks(range(X.shape[1]),
                   [X.columns[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        fi_path = "artifacts/feature_importance.png"
        plt.savefig(fi_path)
        plt.close()
        mlflow.log_artifact(fi_path, "plots")

        # Log model
        mlflow.sklearn.log_model(model, "model")
        print("Model and artifacts logged to MLflow successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()
    train_model(args.n_estimators, args.max_depth)
