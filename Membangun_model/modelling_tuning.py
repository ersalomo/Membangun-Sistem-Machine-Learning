"""
modelling_tuning.py - Hyperparameter Tuning dengan GridSearchCV + MLflow Manual Logging
Kriteria 2 (Advance): Menggunakan GridSearchCV untuk hyperparameter tuning
- Manual MLflow logging (bukan autolog)
- Log semua kombinasi parameter dan hasil CV
- Simpan model terbaik dengan signature
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
import os


def train_tuned_model():
    print("Starting hyperparameter tuned model training (Manual MLflow Logging)...")

    # Set the local MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Wine_Quality_Tuning_Experiment")

    # 1. Load preprocessed data
    data_path = "wine_preprocessed.csv"
    if not os.path.exists(data_path):
        print(f"Error: Preprocessed data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=['quality'])
    y = df['quality']

    # Check class distribution
    n_classes = len(y.unique())
    avg_method = 'binary' if n_classes == 2 else 'weighted'
    print(f"Classes detected: {sorted(y.unique().tolist())} → using average='{avg_method}'")

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

    # GridSearchCV with F1 (weighted for multi-class, binary for binary)
    scoring = 'f1' if n_classes == 2 else 'f1_weighted'
    print(f"\nRunning Grid Search (scoring='{scoring}') with {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split'])} combinations...")
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

    # 4. Start manual MLflow logging run
    with mlflow.start_run(run_name="RandomForest_GridSearchCV_Manual"):

        # --- Log hyperparameters (best) ---
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_param("cv_folds",         5)
        mlflow.log_param("scoring",          scoring)
        mlflow.log_param("train_samples",    X_train.shape[0])
        mlflow.log_param("test_samples",     X_test.shape[0])
        mlflow.log_param("n_features",       X_train.shape[1])
        mlflow.log_param("avg_method",       avg_method)
        mlflow.log_param("class_weight",     "balanced")

        # --- Log best CV score ---
        mlflow.log_metric("best_cv_score", best_cv_score)

        # --- Predict and Evaluate ---
        y_pred  = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        recall    = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1        = f1_score(y_test, y_pred, average=avg_method, zero_division=0)

        print("\n--- Tuned Model Evaluation ---")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}\n")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Log metrics to MLflow
        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)

        # --- Generate Artifacts ---
        os.makedirs("./temp_artifacts", exist_ok=True)

        # Artifact 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=ax)
        plt.title(f"Confusion Matrix - GridSearchCV Best Model\n(Acc={accuracy:.3f}, F1={f1:.3f})", fontsize=13)
        plt.tight_layout()
        cm_path = "./temp_artifacts/confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path, "plots")

        # Artifact 2: Feature Importance
        importances  = best_model.feature_importances_
        indices      = np.argsort(importances)[::-1]
        feature_names = X.columns.tolist()

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
        ax.bar(range(len(feature_names)), importances[indices], color=colors, alpha=0.85)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right', fontsize=10)
        ax.set_title("Feature Importance - RandomForest (Tuned)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Features", fontsize=12)
        ax.set_ylabel("Importance Score", fontsize=12)
        plt.tight_layout()
        fi_path = "./temp_artifacts/feature_importance.png"
        plt.savefig(fi_path, dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(fi_path, "plots")

        # Artifact 3: CV Results (GridSearch scores per combination)
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        cv_cols = ['param_n_estimators', 'param_max_depth', 'param_min_samples_split',
                   'mean_test_score', 'std_test_score', 'rank_test_score']
        cv_cols_exist = [c for c in cv_cols if c in cv_results_df.columns]
        cv_results_path = "./temp_artifacts/cv_results.csv"
        cv_results_df[cv_cols_exist].sort_values('rank_test_score').to_csv(cv_results_path, index=False)
        mlflow.log_artifact(cv_results_path, "tuning")

        # Log Sklearn Model with signature
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            best_model,
            "model",
            signature=signature,
            input_example=X_test.iloc[:3]
        )

        # Save model locally for production/serving
        joblib.dump(best_model, "model.joblib")
        print("Model saved locally to model.joblib")
        print("All artifacts and metrics logged to MLflow successfully!\n")

        # Log run ID for reference
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Cleanup temp files
        for f in [cm_path, fi_path, cv_results_path]:
            if os.path.exists(f):
                os.remove(f)
        try:
            os.rmdir("./temp_artifacts")
        except OSError:
            pass


if __name__ == '__main__':
    train_tuned_model()
