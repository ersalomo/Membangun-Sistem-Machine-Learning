import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn
import os

def train_tuned_model():
    print("Starting hyperparameter tuned model training...")
    
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
    
    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Setup Hyperparameter Grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    }
    
    base_rf = RandomForestClassifier(random_state=42)
    
    # GridSearchCV
    print("Running Grid Search Hyperparameter Tuning...")
    grid_search = GridSearchCV(estimator=base_rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print(f"Best Parameters found: {best_params}")
    
    # Start manual MLflow logging run
    with mlflow.start_run(run_name="RandomForest_Tuned_Manual"):
        # Log manual hyperparameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
            
        # Log training sample counts
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        
        # 4. Predict and Evaluate
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("\n--- Tuned Model Evaluation ---")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}\n")
        
        # Log manual metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # 5. Generate and save custom artifacts
        os.makedirs("./temp_artifacts", exist_ok=True)
        
        # Artifact 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap='Blues', ax=ax)
        plt.title("Confusion Matrix - Tuned Model")
        cm_path = "./temp_artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        
        # Log Confusion Matrix artifact
        mlflow.log_artifact(cm_path, "plots")
        
        # Artifact 2: Feature Importance
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features_list = X.columns
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title("Feature Importance - RandomForest Tuned")
        plt.bar(range(X.shape[1]), importances[indices], align="center", color='skyblue')
        plt.xticks(range(X.shape[1]), [features_list[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        fi_path = "./temp_artifacts/feature_importance.png"
        plt.savefig(fi_path)
        plt.close()
        
        # Log Feature Importance artifact
        mlflow.log_artifact(fi_path, "plots")
        
        # Log Sklearn Model
        mlflow.sklearn.log_model(best_model, "model")
        print("Manual logging completed. Model and artifacts successfully saved to MLflow!")
        
        # Clean up local temporary files
        if os.path.exists(cm_path):
            os.remove(cm_path)
        if os.path.exists(fi_path):
            os.remove(fi_path)
        if os.path.exists("./temp_artifacts"):
            os.rmdir("./temp_artifacts")

if __name__ == '__main__':
    train_tuned_model()
