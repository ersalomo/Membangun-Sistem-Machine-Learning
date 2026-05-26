import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os

def train_basic_model():
    print("Starting basic model training...")
    
    # Set the local MLflow tracking URI
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
    
    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Enable MLflow Autologging
    mlflow.autolog()
    
    with mlflow.start_run(run_name="RandomForest_Autolog"):
        # 3. Instantiate and train model
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # 4. Predict and evaluate
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("\n--- Basic Model Evaluation ---")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}\n")
        
        # MLflow autolog will automatically log parameters, metrics and the model artifact!

if __name__ == '__main__':
    train_basic_model()
