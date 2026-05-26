import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys

def automated_preprocessing(raw_data_path, output_data_path):
    """
    Automates the preprocessing steps:
    1. Loading the dataset
    2. Dropping duplicates
    3. Filling missing values (imputing pH and citric_acid with median values)
    4. Scaling features using StandardScaler
    5. Saving the preprocessed dataset
    """
    print(f"Loading raw dataset from: {raw_data_path}")
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw dataset path {raw_data_path} does not exist!")
        sys.exit(1)
        
    df = pd.read_csv(raw_data_path)
    
    # 1. Remove duplicate rows
    df_clean = df.drop_duplicates().copy()
    print(f"Removed duplicates. Records remaining: {len(df_clean)} (from {len(df)})")
    
    # 2. Impute missing values with medians
    impute_cols = ['citric_acid', 'pH']
    for col in impute_cols:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"Imputed missing values in '{col}' with median: {median_val:.4f}")
            
    # 3. Standardize features
    if 'quality' in df_clean.columns:
        features = df_clean.drop(columns=['quality'])
        target = df_clean['quality']
    else:
        features = df_clean
        target = None
        
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Convert scaled features back to DataFrame
    df_preprocessed = pd.DataFrame(scaled_features, columns=features.columns)
    
    if target is not None:
        df_preprocessed['quality'] = target.reset_index(drop=True)
        
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_data_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    df_preprocessed.to_csv(output_data_path, index=False)
    print(f"Successfully saved preprocessed dataset to: {output_data_path}")
    return df_preprocessed

if __name__ == '__main__':
    # Default paths for execution
    raw_path = '../wine_quality_raw.csv'
    output_path = './namadataset_preprocessing/wine_preprocessed.csv'
    
    # If parameters are provided as arguments
    if len(sys.argv) > 2:
        raw_path = sys.argv[1]
        output_path = sys.argv[2]
        
    automated_preprocessing(raw_path, output_path)
    
    # Also save to Membangun_model folder and Workflow-CI for convenience of subsequent criteria
    membangun_model_path = '../../Membangun_model/wine_preprocessed.csv'
    automated_preprocessing(raw_path, membangun_model_path)
    
    ci_path = '../../Workflow-CI/MLProject/wine_preprocessed.csv'
    automated_preprocessing(raw_path, ci_path)
