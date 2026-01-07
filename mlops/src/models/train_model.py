import pandas as pd
import xgboost as xgb
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def load_data(filepath):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df

def train_and_evaluate(dataset_name):
    file_path = f"data/raw/{dataset_name}.csv"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = load_data(file_path)
    print("data loaded")
    price_col = [col for col in df.columns if 'price' in col.lower()][0]
    feature_cols = [col for col in df.columns if col != price_col]
    
    X = df[feature_cols]
    y = df[price_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    smape = 100 * np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred) + 1e-8))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Dataset: {dataset_name} | R2: {r2:.4f} | sMAPE: {smape:.2f}%")

def main():
    datasets = ['NP', 'PJM', 'BE', 'FR', 'DE']
    for dataset in datasets:
        train_and_evaluate(dataset)

if __name__ == "__main__":
    main()