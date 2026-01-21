import os

import numpy as np
import pandas as pd
import wandb
import xgboost as xgb
import yaml
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def load_data(filepath):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def train_and_evaluate(dataset_name):
    file_path = f"data/raw/{dataset_name}.csv"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load XGBoost config
    config_path = os.path.join(os.path.dirname(__file__), "xgb_config.yaml")
    with open(config_path, "r") as f:
        xgb_config = yaml.safe_load(f)

    # Initialize wandb run
    wandb.init(project="mlops", name=f"{dataset_name}_xgb", reinit=True)
    wandb.config.update(xgb_config)
    wandb.config.update({"dataset": dataset_name})

    df = load_data(file_path)
    print("data loaded")
    price_col = [col for col in df.columns if "price" in col.lower()][0]
    feature_cols = [col for col in df.columns if col != price_col]

    X = df[feature_cols]
    y = df[price_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBRegressor(**xgb_config)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    smape = 100 * np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred) + 1e-8))
    r2 = r2_score(y_test, y_pred)

    # Log metrics to wandb
    wandb.log({"r2": r2, "smape": smape})

    print(f"Dataset: {dataset_name} | R2: {r2:.4f} | sMAPE: {smape:.2f}%")

    wandb.finish()


def main():
    datasets = ["NP", "PJM", "BE", "FR", "DE"]
    for dataset in datasets:
        train_and_evaluate(dataset)


if __name__ == "__main__":
    main()
