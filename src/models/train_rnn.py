import os
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.getcwd())
from src.models.rnn import PriceGRU, PriceLSTM

# Optional GCS support
try:
    from google.cloud import storage

    HAS_GCS = True
except ImportError:
    HAS_GCS = False

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

DEFAULT_WINDOW_SIZE = 168  # 1 week in hours
DEFAULT_INPUT_FEATURES = ["Price", "Load", "Production"]


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for creating sequences from time series data in a CSV file.
    """

    def __init__(self, csv_path, window_size=DEFAULT_WINDOW_SIZE, input_features=None):
        """
        Args:
            csv_path (str): Path to the CSV file.
            window_size (int): Number of time steps in each input sequence.
            input_features (list): List of feature column names to use as input.
        """
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        self.data = df[input_features].values.astype(np.float32)
        self.targets = df["Price"].values.astype(np.float32)
        self.window_size = window_size

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (input_sequence, target_value)
        """
        x = self.data[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size]
        return torch.tensor(x), torch.tensor(y)


@hydra.main(version_base=None, config_path=".", config_name="rnn_config.yaml")
def train(cfg: DictConfig):
    """
    Train an RNN model (LSTM/GRU) on a selected region's grouped data using Hydra config.
    Hydra can be overriden using command line options. e.g. --cfg.model_type=gru --cfg.region=DE

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    print(f"Training {cfg.model_type.upper()} on {cfg.region} grouped data")
    print(OmegaConf.to_yaml(cfg))

    window_size = cfg.get("window_size", DEFAULT_WINDOW_SIZE)
    input_features = cfg.get("input_features", DEFAULT_INPUT_FEATURES)

    # Prepare data paths
    data_path = cfg.get("data_path", "data/grouped")
    if not os.path.isabs(data_path):
        root_dir = hydra.utils.get_original_cwd()
        data_path = os.path.join(root_dir, data_path)

    print(f"Loading data from: {data_path}")
    train_csv = os.path.join(data_path, f"{cfg.region}_train.csv")
    test_csv = os.path.join(data_path, f"{cfg.region}_test.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training data not found at {train_csv}")

    train_set = SequenceDataset(train_csv, window_size=window_size, input_features=input_features)
    test_set = SequenceDataset(test_csv, window_size=window_size, input_features=input_features)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size)

    # Model selection
    if cfg.model_type.lower() == "lstm":
        model = PriceLSTM(
            input_size=len(input_features), hidden_size=cfg.hidden_size, num_layers=cfg.num_layers
        ).to(DEVICE)
        model_name = "lstm"
    elif cfg.model_type.lower() == "gru":
        model = PriceGRU(
            input_size=len(input_features), hidden_size=cfg.hidden_size, num_layers=cfg.num_layers
        ).to(DEVICE)
        model_name = "gru"
    else:
        raise ValueError("model_type must be 'lstm' or 'gru'")

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    statistics = {"train_loss": [], "test_loss": []}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        total_train_samples = 0
        # Training loop
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(x)
            y = y.unsqueeze(1)  # (batch, 1)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            batch_size = x.size(0)
            epoch_loss += loss.item() * batch_size
            total_train_samples += batch_size
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
        avg_loss = epoch_loss / total_train_samples if total_train_samples > 0 else float("nan")
        statistics["train_loss"].append(avg_loss)

        # Test set evaluation
        model.eval()
        test_loss = 0.0
        total_test_samples = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pred = model(x)
                y = y.unsqueeze(1)
                loss = loss_fn(y_pred, y)
                batch_size = x.size(0)
                test_loss += loss.item() * batch_size
                total_test_samples += batch_size
        avg_test_loss = test_loss / total_test_samples if total_test_samples > 0 else float("nan")
        statistics["test_loss"].append(avg_test_loss)
        print(f"Epoch {epoch} avg loss: {avg_loss:.6f} | test loss: {avg_test_loss:.6f}")
        model.train()

    print("Training complete")
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"model_{model_name}_{cfg.region}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save training plot
    plt.figure(figsize=(10, 5))
    plt.plot(statistics["train_loss"], label="Train loss")
    plt.plot(statistics["test_loss"], label="Test loss")
    plt.title(f"Train/Test loss (MSE) - {cfg.model_type.upper()} {cfg.region}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(models_dir, f"training_statistics_{model_name}_{cfg.region}.png")
    plt.savefig(plot_path)
    print(f"Training plot saved to {plot_path}")

    # Optional: Upload to GCS if configured
    gcs_bucket = os.environ.get("GCS_MODEL_BUCKET")
    gcs_path = os.environ.get("GCS_MODEL_PATH", "models")

    if gcs_bucket and HAS_GCS:
        try:
            print(f"Uploading model to GCS: gs://{gcs_bucket}/{gcs_path}/")
            client = storage.Client()
            bucket = client.bucket(gcs_bucket)

            # Upload model file
            blob = bucket.blob(f"{gcs_path}/model_{model_name}_{cfg.region}.pth")
            blob.upload_from_filename(model_path)
            print(f"Uploaded {model_path} to gs://{gcs_bucket}/{gcs_path}/")

            # Upload training plot
            plot_blob = bucket.blob(f"{gcs_path}/training_statistics_{model_name}_{cfg.region}.png")
            plot_blob.upload_from_filename(plot_path)
            print(f"Uploaded {plot_path} to gs://{gcs_bucket}/{gcs_path}/")
        except Exception as e:
            print(f"Warning: Could not upload to GCS: {e}")
    elif gcs_bucket and not HAS_GCS:
        print("Warning: GCS_MODEL_BUCKET set but google-cloud-storage not installed")
    else:
        print("GCS upload skipped (GCS_MODEL_BUCKET not set)")


if __name__ == "__main__":
    train()
