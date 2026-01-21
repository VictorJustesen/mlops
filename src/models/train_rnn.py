import os
import sys

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import typer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

import wandb
from src.models.rnn import PriceGRU, PriceLSTM, get_default_callbacks

sys.path.append(os.getcwd())

app = typer.Typer(add_completion=False, invoke_without_command=True)

# Optional GCS support
try:
    from google.cloud import storage  # noqa: F401

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

    # Initialize wandb run
    wandb.init(project="mlops", name=f"{cfg.model_type}_{cfg.region}_rnn", reinit=True)
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    window_size = cfg.get("window_size", DEFAULT_WINDOW_SIZE)
    input_features = cfg.get("input_features", DEFAULT_INPUT_FEATURES)

    # prepare data paths
    data_path = cfg.get("data_path", "data/grouped")
    if not os.path.isabs(data_path):
        root_dir = hydra.utils.get_original_cwd()
        data_path = os.path.join(root_dir, data_path)

    # Prepare data loaders
    root_dir = hydra.utils.get_original_cwd()
    train_csv = os.path.join(data_path, "data/rnn", f"{cfg.region}_train.csv")
    test_csv = os.path.join(data_path, "data/rnn", f"{cfg.region}_test.csv")
    train_set = SequenceDataset(train_csv, window_size=window_size, input_features=input_features)
    test_set = SequenceDataset(test_csv, window_size=window_size, input_features=input_features)
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, num_workers=7, shuffle=False
    )  # Make sure shuffle is False.
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, num_workers=7)

    # Model selection
    if cfg.model_type.lower() == "lstm":
        model = PriceLSTM(
            input_size=len(input_features),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            lr=cfg.lr,
        )
        model_name = "lstm"
    elif cfg.model_type.lower() == "gru":
        model = PriceGRU(
            input_size=len(input_features),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        model_name = "gru"
    else:
        raise ValueError("model_type must be 'lstm' or 'gru'")

    # PyTorch Lightning Trainer setup
    callbacks = get_default_callbacks()
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=callbacks,
        logger=pl.loggers.WandbLogger(
            project="mlops", name=f"{cfg.model_type}_{cfg.region}_rnn", reinit=True
        ),
    )

    # Train the model
    trainer.fit(model, train_loader, test_loader)

    # Save model checkpoint
    trainer.save_checkpoint(f"model_{model_name}_{cfg.region}.ckpt")
    wandb.finish()


@app.callback()
def main(
    batch_size: int = typer.Option(
        None, "--batch-size", "--batch_size", help="Batch size override for Hydra"
    ),
    dropout: float = typer.Option(
        None, "--dropout", "--dropout", help="Dropout override for Hydra"
    ),
    hidden_size: int = typer.Option(
        None, "--hidden-size", "--hidden_size", help="Hidden size override for Hydra"
    ),
    lr: float = typer.Option(None, "--lr", "--lr", help="Learning rate override for Hydra"),
    model_type: str = typer.Option(
        None, "--model-type", "--model_type", help="Model type override for Hydra"
    ),
    num_layers: int = typer.Option(
        None, "--num-layers", "--num_layers", help="Num layers override for Hydra"
    ),
    data_path: str = typer.Option(
        None, "--data-path", "--data_path", help="Data path override for Hydra"
    ),
):
    """Default command: runs train(). Allows Hydra config overrides via CLI options."""
    # Build sys.argv for Hydra overrides
    overrides = []
    if batch_size is not None:
        overrides.append(f"batch_size={batch_size}")
    if dropout is not None:
        overrides.append(f"dropout={dropout}")
    if hidden_size is not None:
        overrides.append(f"hidden_size={hidden_size}")
    if lr is not None:
        overrides.append(f"lr={lr}")
    if model_type is not None:
        overrides.append(f"model_type={model_type}")
    if num_layers is not None:
        overrides.append(f"num_layers={num_layers}")
    if data_path is not None:
        overrides.append(f"data_path={data_path}")
    sys.argv = [sys.argv[0]] + overrides
    train()


if __name__ == "__main__":
    app()
