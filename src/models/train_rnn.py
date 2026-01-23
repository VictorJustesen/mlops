import os
import random
import subprocess
import sys
from datetime import datetime
from typing import Any, Type, Union

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import typer
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler
from torch.utils.data import DataLoader, Dataset

from src.models.rnn import PriceGRU, PriceLSTM, get_default_callbacks


# Utility to load model from checkpoint
def load_model(model_class, checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    return model_class.load_from_checkpoint(checkpoint_path)


# Set default seed for reproducibility
DEFAULT_SEED = 6

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

    def __init__(self, csv_paths, window_size=DEFAULT_WINDOW_SIZE, input_features=None):
        """
        Args:
            csv_paths (list or str): List of CSV file paths or a single path.
            window_size (int): Number of time steps in each input sequence.
            input_features (list): List of feature column names to use as input.
        """
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        region_dfs = []
        for path in csv_paths:
            region_name = os.path.basename(path).split("_")[0]
            df = pd.read_csv(path, parse_dates=["Date"])
            df["region"] = region_name
            region_dfs.append(df)
        # Concatenate all regions, keep region column
        all_df = pd.concat(region_dfs, ignore_index=True)
        all_df = all_df.sort_values("Date")
        self.sequences = []
        self.targets = []
        self.window_size = window_size
        # Get all unique dates in order
        unique_dates = all_df["Date"].sort_values().unique()
        # For each possible window
        for start_idx in range(len(unique_dates) - window_size):
            window_dates = unique_dates[start_idx : start_idx + window_size + 1]
            # For each region, extract window if all dates present
            for region in all_df["region"].unique():
                region_df = all_df[all_df["region"] == region]
                region_window = region_df[region_df["Date"].isin(window_dates)]
                # Only use if window is complete
                if len(region_window) == window_size + 1:
                    region_window = region_window.sort_values("Date")
                    x = region_window[input_features].values[:window_size].astype(np.float32)
                    y = region_window["Price"].values[window_size].astype(np.float32)
                    self.sequences.append(torch.tensor(x))
                    self.targets.append(torch.tensor(y))
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (input_sequence, target_value)
        """
        return self.sequences[idx], self.targets[idx]


@hydra.main(version_base=None, config_path=".", config_name="rnn_config.yaml")
def train(cfg: DictConfig):
    """
    Train an RNN model (LSTM/GRU) on a selected region's grouped data using Hydra config.
    Hydra can be overriden using command line options. e.g. --cfg.model_type=gru --cfg.regions=[DE,NP]

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    print(f"Training {cfg.model_type.upper()} on {cfg.regions} grouped data")
    print(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    seed = cfg.get("seed", DEFAULT_SEED)
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize wandb run
    try:
        wandb.init(project="mlops", name=f"{cfg.model_type}_{cfg.regions}_rnn", reinit=True)
    except Exception as e:
        print(f"Warning: WandB failed to initialize ({e}). Falling back to offline mode.")
        wandb.init(
            project="mlops", name=f"{cfg.model_type}_{cfg.regions}_rnn", reinit=True, mode="offline"
        )

    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    window_size = cfg.get("window_size", DEFAULT_WINDOW_SIZE)
    input_features = cfg.get("input_features", DEFAULT_INPUT_FEATURES)

    print("Preparing data paths and loading datasets...")
    data_path = cfg.get("data_path", "data/grouped")
    if not os.path.isabs(data_path):
        root_dir = hydra.utils.get_original_cwd()
        data_path = os.path.join(root_dir, data_path)

    # Prepare data loaders for multiple regions
    regions = cfg.get("regions", [cfg.get("region", "DE")])
    train_csvs = [os.path.join(data_path, f"{region}_train.csv") for region in regions]
    test_csvs = [os.path.join(data_path, f"{region}_test.csv") for region in regions]
    print(f"Training CSVs: {train_csvs}")
    print(f"Test CSVs: {test_csvs}")
    train_set = SequenceDataset(train_csvs, window_size=window_size, input_features=input_features)
    test_set = SequenceDataset(test_csvs, window_size=window_size, input_features=input_features)
    print(f"Number of training samples: {len(train_set)}")
    print(f"Number of test samples: {len(test_set)}")
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, num_workers=7, shuffle=False
    )  # Make sure shuffle is False.
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, num_workers=7)
    print("Data loaders ready.")

    # Model selection
    checkpoint_path = cfg.get("checkpoint_path", None)

    model_class: Type[Union[PriceLSTM, PriceGRU]]
    if cfg.model_type.lower() == "lstm":
        model_class = PriceLSTM
        model_name = "lstm"
    elif cfg.model_type.lower() == "gru":
        model_class = PriceGRU
        model_name = "gru"
    else:
        raise ValueError("model_type must be 'lstm' or 'gru'")

    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = load_model(model_class, checkpoint_path)
    else:
        print("Initializing new model instance...")
        model = model_class(
            input_size=len(input_features),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            lr=cfg.lr,
        )

    print("Setting up PyTorch Lightning Trainer...")
    callbacks = get_default_callbacks()
    profiler_type = getattr(cfg, "profiler_type", "simple")
    logger: Any
    try:
        logger = pl.loggers.WandbLogger(
            project="mlops", name=f"{cfg.model_type}_{cfg.regions}_rnn", reinit=True
        )
    except Exception as e:
        print(f"Warning: Failed to initialize WandB logger. Proceeding without it. Error: {e}")
        logger = True  # Fallback to default logger

    profiler: Any = None
    if profiler_type == "advanced":
        profiler = AdvancedProfiler()
    elif profiler_type == "simple":
        profiler = SimpleProfiler()
    # If profiler_type is 'none', profiler remains None
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=callbacks,
        logger=logger,
        profiler=profiler,
    )

    print("Starting model training...")
    trainer.fit(model, train_loader, test_loader)
    print("Training complete.")

    # Save model checkpoint in 'models' directory at repo root
    print("Saving model checkpoint...")
    models_dir = os.path.join(hydra.utils.get_original_cwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M")
    checkpoint_path = os.path.join(models_dir, f"model_{model_name}_{timestamp}.ckpt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")
    if isinstance(logger, pl.loggers.WandbLogger):
        wandb.finish()
        print("wandb run finished.")


@app.callback()
def main(
    batch_size: int = typer.Option(
        None, "--batch-size", "--batch_size", help="Batch size override for Hydra"
    ),
    dropout: float = typer.Option(None, "--dropout", help="Dropout override for Hydra"),
    hidden_size: int = typer.Option(
        None, "--hidden-size", "--hidden_size", help="Hidden size override for Hydra"
    ),
    lr: float = typer.Option(None, "--lr", help="Learning rate override for Hydra"),
    model_type: str = typer.Option(
        None, "--model-type", "--model_type", help="Model type override for Hydra"
    ),
    regions: str = typer.Option(
        None, "--regions", help="Comma-separated list of regions to train on"
    ),
    num_layers: int = typer.Option(
        None, "--num-layers", "--num_layers", help="Num layers override for Hydra"
    ),
    data_path: str = typer.Option(
        None, "--data-path", "--data_path", help="Data path override for Hydra"
    ),
    seed: int = typer.Option(None, "--seed", help="Random seed override for Hydra"),
    epochs: int = typer.Option(None, "--epochs", help="Epochs override for Hydra"),
    checkpoint_path: str = typer.Option(
        None,
        "--checkpoint-path",
        "--checkpoint_path",
        help=("Path to checkpoint for resuming/new training"),
    ),
    sweep: bool = typer.Option(False, "--sweep", help="Run a simple hyperparameter sweep"),
    profiler_type: str = typer.Option(
        "simple",
        "--profiler-type",
        help="Profiler type: 'simple', 'advanced', 'pytorch', or 'none'",
    ),
):
    """Default command: runs train(). Allows Hydra config overrides via CLI options."""

    if sweep:
        # Simple random search: runs subprocesses to avoid Hydra re-init issues
        for i in range(5):
            lr = 10 ** random.uniform(-4, -2)
            hidden_size = random.choice([32, 64, 128])
            print(f"\n--- Sweep Trial {i + 1}/5: lr={lr:.5f}, hidden_size={hidden_size} ---")
            subprocess.run(
                [
                    sys.executable,
                    sys.argv[0],
                    f"--lr={lr}",
                    f"--hidden-size={hidden_size}",
                    "--epochs=1",
                ]
            )
        return

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
    if regions is not None:
        # Accept comma-separated or bracketed regions from CLI
        if regions.startswith("[") and regions.endswith("]"):
            # Remove brackets and split
            region_list = [r.strip().strip("\"'") for r in regions[1:-1].split(",")]
        else:
            region_list = [r.strip() for r in regions.split(",")]
        overrides.append(f"regions={region_list}")
    if num_layers is not None:
        overrides.append(f"num_layers={num_layers}")
    if data_path is not None:
        overrides.append(f"data_path={data_path}")
    if seed is not None:
        overrides.append(f"seed={seed}")
    if epochs is not None:
        overrides.append(f"epochs={epochs}")
    if checkpoint_path is not None:
        overrides.append(f"checkpoint_path={checkpoint_path}")
    if profiler_type is not None:
        overrides.append(f"profiler_type={profiler_type}")
    sys.argv = [sys.argv[0]] + overrides
    train()


if __name__ == "__main__":
    app()
