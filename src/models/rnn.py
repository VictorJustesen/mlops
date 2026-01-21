# PyTorch RNN models for time series forecasting

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar


class PriceLSTM(pl.LightningModule):
    """
    LSTM-based model for time series forecasting using PyTorch Lightning.
    Predicts the next value in a sequence.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        """
        Forward pass for LSTM.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            Tensor: Output tensor of shape (batch_size, 1)
        """
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use last time step's output
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)  # (batch, 1)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class PriceGRU(pl.LightningModule):
    """
    GRU-based model for time series forecasting using PyTorch Lightning.
    Predicts the next value in a sequence.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        """
        Forward pass for GRU.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            Tensor: Output tensor of shape (batch_size, 1)
        """
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)  # (batch, 1)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def get_default_callbacks():
    """Return a list of useful PyTorch Lightning callbacks for training."""
    checkpoint_cb = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")
    progress_bar_cb = RichProgressBar()
    return [checkpoint_cb, lr_monitor_cb, progress_bar_cb]
