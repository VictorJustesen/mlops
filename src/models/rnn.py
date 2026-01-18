# PyTorch RNN models for time series forecasting
import torch.nn as nn


class PriceLSTM(nn.Module):
    """
    LSTM-based model for time series forecasting.
    Predicts the next value in a sequence.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

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


class PriceGRU(nn.Module):
    """
    GRU-based model for time series forecasting.
    Predicts the next value in a sequence.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

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
