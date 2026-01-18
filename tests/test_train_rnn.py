import pytest
import torch
import pandas as pd
import tempfile
from pathlib import Path


def test_train_function_exists():
    from src.models.train_rnn import train
    assert callable(train)

def test_model_initialization():
    from src.models.rnn import PriceLSTM, PriceGRU
    
    lstm = PriceLSTM(input_size=3, hidden_size=32, num_layers=1)
    assert lstm is not None
    
    gru = PriceGRU(input_size=3, hidden_size=32, num_layers=1)
    assert gru is not None


def test_model_forward_pass():
    from src.models.rnn import PriceLSTM, PriceGRU
    
    x = torch.randn(2, 10, 3)
    
    lstm = PriceLSTM(input_size=3, hidden_size=32, num_layers=1)
    output_lstm = lstm(x)
    assert output_lstm.shape == (2, 1) 
    
    gru = PriceGRU(input_size=3, hidden_size=32, num_layers=1)
    output_gru = gru(x)
    assert output_gru.shape == (2, 1)  

