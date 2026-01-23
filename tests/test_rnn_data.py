import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest


def test_main_function_exists():
    from src.features.rnn_data import time_series_split_save

    assert callable(time_series_split_save)


@pytest.fixture
def temp_data_dirs():
    temp_dir = tempfile.mkdtemp()
    input_dir = Path(temp_dir) / "raw"
    output_dir = Path(temp_dir) / "processed"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    yield input_dir, output_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_csv(temp_data_dirs):
    input_dir, _ = temp_data_dirs

    # Create a simple test dataset
    dates = pd.date_range("2024-01-01", periods=10, freq="h")
    data = {
        "Date": dates,
        "Prices": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        "System load forecast": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        "Generation forecast": [80, 85, 90, 95, 100, 105, 110, 115, 120, 125],
    }
    df = pd.DataFrame(data)

    csv_path = input_dir / "BE.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def test_time_series_split_creates_files(temp_data_dirs, sample_csv):
    from src.features.rnn_data import time_series_split_save

    input_dir, output_dir = temp_data_dirs

    # Run the split function
    time_series_split_save(input_dir=str(input_dir), output_dir=str(output_dir))

    # Check that train and test files were created
    train_file = output_dir / "BE_train.csv"
    test_file = output_dir / "BE_test.csv"
    test_file = output_dir / "BE_test.csv"

    assert train_file.exists(), "Train file was not created"
    assert test_file.exists(), "Test file was not created"


def test_time_series_split_correct_split(temp_data_dirs, sample_csv):
    from src.features.rnn_data import time_series_split_save

    input_dir, output_dir = temp_data_dirs

    # Run the split with 80/20 split
    time_series_split_save(input_dir=str(input_dir), output_dir=str(output_dir), split_ratio=0.8)

    # Read the files
    train_df = pd.read_csv(output_dir / "BE_train.csv")
    test_df = pd.read_csv(output_dir / "BE_test.csv")

    # Check lengths (10 rows total: 8 train, 2 test)
    assert len(train_df) == 8, f"Expected 8 train rows, got {len(train_df)}"
    assert len(test_df) == 2, f"Expected 2 test rows, got {len(test_df)}"


def test_time_series_output_has_correct_columns(temp_data_dirs, sample_csv):
    from src.features.rnn_data import time_series_split_save

    input_dir, output_dir = temp_data_dirs

    time_series_split_save(input_dir=str(input_dir), output_dir=str(output_dir))

    train_df = pd.read_csv(output_dir / "BE_train.csv")

    # Check for standardized column names
    expected_columns = ["Date", "Price", "Load", "Production"]
    assert list(train_df.columns) == expected_columns, (
        f"Expected columns {expected_columns}, got {list(train_df.columns)}"
    )
