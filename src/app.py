import glob
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Union

import pandas as pd
import torch
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

sys.path.append(os.getcwd())
from src.models.rnn import PriceGRU, PriceLSTM

# Optional GCS support
try:
    from google.cloud import storage

    HAS_GCS = True
except ImportError:
    HAS_GCS = False

# Use GPU if available, fallback to MPS (Mac), then CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


model: Union[PriceLSTM, PriceGRU, None] = None
model_info: Dict[str, Any] = {}


def download_model_from_gcs(bucket_name, gcs_path, local_dir="models"):
    """Download the latest model from GCS bucket."""
    if not HAS_GCS:
        raise ImportError("google-cloud-storage package not available")

    os.makedirs(local_dir, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all model files in the GCS path
    blobs = list(bucket.list_blobs(prefix=f"{gcs_path}/model_"))
    model_blobs = [b for b in blobs if b.name.endswith(".pth")]

    if not model_blobs:
        raise FileNotFoundError(f"No model files found in gs://{bucket_name}/{gcs_path}/")

    if len(model_blobs) > 1:
        print(
            f"Warning: Found {len(model_blobs)} models in GCS, using first one: {model_blobs[0].name}"
        )

    # Download the first model
    blob = model_blobs[0]
    local_path = os.path.join(local_dir, os.path.basename(blob.name))
    blob.download_to_filename(local_path)
    print(f"Downloaded model from gs://{bucket_name}/{blob.name} to {local_path}")

    return local_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_info
    try:
        # Determine model source
        model_source = os.environ.get("MODEL_SOURCE", "local")

        # Download from GCS if configured
        if model_source == "gcs":
            gcs_bucket = os.environ.get("GCS_MODEL_BUCKET")
            gcs_path = os.environ.get("GCS_MODEL_PATH", "models")

            if not gcs_bucket:
                raise ValueError("MODEL_SOURCE=gcs but GCS_MODEL_BUCKET not set")

            print(f"Downloading model from GCS: gs://{gcs_bucket}/{gcs_path}/")
            try:
                download_model_from_gcs(gcs_bucket, gcs_path)
            except Exception as e:
                print(f"✗ Failed to download from GCS: {e}")
                raise
        else:
            print("Using local models/ directory")

        # Load config from project root
        root_dir = os.getcwd()
        config_path = os.path.join(root_dir, "src/models/rnn_config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model_files = glob.glob("models/model_*.pth")
        if not model_files:
            print("ERROR: No model files found in models/ directory")
            model = None
            model_info = {"error": "No model files found in models/"}
            yield
            return

        if len(model_files) > 1:
            print(f"ERROR: Found {len(model_files)} model files, expected exactly 1: {model_files}")
            print("Please ensure only one trained model is present in models/")
            model = None
            model_info = {"error": f"Multiple model files found: {model_files}"}
            yield
            return

        model_path = model_files[0]
        model_name = os.path.basename(model_path).replace(".pth", "")
        parts = model_name.split("_")
        model_type = parts[1] if len(parts) > 1 else config.get("model_type")
        region = parts[2] if len(parts) > 2 else config.get("region")

        # Get model hyperparameters from config
        input_size = len(config.get("input_features", []))
        hidden_size = config.get("hidden_size")
        num_layers = config.get("num_layers")

        # Validate we have required config values
        if not all([model_type, region, input_size, hidden_size, num_layers]):
            raise ValueError(
                f"Missing required config values: model_type={model_type}, region={region}, "
                f"input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}"
            )

        # Create and load model

        if model_type == "lstm":
            model = PriceLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        elif model_type == "gru":
            model = PriceGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'lstm' or 'gru'")

        state_dict = torch.load(model_path, map_location=DEVICE)
        if model is not None:
            model.load_state_dict(state_dict)
            model.eval()
        else:
            raise RuntimeError("Model was not initialized correctly.")

        # Only set model_info AFTER successful load
        model_info = {
            "type": model_type,
            "region": region,
            "path": model_path,
            "device": str(DEVICE),
            "config": {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
            },
        }

        print(f"Successfully loaded {model_type.upper()} model for region {region}")
        print(f"Path: {model_path}")
        print(f"Device: {DEVICE}")
        print(
            f"  Architecture: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        model_info = {"error": str(e)}
    yield


app = FastAPI(lifespan=lifespan)

Instrumentator().instrument(app).expose(app)


class PredictionInput(BaseModel):
    # Expects list of [Price, Load, Production]
    features: list[list[float]]


@app.get("/")
def read_root():
    return {
        "message": "Electricity Price Forecasting API",
        "model": model_info if model_info else "No model loaded",
    }


@app.post("/predict")
def predict(input_data: PredictionInput, background_tasks: BackgroundTasks):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to tensor: (batch_size=1, seq_len, input_size)
        data = torch.tensor(input_data.features).float().unsqueeze(0)

        with torch.no_grad():
            prediction = model(data)

        pred_val = prediction.item()

        # Add background task to save data without blocking response
        background_tasks.add_task(save_prediction_log, input_data.features, pred_val)

        return {"predicted_price": pred_val}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring", response_class=HTMLResponse)
def monitoring():
    """Simple endpoint to return Evidently drift report"""
    try:
        # Use legacy as per your monitor_drift.py
        from evidently.legacy.metric_preset import DataDriftPreset
        from evidently.legacy.report import Report

        log_file = "prediction_log.csv"
        if not os.path.exists(log_file):
            return "<h1>No data logged yet</h1>"

        # Load current data
        current_data = pd.read_csv(log_file)

        # For this simple implementation, we compare the first half of logged data
        # against the second half if we don't have a static reference loaded.
        # Ideally, you load your training data reference here.
        if len(current_data) < 10:
            return "<h1>Not enough data for report</h1>"

        mid = len(current_data) // 2
        ref = current_data.iloc[:mid]
        cur = current_data.iloc[mid:]

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)

        # Return HTML report directly
        return report.get_html()

    except Exception as e:
        return f"<h1>Error generating report: {e}</h1>"


def save_prediction_log(features: list[list[float]], prediction: float):
    """Log predictions to a local CSV file (Background Task)"""
    # Flatten features because Evidently expects tabular data
    # (seq_len * n_features) columns
    flat_features = [item for sublist in features for item in sublist]

    row = {"timestamp": datetime.now().isoformat(), "prediction": prediction}
    # Add feature columns f0, f1...
    for i, f in enumerate(flat_features):
        row[f"f{i}"] = f

    df = pd.DataFrame([row])

    # Append to file (Note: In Cloud Run, this file is ephemeral and lost on restart)
    # For production, you would upload this to GCS or BigQuery
    file_path = "prediction_log.csv"
    hdr = not os.path.exists(file_path)
    df.to_csv(file_path, mode="a", header=hdr, index=False)
