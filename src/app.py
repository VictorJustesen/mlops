import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.models.rnn import PriceGRU

# Constants matching your training config
INPUT_SIZE = 3  # Price, Load, Production
HIDDEN_SIZE = 64
NUM_LAYERS = 1
MODEL_PATH = "model_gru_DE.pth" 
DEVICE = torch.device("cpu")

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup."""
    global model
    try:
        # Initialize model architecture
        model = PriceGRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
        
        # Load string weights
        if torch.cuda.is_available():
             state_dict = torch.load(MODEL_PATH)
        else:
             state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
             
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Successfully loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Warning: Model file {MODEL_PATH} not found. Prediction endpoint will fail.")
        model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    yield
    # Cleanup if needed

app = FastAPI(lifespan=lifespan)

class PredictionInput(BaseModel):
    # Expects list of [Price, Load, Production]
    features: list[list[float]] 

@app.get("/")
def read_root():
    return {"message": "Electricity Price Forecasting API"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to tensor: (batch_size=1, seq_len, input_size)
        data = torch.tensor(input_data.features).float().unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(data)
            
        return {"predicted_price": prediction.item()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))