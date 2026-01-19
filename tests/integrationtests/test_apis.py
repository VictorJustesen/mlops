from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


def test_read_root():
    """Test the root endpoint returns 200 and correct message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Electricity Price Forecasting API",
        "model": "No model loaded",
    }


def test_predict_endpoint_structure():
    """
    Test the structure of the predict endpoint.
    Even if the model isn't loaded (503), the API should handle the request validation correctly.
    """
    # Create dummy input data matching [Price, Load, Production] for sequence length 168
    # Just sending minimal valid data structure to check pydantic model
    dummy_input = {"features": [[10.0, 50.0, 20.0] for _ in range(168)]}

    response = client.post("/predict", json=dummy_input)

    # We expect either 200 (if model loaded) or 503 (if model file missing during test)
    # The important part is that we didn't get 422 (Validation Error)
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        assert "predicted_price" in response.json()
        assert isinstance(response.json()["predicted_price"], float)


def test_predict_invalid_input():
    """Test that invalid input returns 422 Validation Error."""
    # Missing 'features' key
    response = client.post("/predict", json={"wrong_key": []})
    assert response.status_code == 422
