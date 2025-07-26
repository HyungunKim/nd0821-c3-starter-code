import os
import sys
import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient
import pickle
import os
import sys

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import run_local_inference
from main import app
from starter.ml.data import process_data
from starter.ml.model import train_model

# Define paths for model artifacts relative to the project root
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
LB_PATH = os.path.join(MODEL_DIR, "lb.pkl")

@pytest.fixture(scope="module")
def setup_model_artifacts():
    """
    Sets up dummy model artifacts in a temporary directory for testing.
    This fixture will run once per module.
    """
    # Create a temporary directory for model artifacts
    temp_model_dir = os.path.join(MODEL_DIR, "temp_test_model")
    if not os.path.exists(temp_model_dir):
        os.makedirs(temp_model_dir)

    temp_model_path = os.path.join(temp_model_dir, "model.pkl")
    temp_encoder_path = os.path.join(temp_model_dir, "encoder.pkl")
    temp_lb_path = os.path.join(temp_model_dir, "lb.pkl")

    # Create dummy data for training a simple model
    data = {
        'age': [25, 30, 45, 50, 35, 40],
        'workclass': ['Private', 'Self-emp', 'Private', 'Federal-gov', 'Private', 'Self-emp'],
        'fnlgt': [100000, 150000, 200000, 250000, 120000, 180000],
        'education': ['Bachelors', 'HS-grad', 'Masters', 'Doctorate', 'Bachelors', 'Masters'],
        'education-num': [13, 9, 14, 16, 13, 14],
        'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-civ-spouse', 'Never-married', 'Married-civ-spouse'],
        'occupation': ['Adm-clerical', 'Exec-managerial', 'Prof-specialty', 'Exec-managerial', 'Sales', 'Prof-specialty'],
        'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Husband', 'Not-in-family', 'Husband'],
        'race': ['White', 'Black', 'Asian-Pac-Islander', 'White', 'White', 'White'],
        'sex': ['Male', 'Male', 'Female', 'Male', 'Female', 'Male'],
        'capital-gain': [0, 0, 0, 0, 0, 0],
        'capital-loss': [0, 0, 0, 0, 0, 0],
        'hours-per-week': [40, 45, 50, 60, 35, 40],
        'native-country': ['United-States', 'United-States', 'India', 'United-States', 'United-States', 'United-States'],
        'salary': ['<=50K', '>50K', '>50K', '>50K', '<=50K', '>50K']
    }
    df = pd.DataFrame(data)

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(
        df, categorical_features=categorical_features, label="salary", training=True
    )

    model = train_model(X, y)

    # Save artifacts to the temporary directory
    with open(temp_model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(temp_encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    with open(temp_lb_path, 'wb') as f:
        pickle.dump(lb, f)

    # Pass temporary paths to the tests
    yield temp_model_path, temp_encoder_path, temp_lb_path

    # Teardown: remove dummy artifacts and the temporary directory
    os.remove(temp_model_path)
    os.remove(temp_encoder_path)
    os.remove(temp_lb_path)
    os.rmdir(temp_model_dir)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_inference_data():
    """
    Provides a sample data point for inference.
    """
    return pd.DataFrame([
        {
            "age": 35,
            "workclass": "Private",
            "fnlgt": 200000,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        }
    ])

@pytest.fixture
def sample_api_data():
    """
    Provides a sample data point in API request format.
    """
    return {
        "age": 35,
        "workclass": "Private",
        "fnlgt": 200000,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

def test_local_inference(setup_model_artifacts, sample_inference_data):
    """
    Test the run_local_inference function with temporary model artifacts.
    """
    model_path, encoder_path, lb_path = setup_model_artifacts
    
    # Run inference with the temporary artifacts
    predicted_label = run_local_inference(
        sample_inference_data,
        model_path=model_path,
        encoder_path=encoder_path,
        lb_path=lb_path
    )
    
    assert predicted_label is not None
    assert isinstance(predicted_label, np.ndarray)
    assert predicted_label.shape == (1,)
    assert predicted_label[0] in ['<=50K', '>50K']

def test_inference_consistency_with_api(setup_model_artifacts, client, sample_inference_data, sample_api_data, monkeypatch):
    """
    Test that local inference and API inference produce consistent results.
    """
    model_path, encoder_path, lb_path = setup_model_artifacts

    # 1. Perform local inference
    local_prediction = run_local_inference(
        sample_inference_data,
        model_path=model_path,
        encoder_path=encoder_path,
        lb_path=lb_path
    )
    assert local_prediction is not None
    local_label = local_prediction[0]

    # 2. Perform API inference
    # Monkeypatch the main module to use the temporary model artifacts
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    with open(lb_path, 'rb') as f:
        lb = pickle.load(f)

    monkeypatch.setattr("main.model", model)
    monkeypatch.setattr("main.encoder", encoder)
    monkeypatch.setattr("main.lb", lb)

    response = client.post("/predict", json=sample_api_data)
    assert response.status_code == 200
    api_prediction_data = response.json()
    api_label = api_prediction_data["prediction"]

    # 3. Compare results
    assert local_label == api_label
