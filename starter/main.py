import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Optional
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import functions from the ML module
from starter.ml.data import process_data
from starter.ml.model import inference

# Create FastAPI app
app = FastAPI(
    title="Census Income Prediction API",
    description="API for predicting whether income exceeds $50K/yr based on census data",
    version="1.0.0"
)

# Define the categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Define the input data model with example
class CensusItem(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }

# Define the output data model
class PredictionResponse(BaseModel):
    prediction: str
    probability: float

# Define global variables for model artifacts
model = None
encoder = None
lb = None

# Load the model, encoder, and label binarizer
@app.on_event("startup")
async def startup_event():
    global model, encoder, lb

    try:
        # Get the absolute path to the project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, "starter/model", "model.pkl")
        encoder_path = os.path.join(project_root, "starter/model", "encoder.pkl")
        lb_path = os.path.join(project_root, "starter/model", "lb.pkl")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)

        with open(lb_path, 'rb') as f:
            lb = pickle.load(f)

        logger.info("Model, encoder, and label binarizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model, encoder, or label binarizer: {e}")
        # We'll continue and let the API start, but prediction endpoints will fail
        model = None
        encoder = None
        lb = None

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Census Income Prediction API"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(item: CensusItem):
    if model is None or encoder is None or lb is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    try:
        # Convert input data to DataFrame
        data_dict = item.dict(by_alias=True)
        df = pd.DataFrame([data_dict])

        # Process the input data
        X, _, _, _ = process_data(
            df,
            categorical_features=cat_features,
            training=False,
            encoder=encoder,
            lb=lb
        )

        # Make prediction
        prediction = inference(model, X)

        # Get probability
        probability = model.predict_proba(X)[0][1]

        # Convert prediction to label
        prediction_label = ">50K" if prediction[0] == 1 else "<=50K"

        return {"prediction": prediction_label, "probability": float(probability)}

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
