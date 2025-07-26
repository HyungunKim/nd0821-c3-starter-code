import pickle
import pandas as pd
import numpy as np
import os

from starter.ml.model import inference
from starter.ml.data import process_data

def run_local_inference(custom_data_df, model_path=None, encoder_path=None, lb_path=None):
    """
    Performs local inference on custom data using the trained model and artifacts.

    Args:
        custom_data_df (pd.DataFrame): DataFrame containing the custom data for inference.
        model_path (str, optional): Path to the model file. Defaults to "model/model.pkl".
        encoder_path (str, optional): Path to the encoder file. Defaults to "model/encoder.pkl".
        lb_path (str, optional): Path to the label binarizer file. Defaults to "model/lb.pkl".

    Returns:
        np.array: Predicted labels.
    """
    # Set default paths if not provided
    if model_path is None:
        model_path = "model/model.pkl"
    if encoder_path is None:
        encoder_path = "model/encoder.pkl"
    if lb_path is None:
        lb_path = "model/lb.pkl"

    # Adjust paths for script execution context if necessary
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    model_full_path = os.path.join(project_root, model_path) if not os.path.isabs(model_path) else model_path
    encoder_full_path = os.path.join(project_root, encoder_path) if not os.path.isabs(encoder_path) else encoder_path
    lb_full_path = os.path.join(project_root, lb_path) if not os.path.isabs(lb_path) else lb_path
    
    if not os.path.exists(model_full_path):
        print(f"Error: Model file not found at {model_full_path}")
        return None
    if not os.path.exists(encoder_full_path):
        print(f"Error: Encoder file not found at {encoder_full_path}")
        return None
    if not os.path.exists(lb_full_path):
        print(f"Error: LabelBinarizer file not found at {lb_full_path}")
        return None

    with open(model_full_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_full_path, 'rb') as f:
        encoder = pickle.load(f)
    with open(lb_full_path, 'rb') as f:
        lb = pickle.load(f)

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

    X_processed, _, _, _ = process_data(
        custom_data_df,
        categorical_features=categorical_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    predictions = inference(model, X_processed)
    predicted_labels = lb.inverse_transform(predictions)
    return predicted_labels

if __name__ == "__main__":
    # Example custom data
    example_data = pd.DataFrame([
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

    print("Running local inference with example data...")
    predicted_label = run_local_inference(example_data)
    if predicted_label is not None:
        print(f"Predicted salary: {predicted_label[0]}")

    # Another example
    example_data_2 = pd.DataFrame([
        {
            "age": 22,
            "workclass": "Private",
            "fnlgt": 150000,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Never-married",
            "occupation": "Other-service",
            "relationship": "Own-child",
            "race": "Black",
            "sex": "Female",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 30,
            "native-country": "United-States"
        }
    ])
    print("\nRunning local inference with another example data...")
    predicted_label_2 = run_local_inference(example_data_2)
    if predicted_label_2 is not None:
        print(f"Predicted salary: {predicted_label_2[0]}")
