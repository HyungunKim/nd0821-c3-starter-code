# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import pickle
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_model_metrics_on_slices

def main():
    """
    Main function to train and save the model.
    """
    # Create directories for model and metrics if they don't exist
    os.makedirs(os.path.join("..", "model"), exist_ok=True)
    os.makedirs(os.path.join("..", "metrics"), exist_ok=True)

    # Load data
    logger.info("Loading data...")
    try:
        data_path = os.path.join("..", "data", "clean_data.csv")
        data = pd.read_csv(data_path)

        # Clean the data by stripping spaces
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].str.strip()

        logger.info(f"Data loaded successfully with shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Define categorical features
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

    # Split data into train and test sets
    logger.info("Splitting data into train and test sets...")
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    # Process the training data
    logger.info("Processing training data...")
    X_train, y_train, encoder, lb = process_data(
        train, 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )

    # Process the test data
    logger.info("Processing test data...")
    X_test, y_test, _, _ = process_data(
        test, 
        categorical_features=cat_features, 
        label="salary", 
        training=False, 
        encoder=encoder, 
        lb=lb
    )

    # Train model
    logger.info("Training model...")
    model = train_model(X_train, y_train)

    # Make predictions on test data
    logger.info("Making predictions on test data...")
    preds = inference(model, X_test)

    # Compute metrics on test data
    logger.info("Computing metrics on test data...")
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    logger.info(f"Test metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fbeta:.4f}")

    # Compute metrics on slices
    logger.info("Computing metrics on slices...")
    # We need to convert X_test back to a DataFrame with original column names for slice metrics
    # First, get the original feature names
    feature_names = []
    # Add names for continuous features
    continuous_features = [col for col in train.columns if col not in cat_features and col != "salary"]
    feature_names.extend(continuous_features)

    # Add names for one-hot encoded features
    for feature in cat_features:
        unique_values = train[feature].unique()
        for value in unique_values:
            feature_names.append(f"{feature}_{value}")

    # Create a DataFrame with the processed test data
    X_test_df = pd.DataFrame(X_test)

    # Compute slice metrics
    slice_metrics = compute_model_metrics_on_slices(model, X_test, y_test, cat_features)

    # Save slice metrics to file
    slice_metrics_path = os.path.join("..", "metrics", "slice_metrics.csv")
    slice_metrics.to_csv(slice_metrics_path, index=False)
    logger.info(f"Slice metrics saved to {slice_metrics_path}")

    # Save model, encoder, and label binarizer
    logger.info("Saving model, encoder, and label binarizer...")
    model_path = os.path.join("..", "model", "model.pkl")
    encoder_path = os.path.join("..", "model", "encoder.pkl")
    lb_path = os.path.join("..", "model", "lb.pkl")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)

    with open(lb_path, 'wb') as f:
        pickle.dump(lb, f)

    logger.info("Model training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
