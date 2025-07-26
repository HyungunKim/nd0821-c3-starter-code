from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    logger.info("Training model...")
    # Initialize a RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    # Fit the model to the training data
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    logger.info("Running inference...")
    preds = model.predict(X)
    return preds


def compute_model_metrics_on_slices(model, X, y, categorical_features):
    """
    Computes model metrics on slices of the data for each categorical feature.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : pd.DataFrame
        Data used for prediction.
    y : np.array
        Known labels, binarized.
    categorical_features : list
        List of categorical feature names.

    Returns
    -------
    slice_metrics : pd.DataFrame
        DataFrame containing metrics for each slice.
    """
    logger.info("Computing metrics on slices...")
    slice_metrics = []

    # Convert X to DataFrame if it's a numpy array
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()

    # Make predictions
    preds = inference(model, X)

    # Compute overall metrics
    precision, recall, fbeta = compute_model_metrics(y, preds)
    slice_metrics.append({
        'feature': 'Overall',
        'slice': 'Overall',
        'precision': precision,
        'recall': recall,
        'fbeta': fbeta,
        'samples': len(y)
    })

    # Compute metrics for each categorical feature
    for feature in categorical_features:
        if feature in X_df.columns:
            # Get unique values for the feature
            unique_values = X_df[feature].unique()

            for value in unique_values:
                # Get indices for this slice
                indices = X_df[feature] == value

                # Skip if there are no samples in this slice
                if sum(indices) == 0:
                    continue

                # Get predictions and true values for this slice
                slice_preds = preds[indices]
                slice_y = y[indices]

                # Compute metrics for this slice
                precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)

                # Add to results
                slice_metrics.append({
                    'feature': feature,
                    'slice': value,
                    'precision': precision,
                    'recall': recall,
                    'fbeta': fbeta,
                    'samples': sum(indices)
                })

    # Convert to DataFrame
    slice_metrics_df = pd.DataFrame(slice_metrics)
    logger.info("Slice metrics computation completed")

    return slice_metrics_df
