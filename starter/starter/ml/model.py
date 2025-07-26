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


def compute_model_metrics_on_slices(model, X, y, categorical_features, encoder, lb):
    """
    Computes model metrics on slices of the data for each categorical feature.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : pd.DataFrame
        Original data used for prediction (before one-hot encoding).
    y : np.array
        Known labels, binarized.
    categorical_features : list
        List of categorical feature names.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.

    Returns
    -------
    slice_metrics : pd.DataFrame
        DataFrame containing metrics for each slice.
    """
    logger.info("Computing metrics on slices...")
    slice_metrics = []

    # Import process_data here to avoid circular dependency if ml.data imports ml.model
    from .data import process_data

    # Compute overall metrics on the full processed data
    # First, process the full X using the provided encoder and lb
    X_processed_overall, y_processed_overall, _, _ = process_data(
        X.copy(), categorical_features=categorical_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    preds_overall = inference(model, X_processed_overall)
    precision_overall, recall_overall, fbeta_overall = compute_model_metrics(y_processed_overall, preds_overall)
    slice_metrics.append({
        'feature': 'Overall',
        'slice': 'Overall',
        'precision': precision_overall,
        'recall': recall_overall,
        'fbeta': fbeta_overall,
        'samples': len(y_processed_overall)
    })
    logger.info(f"Overall metrics computed: Precision={precision_overall:.4f}, Recall={recall_overall:.4f}, Fbeta={fbeta_overall:.4f}")

    # Compute metrics for each categorical feature
    logger.info(f"Categorical features to slice by: {categorical_features}")
    for feature in categorical_features:
        logger.info(f"Processing feature: {feature}")
        # Ensure the feature exists in the original DataFrame
        if feature in X.columns:
            unique_values = X[feature].unique()
            logger.info(f"Unique values for {feature}: {unique_values}")

            for value in unique_values:
                logger.info(f"  Processing slice: {feature}={value}")
                # Filter the original DataFrame for the current slice
                X_slice_original = X[X[feature] == value]
                y_slice_original = y[X[feature] == value]

                # Skip if there are no samples in this slice
                if len(X_slice_original) == 0:
                    logger.info(f"    Skipping empty slice: {feature}={value}")
                    continue

                # Process the slice data using the trained encoder and lb
                X_slice_processed, _, _, _ = process_data(
                    X_slice_original.copy(), 
                    categorical_features=categorical_features, 
                    label=None, 
                    training=False, 
                    encoder=encoder, 
                    lb=lb
                )

                # Make predictions on the processed slice
                slice_preds = inference(model, X_slice_processed)

                # Compute metrics for this slice
                precision, recall, fbeta = compute_model_metrics(y_slice_original, slice_preds)
                logger.info(f"    Metrics for {feature}={value}: Precision={precision:.4f}, Recall={recall:.4f}, Fbeta={fbeta:.4f}, Samples={len(y_slice_original)}")

                # Add to results
                slice_metrics.append({
                    'feature': feature,
                    'slice': value,
                    'precision': precision,
                    'recall': recall,
                    'fbeta': fbeta,
                    'samples': len(y_slice_original)
                })

    # Convert to DataFrame
    slice_metrics_df = pd.DataFrame(slice_metrics)
    logger.info("Slice metrics computation completed")

    return slice_metrics_df
