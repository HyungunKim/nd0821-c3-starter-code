import os
import sys
import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from starter.ml.model import train_model, compute_model_metrics, inference, compute_model_metrics_on_slices


@pytest.fixture
def sample_data():
    """
    Create a small sample dataset for testing.
    """
    data = {
        'age': [25, 30, 45, 50, 35, 40],
        'workclass': ['Private', 'Self-emp', 'Private', 'Federal-gov', 'Private', 'Self-emp'],
        'education': ['Bachelors', 'HS-grad', 'Masters', 'Doctorate', 'Bachelors', 'Masters'],
        'salary': ['<=50K', '>50K', '>50K', '<=50K', '<=50K', '>50K']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_model():
    """
    Create a simple trained model for testing.
    """
    # Create a simple dataset
    X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # Train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model, X, y


def test_train_model():
    """
    Test the train_model function.
    """
    # Create a simple dataset
    X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    # Train a model
    model = train_model(X, y)
    
    # Check that the model is a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)
    
    # Check that the model can make predictions
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_compute_model_metrics():
    """
    Test the compute_model_metrics function.
    """
    # Create some test data
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    
    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    # Check that the metrics are as expected
    assert precision == 2/3  # 2 true positives, 1 false positive
    assert recall == 2/3     # 2 true positives, 1 false negative
    assert fbeta == 2/3      # F1 score is the harmonic mean of precision and recall


def test_inference(sample_model):
    """
    Test the inference function.
    """
    # Get the sample model
    model, X, y = sample_model
    
    # Make predictions
    preds = inference(model, X)
    
    # Check that the predictions have the right shape
    assert preds.shape == y.shape
    
    # Check that the predictions are either 0 or 1
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_on_slices(sample_data):
    """
    Test the compute_model_metrics_on_slices function.
    """
    # Create a simple model
    X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Define categorical features
    cat_features = ['workclass', 'education']
    
    # Compute metrics on slices
    slice_metrics = compute_model_metrics_on_slices(model, X, y, cat_features)
    
    # Check that the slice_metrics is a DataFrame
    assert isinstance(slice_metrics, pd.DataFrame)
    
    # Check that it has the expected columns
    expected_columns = ['feature', 'slice', 'precision', 'recall', 'fbeta', 'samples']
    assert all(col in slice_metrics.columns for col in expected_columns)
    
    # Check that there's at least one row for the overall metrics
    assert len(slice_metrics) >= 1
    assert 'Overall' in slice_metrics['feature'].values