"""
Test module for the data processing functionality.
"""
import os
import sys
import pandas as pd
import numpy as np
import pytest

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from starter.ml.data import process_data


@pytest.fixture
def sample_data():
    """
    Create a small sample dataset for testing.
    """
    data = {
        'age': [25, 30, 45, 50],
        'workclass': ['Private', 'Self-emp', 'Private', 'Federal-gov'],
        'education': ['Bachelors', 'HS-grad', 'Masters', 'Doctorate'],
        'salary': ['<=50K', '>50K', '>50K', '<=50K']
    }
    return pd.DataFrame(data)


def test_process_data_without_label(sample_data):
    """
    Test process_data function without providing a label.
    """
    # Process data without label
    X, y, encoder, _ = process_data(
        sample_data,
        categorical_features=['workclass', 'education'],
        label=None,
        training=True
    )

    # Check that X has the right shape (4 rows, and columns for age + one-hot encoded categories)
    # workclass has 3 unique values, education has 4 unique values
    expected_cols = 1 + 3 + 4  # age + one-hot workclass + one-hot education
    assert X.shape == (4, expected_cols)

    # Check that y is empty
    assert len(y) == 0


def test_process_data_with_label(sample_data):
    """
    Test process_data function with a label.
    """
    # Process data with label
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=['workclass', 'education'],
        label='salary',
        training=True
    )

    # Check that X has the right shape
    expected_cols = 1 + 3 + 4  # age + one-hot workclass + one-hot education
    assert X.shape == (4, expected_cols)

    # Check that y has the right shape
    assert y.shape == (4,)

    # Check that encoder and lb are returned
    assert encoder is not None
    assert lb is not None


def test_process_data_inference(sample_data):
    """
    Test process_data function in inference mode.
    """
    # First train the encoder and lb
    _, _, encoder, lb = process_data(
        sample_data,
        categorical_features=['workclass', 'education'],
        label='salary',
        training=True
    )

    # Now use them for inference
    X_inference, y_inference, encoder_returned, lb_returned = process_data(
        sample_data,
        categorical_features=['workclass', 'education'],
        label='salary',
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Check that the returned encoder and lb are the same objects
    assert encoder_returned is encoder
    assert lb_returned is lb

    # Check that X_inference has the right shape
    expected_cols = 1 + 3 + 4  # age + one-hot workclass + one-hot education
    assert X_inference.shape == (4, expected_cols)

    # Check that y_inference has the right shape
    assert y_inference.shape == (4,)
