import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    # If 'salary' is in the columns but not specified as the label, we should drop it
    # This handles the case where label=None but 'salary' is still in the dataframe
    if 'salary' in X.columns and (label is None or label != 'salary'):
        X = X.drop(['salary'], axis=1)

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(categorical_features, axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        # Check if y is already a numpy array
        y_values = y.values if hasattr(y, 'values') else y
        # Only fit the LabelBinarizer if y is not empty
        if len(y_values) > 0:
            y = lb.fit_transform(y_values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            # Check if y is already a numpy array
            y_values = y.values if hasattr(y, 'values') else y
            # Only transform y if it's not empty
            if len(y_values) > 0:
                y = lb.transform(y_values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
