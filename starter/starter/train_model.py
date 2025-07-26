# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Add the necessary imports for the starter code.
# Import the process_data function from the ml directory
from ml.data import process_data

# Add code to load in the data.
# For example:
# data = pd.read_csv(os.path.join("data", "census.csv"))

# Comment out the following line until you've defined the 'data' variable above
# train, test = train_test_split(data, test_size=0.20)

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

# Comment out the following lines until you've defined the 'train' variable above
# X_train, y_train, encoder, lb = process_data(
#     train, categorical_features=cat_features, label="salary", training=True
# )

# Proces the test data with the process_data function.

# Train and save a model.
