"""
Script to train machine learning model.
Author: Andrei Goponenko
Date: 5 April 2025
"""

import logging
import pickle

from pandas import read_csv
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.model import (train_model, compute_model_metrics,
                      inference, compute_metrics_for_slice)
from ml.data import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
data_input = read_csv("../data/census.csv")
logger.info("Downloaded input dataset")

# Clean data
data = data_input.copy()
data.columns = [col.strip() for col in data.columns]
df_obj = data.select_dtypes('object')
data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
data.to_csv("../data/census_cleaned.csv")
logger.info("Cleaned input dataset and saved one")

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20, stratify=data['salary'])

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Save encoder and  label binarizer
pickle.dump(encoder, open("../model/encoder.pickle", "wb"))
pickle.dump(lb, open("../model/lb.pickle", "wb"))

# Proces the test data with the process_data function
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",
    training=False, encoder=encoder, lb=lb)
logger.info("Input data is preprocessed and preprocessing artifacts are saved")

# Train and save a model
model = train_model(X_train, y_train)
pickle.dump(model, open("../model/model.pickle", "wb"))
logger.info("Model is trained and saved")

# Compute model metrics on held-out dataset
preds = inference(model, X_test)
logger.info("Model inference is finished")

# Save metrics to txt file
precision, recall, fbeta = compute_model_metrics(y_test, preds)
with open("../model/model_metrics.txt", "w") as f:
    print(f"Precision: {precision:.4f}", file=f)
    print(f"Recall: {recall:.4f}", file=f)
    print(f"Fbeta: {fbeta:.4f}", file=f)
logger.info("Metrics for the whole dataset are evaluated and saved")

# Get metrics for slice of data
test = test.reset_index(drop=True)
feature_for_data_slice_metrics = 'education'
compute_metrics_for_slice(test, y_test, preds, 'education')
logger.info(f"Metrics for data slice are evaluated and saved \
    for the following feature(-s): {feature_for_data_slice_metrics}")
