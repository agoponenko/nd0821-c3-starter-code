"""
Tests for main scripts to train, evaluate and
make inference for the income prediction service.
Author: Andrei Goponenko
Date: 5 April 2025
"""

from pathlib import Path
import pytest

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

cur_folder_path = Path(__file__).resolve().parent


@pytest.fixture
def data():
    """
    Function to get input data, preprocess and split one,
    get the label encoder and features encoder.
    """

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

    data = read_csv(cur_folder_path / "../data/census_cleaned.csv")
    train, test = train_test_split(
        data, test_size=0.20, stratify=data['salary'])
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    # Proces the test data with the process_data function
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb)

    return X_train, y_train, X_test, y_test, encoder, lb


@pytest.fixture
def model(data):
    """Function to get trained model."""
    X_train, y_train, _, _, _, _ = data
    model = RandomForestClassifier(
        max_depth=100,
        n_estimators=30,
        random_state=0)
    model.fit(X_train, y_train)
    return model


def test_train_model(data):
    """Test if model is trained aftet train_model is called."""
    X_train, y_train, _, _, _, _ = data
    model = train_model(X_train, y_train)
    # Check if this is a classification model
    assert isinstance(
        model, BaseEstimator) and isinstance(
        model, ClassifierMixin)
    # Check if the model was trained
    check_is_fitted(model)


def test_compute_model_metrics(data, model):
    """Test if the metrics were actually computed and
    lay between expected limits."""
    _, _, X_test, y_test, _, _ = data

    # Compute model metrics on held-out dataset
    preds = inference(model, X_test)

    # Calculate metrics and check the correctness
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_inference(data, model):
    """Test the inference function."""
    _, _, X_test, y_test, _, _ = data
    preds = inference(model, X_test)
    # Check if the length of predictions' vector is aligned with the length of
    # features' vector
    assert len(preds) == len(X_test)
