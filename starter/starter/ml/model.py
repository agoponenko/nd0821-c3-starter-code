"""
File with functions to train ML model, make inference, evaluation using classification metrics.
Author: Andrei Goponenko
Date: 5 April 2025
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


# Optional: implement hyperparameter tuning.
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

    model = RandomForestClassifier(max_depth=100, n_estimators=30, random_state=0)
    model.fit(X_train, y_train)
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
    fbeta = round(fbeta_score(y, preds, beta=1, zero_division=1), 4)
    precision = round(precision_score(y, preds, zero_division=1), 4)
    recall = round(recall_score(y, preds, zero_division=1), 4)
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
    preds = model.predict(X)
    return preds

def compute_metrics_for_slice(data, y_true, y_pred, slice_feature):
    """Calculate metrics for slice and save them in the text file.

    Inputs
    ------
    data : pandas Dataframe
        Input dataframe
    y_true : np.array
        True labels
    y_pred : np.array
        Predictions from the model
    slice_feature : str
        Name of the feature to calculate the metrics for
    Returns
    -------
    None    
    """
    with open("../model/slice_output.txt", "w") as f:
        print(f"Performance on slice of data using feature: {slice_feature} \n", file=f)
        for slice_value in data[slice_feature].unique():
            slice_index = data.index[data[slice_feature] == slice_value]
            print(slice_feature, '=', slice_value, file=f)
            precision, recall, fbeta = compute_model_metrics(y_true[slice_index], y_pred[slice_index])
            print(f'{precision=}, {recall=}, {fbeta=} \n', file=f)
