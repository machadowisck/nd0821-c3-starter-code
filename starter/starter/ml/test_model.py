# Script to test machine learning model.

# Add the necessary imports for the starter code.
from starter.starter.ml.model import (train_model,
                                      inference,
                                      compute_model_metrics)
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def test_train_model():
    """
    Tests the Training of a machine learning model and returns it.

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    return rf_model
    """
    X = np.random.rand(10, 5)
    y = np.random.randint(2, size=10)
    model = train_model(X, y)
    # Check that this is a classification model
    assert isinstance(model, BaseEstimator) and isinstance(model,
                                                           ClassifierMixin)


def test_compute_model_metrics():
    """
    Tests the trained machine learning model using precision, recall, and F1.

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta
    """
    y = [1, 1, 0]
    preds = [0, 1, 0]
    precision, recall, fbeta = compute_model_metrics(y, preds)
    # Assert that the metrics are close to the expected value:
    # precision = 1.0, recall = 0.5, fbeta = 0.6667
    assert abs(precision - 1) < 0.01 and abs(recall - 0.5) < 0.01 and abs(fbeta - 0.67) < 0.01


def test_inference():
    """
    Tests model inferences and return the predictions.

    return  model.predict(X)
    """
    X = np.random.rand(10, 5)
    y = np.random.randint(2, size=1)
    model = train_model(X, y)
    pred = inference(model, X)
    # Check if pred and y shapes are equal
    assert y.shape == pred.shape
