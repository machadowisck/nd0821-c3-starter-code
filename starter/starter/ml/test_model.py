# Script to test machine learning model.

# Add the necessary imports for the starter code.
from starter.starter.ml.model import (train_model,
                                      inference,
                                      compute_model_metrics)
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier


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
    assert all([
        abs(precision - 1) < 0.01,
        abs(recall - 0.5) < 0.01,
        abs(fbeta - 0.67) < 0.01
    ])


def test_inference():
    """
    Tests model inferences and return the predictions.

    return  model.predict(X)
    """
    X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [0, 1, 1, 0]
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    result = inference(clf, [[0, 0], [0, 1]])
    assert list(result) == [0, 1]


def test_local_model_inference_le():
    pass


def test_local_model_inference_gt():
    pass
