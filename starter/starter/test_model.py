# Script to test machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
import os
import sys
import pandas as pd
import pickle

# Add code to load in the data.
file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)
# data = pd.read_csv(os.path.join(file_dir, '../data/clean_census.csv'))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
# train, test = train_test_split(data, test_size=0.20)

'''cat_features = [
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

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)
# Train and save a model.
rf_model = train_model(X_train, y_train)

model_path = os.path.join(file_dir, '../model/rf_model.pkl')
pickle.dump(rf_model, open(model_path, 'wb'))

encoder_path = os.path.join(file_dir, '../model/encoder.pkl')
pickle.dump(encoder, open(encoder_path, 'wb'))

lb_path = os.path.join(file_dir, '../model/lb.pkl')
pickle.dump(lb, open(lb_path, 'wb'))
'''

def test_train_model():
    """
    Tests the Training of a machine learning model and returns it.

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    return rf_model
    """



def test_compute_model_metrics():
    """
    Tests the trained machine learning model using precision, recall, and F1.

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta
    """


def test_inference(model, X):
    """ 
    Tests model inferences and return the predictions.

    return  model.predict(X)
    """