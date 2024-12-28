# Script to train machine learning model.
# from sklearn.model_selection import StratifiedKFold
# import numpy as np

# Add the necessary imports for the starter code.
from ml.data import process_data  # , convert_numeric_columns
from ml.model import (train_model,
                      inference,
                      compute_model_metrics,
                      grid_train_model)
import os
import sys
import pandas as pd
import pickle
import argparse
from sklearn.model_selection import train_test_split

# Add code to load in the data.
file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)
data = pd.read_csv(os.path.join(file_dir, '../data/clean_census.csv'))

# Optional enhancement, use K-fold cross validation
# instead of a train-test split.
train, test = train_test_split(
                                data,
                                test_size=0.3,
                                random_state=42,
                                stratify=data['salary'],
                                )
'''
X_train_, X_test_, y_train_, y_test_ = train_test_split(
                                                        data.drop(columns='salary'),
                                                        data['salary'],
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=data['salary'],
                                                        )
'''

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


def static_train(X, y):
    # Train and save a model.
    # global X_train_, y_train_
    rf_model = train_model(X, y)
    return rf_model


def grid_train(X, y, show_progress):
    # Train and save a model.
    # global X_train_, y_train_
    rf_model = grid_train_model(X, y, show_progress=show_progress)
    return rf_model


def test_model(rf_model, encoder, lb):
    global cat_features, test
    # X_test_, y_test_, test

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label='salary',
        training=False,
        encoder=encoder,
        lb=lb
    )
    # y_test = y_test_
    print("Testing features:", X_test.shape)
    y_pred = inference(rf_model, X_test, y_test=y_test)

    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F-Beta: {fbeta:.3f}")


def save_model(rf_model, encoder, lb):
    global file_dir
    model_path = os.path.join(file_dir, '../model/rf_model.pkl')
    pickle.dump(rf_model, open(model_path, 'wb'))

    encoder_path = os.path.join(file_dir, '../model/encoder.pkl')
    pickle.dump(encoder, open(encoder_path, 'wb'))

    lb_path = os.path.join(file_dir, '../model/lb.pkl')
    pickle.dump(lb, open(lb_path, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose the training method.")

    # Add an argument to specify which training method to use
    parser.add_argument(
        '--train_method',
        choices=['static', 'grid'],
        required=True,
        help="Choose between static_train and grid_train"
    )
    parser.add_argument(
        '--show_progress',
        action='store_true',
        help="Enable progress display"
    )
    args = parser.parse_args()

    X, y, encoder, lb = process_data(
        train, categorical_features=cat_features, label='salary', training=True
        # X_train_,categorical_features=cat_features,label=None,training=True
    )
    print("Training features(shape):", X.shape)

    # Call the appropriate function based on the argument
    if args.train_method == 'static':
        rf_model = static_train(X, y)
    elif args.train_method == 'grid':
        show_progress = True if args.show_progress else False
        rf_model = grid_train(X, y, show_progress=show_progress)

    test_model(rf_model, encoder, lb)
    save_model(rf_model, encoder, lb)
