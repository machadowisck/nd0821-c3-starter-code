# Script to train machine learning model.
from sklearn.model_selection import train_test_split, StratifiedKFold

# Add the necessary imports for the starter code.
from ml.data import process_data  # , convert_numeric_columns
from ml.model import train_model, inference, compute_model_metrics, grid_train_model
import os
import sys
import pandas as pd
import numpy as np
import pickle
import argparse

# Add code to load in the data.
file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)
data = pd.read_csv(os.path.join(file_dir, '../data/clean_census.csv'))

keys = data.columns
print("Training feature Names:", keys)

# Optional enhancement, use K-fold cross validation
# instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
# X_train, y_train, encoder, lb = process_data(
X, y, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
print("Training features:", X.shape)

'''
# Stratified K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = []
for train_index, test_index in kf.split(X, y):
    # Split data into training and testing folds
    X_train_, X_test_ = X[train_index], X[test_index]
    y_train_, y_test_ = y[train_index], y[test_index]

    # Train the model
    rf_model = train_model(X_train_, y_train_)

    # Make predictions
    y_pred = inference(rf_model, X_test_)

    # Evaluate the model
    precision, recall, fbeta = compute_model_metrics(y_test_, y_pred)
    fold_metrics.append((precision, recall, fbeta))

# Aggregate metrics
fold_metrics = np.array(fold_metrics)
mean_metrics = fold_metrics.mean(axis=0)

print(f"Mean Precision: {mean_metrics[0]:.3f}")
print(f"Mean Recall: {mean_metrics[1]:.3f}")
print(f"Mean F-Beta: {mean_metrics[2]:.3f}")
# ########################

print("Expected number of features:", rf_model.n_features_in_)
'''

def static_train():
    # Train and save a model.
    rf_model = train_model(X, y)
    return rf_model

def grid_train(show_progress):
    # Train and save a model.
    rf_model = grid_train_model(X, y, show_progress=show_progress)
    return rf_model


def test_model(rf_model):
    global encoder, lb, cat_features
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    print("Testing features:", X_test.shape)
    y_pred = inference(rf_model, X_test)

    precision, recall, fbeta  = compute_model_metrics(y_test, y_pred)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F-Beta: {fbeta:.3f}")

def save_model(rf_model):
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

    # Call the appropriate function based on the argument
    if args.train_method == 'static':
        rf_model = static_train()
    elif args.train_method == 'grid':
        show_progress = True if args.show_progress else False
        rf_model = grid_train(show_progress=show_progress)
    test_model(rf_model)
    save_model(rf_model)
