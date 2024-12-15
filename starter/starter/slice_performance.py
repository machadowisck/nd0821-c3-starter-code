import sys
import os
import pickle
import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, inference
import logging 

def performance(model, data, slice_feature, categorical_features=[]):
    """
    Output the performance of the model on slices of the data

    Inputs
    ------
    model : Machine learning model
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    data : pd.DataFrame
        Dataframe containing the features and label.
    slice_feature: str
        Name of the feature used to make slices (categorical features)
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    Returns
    -------
    None
    """
    logger.info('## {} feature performance'.format(slice_feature))

    X, y, _, _ = process_data(
        data, categorical_features=categorical_features, label="salary", training=True
        )
    preds = inference(model, X)

    for value in data[slice_feature].unique():
       slice_index = data.index[data[slice_feature] == value]
       
       logger.info(slice_feature, '=', value)
       logger.info('data size:', len(slice_index))
       logger.info('precision: {}, recall: {}, fbeta: {}'.format(*compute_model_metrics(y[slice_index], preds[slice_index])))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)-15s %(message)s",
                        filename='slice_output.txt')
    logger = logging.getLogger()


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
    file_dir = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(file_dir, '../data/clean_census.csv'))

    model_path = os.path.join(file_dir, '../model/rf_model.pkl')
    model = pickle.load(open(model_path, 'rb'))

    
    for category in cat_features:
        performance(model, data, category, categorical_features=cat_features)
