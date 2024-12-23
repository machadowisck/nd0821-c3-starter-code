from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from functools import reduce


def grid_train_model(X_train, y_train, show_progress=False):
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

    param_grid = {
        'n_estimators': [648, 756],
        # 'max_depth': [108, 216],
        'min_samples_split': [3, 4, 5],
        'min_samples_leaf': [4, 5, 6],
    }
    cv = 4
    rf_model = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
                            param_grid,
                            scoring="f1",
                            verbose=1,
                            cv=cv)

    print("param_grid.values(): ", param_grid.values())
    for item in list(param_grid.values()):
        print(type(item), item, len(item))

    def count_steps(x,y):
        return x*len(y)

    total_param_grid =  reduce(count_steps, list(param_grid.values()), 1)
    total_fits = cv * total_param_grid
    print("Total Grid Fits: ", total_fits)

    if show_progress:
        with tqdm(total=total_fits, desc="GridSearchCV Progress") as pbar:
            rf_model.fit(X_train, y_train)
            pbar.update(1)
    else:
        rf_model.fit(X_train, y_train)
    print(" ")
    print("Model parameters: ", rf_model.get_params())
    print(" ")
    print("Best Parameters:", rf_model.best_params_)

    return rf_model


# Optional: implement hyperparameter tuning.
# see grid_train_model above
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

    rf_model = RandomForestClassifier(random_state=42,
                                      n_jobs=-1,
                                      class_weight='balanced',
                                      # max_depth=108,
                                      min_samples_leaf=4,
                                      min_samples_split=2,
                                      n_estimators=216)
    rf_model.fit(X_train, y_train)
    print(" ")
    print("Model parameters: ", rf_model.get_params())
    print(" ")

    return rf_model


def compute_model_metrics(y, preds):
    """
    Validates the trained model using precision, recall, and F1.

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
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
    # return model.predict(X[:, :108])
