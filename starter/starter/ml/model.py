from sklearn.metrics import (
                            # ConfusionMatrixDisplay,
                            confusion_matrix,
                            fbeta_score,
                            precision_score,
                            recall_score,
                            classification_report
                            )
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
                                    # GridSearchCV,
                                    RandomizedSearchCV
                                    )
from tqdm import tqdm
from functools import reduce
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


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
    # Compute class weights manually
    classes = np.unique(y_train)  # Unique classes in the target
    print(" ")
    # print("Unique Classes: ", classes)

    class_weights = compute_class_weight(class_weight="balanced",
                                         classes=classes,
                                         y=y_train)

    # Convert to dictionary format (required by the classifier)
    class_weight_dict = dict(zip(classes, class_weights))
    # print("Class weights:", class_weight_dict)

    param_grid = {
        # 'n_estimators': [224, 546, 560, 574, 980],
        'n_estimators': [546,],
        # 'max_depth': [21, 28, 35, 56, 112],
        'max_depth': [35, 56, ],
        'max_features': [0.7, 0.85, 0.92, 0.99],
        'min_samples_leaf': [0.01, 0.03, 0.05],
        # 'min_samples_split': [0.02, 0.03, 0.05],
        'max_samples': [0.7, 0.9],
        'bootstrap': [True,],
        # 'warm_start': [True,],
    }

    print("param_grid: ", param_grid.items())

    def count_steps(x, y):
        return x*len(y)

    cv = 3
    total_fits = cv * reduce(count_steps, list(param_grid.values()), 1)
    print(" ")
    print("Total Grid Fits: ", total_fits)

    # rf_model = GridSearchCV(
    rf_model = RandomizedSearchCV(
        RandomForestClassifier(
                            random_state=42,
                            n_jobs=-1,
                            class_weight=class_weight_dict,
                            # class_weight='balanced_subsample',
                            ),
        param_grid,
        # scoring="f1_weighted",
        scoring="f1",
        verbose=1,
        cv=cv,
        # random_state=42,
        # n_iter=max(int(total_fits*0.15), 100)
        n_iter=4,
        )

    if show_progress:
        with tqdm(total=total_fits, desc="GridSearchCV Progress") as pbar:
            rf_model.fit(X_train, y_train)
            pbar.update(1)
    else:
        rf_model.fit(X_train, y_train)

    # Print the sorted feature importances
    print(" ")
    print("Model parameters: ", rf_model.get_params())
    print(" ")
    print("Best Estimator parameters: ", rf_model.best_estimator_.get_params())
    print(" ")
    print("Num Features used: ", rf_model.n_features_in_)
    print(" ")
    print("Best Estimator Features used: ",
          rf_model.best_estimator_.n_features_in_)
    """
    print(" ")
    print("Feature importances: ", feature_importances)
    """
    print(" ")
    print("Best Parameters:", rf_model.best_params_)
    print(" ")
    """
    n_features_in_ : int
    Number of features seen during fit. Only defined if best_estimator_ is
    defined (see the documentation for the refit parameter for more details)
    and that best_estimator_ exposes n_features_in_ when fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
    Names of features seen during fit. Only defined if best_estimator_ is
    defined (see the documentation for the refit parameter for more details)
    and that best_estimator_ exposes feature_names_in_ when fit.
    """

    rf_model = rf_model.best_estimator_
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
    # Compute class weights manually
    classes = np.unique(y_train)  # Unique classes in the target
    class_weights = compute_class_weight(class_weight="balanced",
                                         classes=classes,
                                         y=y_train)

    # Convert to dictionary format (required by the classifier)
    class_weight_dict = dict(zip(classes, class_weights))
    print(" ")
    print("Unique Classes: ", classes)
    print("Class weights:", class_weight_dict)

    rf_model = RandomForestClassifier(random_state=42,
                                      n_jobs=-1,
                                      class_weight=class_weight_dict,
                                      # class_weight='balanced_subsample',
                                      max_depth=28,
                                      n_estimators=546,
                                      max_features=0.55)
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


def inference(model, X, y_test=None):
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
    # X = X[:, :-1]
    predictions = model.predict(X)

    if y_test is not None:
        cm = confusion_matrix(y_test, predictions, labels=model.classes_)
        # print('CM: ', cm)
        class_names = ['<=50k', '>50k']
        print(" ")
        print("Confusion Matrix: ")
        cm_df = pd.DataFrame(cm,
                             index=[
                                 str(class_names[0]),
                                 str(class_names[1])
                                 ],
                             columns=[
                                 str(class_names[0])+'_pred',
                                 str(class_names[1])+'_pred'
                                 ])

        print(cm_df)
        print(classification_report(y_test,
                                    predictions,
                                    target_names=class_names,
                                    zero_division=0,
                                    digits=3))

    return predictions
