from sklearn.ensemble import RandomForestClassifier
from . import parameter_selection

SEED = 0
HYPERPARAMS = {'n_estimators': 200,
               'max_depth': None,
               'max_features': 'log2'}


def train(X_train, y_train, hyperparameters=HYPERPARAMS, pickle_dump=False):
    """Function that generates a trained RandomForestClassifier model

    Parameters
    ==========

    X_train : numpy.ndarray
        Training data matrix.

    y_train : numpy.ndarray
        Training ground truth vector.

    hyperparameters : dict

        Dictionary with RandomForestClassifier arguments as keys.
        If a None value is passed as argument, hyperparameter tuning
        is conducted.

    pickle_dump: bool

        Boolean variable, if True it pickle dumps best hyperparameters
        in case hyperparameter tuning is conducted.


    Returns
    =======

    rf_clf : RandomForestClassifier
        Trained RandomForestClassifier model

    """

    if hyperparameters is None:
        hyperparameters = parameter_selection.hyperparam_tuning(X_train, y_train, pickle_dump)

    assert isinstance(hyperparameters, dict)
    assert 'n_estimators' in hyperparameters
    assert 'max_depth' in hyperparameters
    assert 'max_features' in hyperparameters

    rf_clf = RandomForestClassifier(**hyperparameters, random_state=SEED)
    rf_clf.fit(X_train, y_train)
    return rf_clf

