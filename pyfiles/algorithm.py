from sklearn.ensemble import RandomForestClassifier
import parameter_selection
import pickle

SEED = 0
HYPERPARAMS = {'n_estimators': 200,
               'max_depth': 20,
               'max_features': 'log2'}

def train(X_train, y_train, hyperparameters=HYPERPARAMS, pickle_dump=False):
    if hyperparameters is None:
        hyperparameters = parameter_selection.hyperparam_tuning(X_train, y_train, pickle_dump)

    assert isinstance(hyperparameters, dict)
    assert 'n_estimators' in hyperparameters
    assert 'max_depth' in hyperparameters
    assert 'max_features' in hyperparameters
    
    rf_clf = RandomForestClassifier(**hyperparameters, random_state=SEED)
    rf_clf.fit(X_train, y_train)
    return rf_clf

