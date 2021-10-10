import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import pickle

N_ESTIMATORS = [50, 100, 200]
MAX_DEPTHS = [None, 10, 20]
MAX_FEATURES = ["sqrt", "log2"]

N_SPLITS = 5
VAL_SIZE = 0.1
SEED = 0


def selection_criteria(parameters, validation_accs):
    """Returns hyperparameters having the best validation accuracy

    Parameters
    ==========

    parameters : list
        List of hyperparameters (dicts)

    validation_accs : list
        List of validation accuracies (floats)


    Returns
    =======

    best_params : dict
        Hyperparameters having the highest validation accuracy

    """

    assert len(parameters) == len(validation_accs)
    return parameters[np.argmax(validation_accs)]


def hyperparam_tuning(X_train, y_train, pickle_dump=False):
    """Grid search hyperparameters tuning via cross validation

    Parameters
    ==========

    X_train : numpy.ndarray
        Training data matrix.

    y_train : numpy.ndarray
        Training ground truth vector.

    pickle_dump : bool
        If true, pickle dumps best hyperparameters.


    Returns
    =======

    best_params : dict
        Hyperparameters having the highest average validation accuracy.

    """

    print("----------------------------------------")
    print("HYPERPARAMETER TUNING:")
    print("n_estimators:", N_ESTIMATORS)
    print("max_depths:", MAX_DEPTHS)
    print("max_features:", MAX_FEATURES)
    print("----------------------------------------")

    validation_accs = []
    parameters = []

    for n_est in N_ESTIMATORS:
        for max_depth in MAX_DEPTHS:
            for max_feat in MAX_FEATURES:

                t = time.time()
                params = {
                    "n_estimators": n_est,
                    "max_depth": max_depth,
                    "max_features": max_feat,
                }
                print(params)
                parameters.append(params)

                # generates n_splits preserving the percentage of samples for each class
                sss = StratifiedShuffleSplit(
                    n_splits=N_SPLITS, test_size=VAL_SIZE, random_state=SEED
                )

                cv_scores = []
                for train_index_cv, test_index_cv in sss.split(X_train, y_train):
                    X_train_cv, X_test_cv = (
                        X_train[train_index_cv],
                        X_train[test_index_cv],
                    )
                    y_train_cv, y_test_cv = (
                        y_train[train_index_cv],
                        y_train[test_index_cv],
                    )

                    rf_clf = RandomForestClassifier(**params, random_state=SEED)

                    rf_clf.fit(X_train_cv, y_train_cv)
                    cv_scores.append(rf_clf.score(X_test_cv, y_test_cv))

                print("Val score:", np.mean(cv_scores))
                validation_accs.append(np.mean(cv_scores))
                print("Computation time:", time.time() - t)
                print("----------------------------------------")

    best_params = selection_criteria(parameters, validation_accs)

    if pickle_dump:
        pickle.dump(best_params, open("./hyperparameters.pkl", "wb"))
    return best_params
