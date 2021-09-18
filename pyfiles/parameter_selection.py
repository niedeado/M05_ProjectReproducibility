import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import pickle

N_ESTIMATORS = np.arange(50,201,50)
CRITERIA = ["entropy"]
MAX_DEPTHS = [None] + list(range(9,31,7))
MAX_FEATURES = ["auto", "sqrt", "log2"]

N_SPLITS = 5
VAL_SIZE = 0.1
SEED = 0

def hyperparam_tuning(X_train, y_train):

    print("----------------------------------------")
    print("HYPERPARAMETER TUNING:")
    print("n_estimators:", N_ESTIMATORS)
    print("criteria", CRITERIA)
    print("max_depths:", MAX_DEPTHS)
    print("max_features:", MAX_FEATURES)
    print("----------------------------------------")

    validation_accs = []
    parameters = []

    for n_est in N_ESTIMATORS:
        for crit in CRITERIA:
            for max_depth in MAX_DEPTHS:
                for max_feat in MAX_FEATURES:
                    
                    t = time.time()
                    params = {"n_estimators": n_est, "criterion": crit,
                              "max_depth": max_depth, "max_features": max_feat}
                    print(params)
                    parameters.append(params)
                    
                    sss = StratifiedShuffleSplit(n_splits=N_SPLITS,
                                                 test_size=VAL_SIZE,
                                                 random_state=SEED)
                    
                    cv_scores = []
                    for train_index_cv, test_index_cv in sss.split(X_train, y_train):
                        
                        X_train_cv, X_test_cv = X[train_index_cv], X[test_index_cv]
                        y_train_cv, y_test_cv = y[train_index_cv], y[test_index_cv]
                        
                        rf_clf = RandomForestClassifier(**params,
                                                        random_state=SEED)
                        
                        rf_clf.fit(X_train_cv, y_train_cv)
                        cv_scores.append(rf_clf.score(X_test_cv,y_test_cv))
                    
                    print("Val score:", np.mean(cv_scores))
                    validation_accs.append(np.mean(cv_scores))
                    print("Computation time:", time.time()-t)
                    print("----------------------------------------")
    
    best_params = parameters[np.argmax(validation_accs)]
    return best_params