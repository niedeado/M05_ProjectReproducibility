from sklearn.ensemble import RandomForestClassifier
import parameter_selection
import pickle

SEED = 0
HYPERPARAM_PATH = "./hyperparameters.pkl"

def train(X_train, y_train, hyperparameters_path=HYPERPARAM_PATH, pickle_dump=False):
    try:
        hyperparameters = pickle.load(open(hyperparameters_path, "rb"))
    except:
        print("hyperparameters file not found, proceeding with hyperparameter tuning...")
        hyperparameters = parameter_selection.hyperparam_tuning(X_train, y_train, pickle_dump)        
    
    rf_clf = RandomForestClassifier(**hyperparameters, random_state=SEED)
    rf_clf.fit(X_train, y_train)
    return rf_clf

