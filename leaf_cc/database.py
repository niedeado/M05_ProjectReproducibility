import pandas as pd
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import pkg_resources

DATA_DIR = "./data"
DATA_TXT = "data_Sha_64.txt"
TEST_SIZE = 0.2
SEED = 0


def load():
    print("CWD", os.getcwd())
    print("SYS", sys.argv[0])
    print("REALPATH", os.path.dirname(os.path.realpath('__file__')))
    try:
        dataset = pd.read_csv(os.path.join(DATA_DIR, DATA_TXT), header=None)
    except FileNotFoundError:
        DATAFILE = pkg_resources.resource_filename(__name__, "data_Sha_64.txt")
        dataset = pd.read_csv(DATAFILE, header=None)

    dataset.columns = ["species"] + \
                      ["shape_"+str(i) for i in range(dataset.shape[1]-1)]
    return dataset

def extract_data_array(dataset):
    
    labels_str = list(dataset.species.unique())
    labels_map = dict(zip(labels_str, list(range(len(labels_str)))))
    labels_inv_map = {num: name for name, num in labels_map.items()}
    
    X = dataset.drop("species",axis=1).to_numpy()
    y = np.array([labels_map[i] for i in dataset.species])
    
    return X, y, labels_inv_map, labels_map

def split_data(X, y, test_size=TEST_SIZE, random_state=SEED):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=test_size,
                                                      stratify=y,
                                                      random_state=random_state)
    return X_train, X_test, y_train, y_test