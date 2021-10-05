import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


DATA_DIR = "./data"
DATA_TXT = "data_Sha_64.txt"
TEST_SIZE = 0.2
SEED = 0


def load():
    """Loads data from a txt file and returns it as a pandas dataframe

    Returns
    =======

    dataset : pandas.DataFrame

        Pandas dataframe with n_samples rows and
        n_features+1 columns.
        The first columns indicates the plant species,
        considered as class in the classification problem.

    """

    dataset = pd.read_csv(os.path.join(DATA_DIR, DATA_TXT), header=None)
    dataset.columns = ["species"] + \
                      ["shape_"+str(i) for i in range(dataset.shape[1]-1)]
    return dataset

def extract_data_array(dataset):
    """Extracts arrays and label maps from pandas dataframe dataset
    
    Parameters
    ==========

    dataset : pandas.DataFrame

        Pandas dataframe with n_samples rows and
        n_features+1 columns.
        The first columns indicates the plant species,
        considered as class in the classification problem.

    
    Returns
    =======

    X : numpy.ndarray
        Data matrix of shape (n_samples, n_features).

    y : numpy.ndarray
        Ground truth vector of shape (n_samples,).

    labels_inv_map : dict

        Dictionary that maps from integer class represention
        to plant species representation.

    labels_map : dict

        Dictionary that maps from plant species representation 
        to integer class represention.

    """
    
    labels_str = list(dataset.species.unique())
    labels_map = dict(zip(labels_str, list(range(len(labels_str)))))
    labels_inv_map = {num: name for name, num in labels_map.items()}
    
    X = dataset.drop("species",axis=1).to_numpy()
    y = np.array([labels_map[i] for i in dataset.species])
    
    return X, y, labels_inv_map, labels_map

def split_data(X, y, test_size=TEST_SIZE, random_state=SEED):
    """Splits data into training and test sets.

    Parameters
    ==========

    X : numpy.ndarray
        Data matrix of shape (n_samples, n_features).

    y : numpy.ndarray
        Ground truth vector of shape (n_samples,).

    test_size : float
        Percentage of data points directed to the test set.

    random_state : int
        Random seed, necessary for reproducibility

    
    Returns
    =======

    X_train : numpy.ndarray
        Training data matrix.

    X_test : numpy.ndarray
        Testing data matrix.

    y_train : numpy.ndarray
        Training ground truth vector.

    y_test : numpy.ndarray
        Testing ground truth vector.

    """

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=test_size,
                                                      stratify=y,
                                                      random_state=random_state)
    return X_train, X_test, y_train, y_test
