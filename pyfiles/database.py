import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


DATA_DIR = "../data"
DATA_TXT = "data_Sha_64.txt"
TEST_SIZE = 0.2
SEED = 0


def load():
    dataset = pd.read_csv(os.path.join(DATA_DIR, DATA_TXT), header=None)
    dataset.columns = ["species"] + \
                      ["shape_"+str(i) for i in range(dataset.shape[1]-1)]
    return dataset