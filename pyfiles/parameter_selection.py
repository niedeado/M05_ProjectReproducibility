import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

N_ESTIMATORS = np.arange(50,201,50)
CRITERIA = ["entropy"]
MAX_DEPTHS = [None] + list(range(9,31,7))
MAX_FEATURES = ["auto", "sqrt", "log2"]

N_SPLITS = 5
VAL_SIZE = 0.1
SEED = 0

