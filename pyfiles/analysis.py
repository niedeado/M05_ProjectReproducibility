import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

ITER_BOUND = 100
PRINT_BOUND = 10

def get_labels_analysis(y_true, y_pred, labels_inv_map):
    labels_order = [labels_inv_map[i] for i in range(len(labels_inv_map))]
    labels_true = [labels_inv_map[i] for i in y_true]
    labels_predict = [labels_inv_map[i] for i in y_pred]
    return labels_true, labels_predict, labels_order