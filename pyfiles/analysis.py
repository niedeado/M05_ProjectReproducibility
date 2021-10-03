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

def visualize_report(y_true, y_pred, labels_inv_map):
    labels_true, labels_predict, labels_order = get_labels_analysis(y_true, y_pred, labels_inv_map)
    return classification_report(labels_true,labels_predict, labels =labels_order)
    

def inspect_misclassified(y_true, y_pred, labels_inv_map,
                         iter_bound=ITER_BOUND, print_bound=PRINT_BOUND):
    iter_bound = min(iter_bound, len(labels_inv_map)**2)
    
    labels_true, labels_predict, labels_order = get_labels_analysis(y_true, y_pred, labels_inv_map)
    
    cm = confusion_matrix(labels_true, labels_predict, labels=labels_order)
    idxs_cm = np.unravel_index(np.argsort(cm, axis=None)[::-1], cm.shape)

    print_count = 0
    i = 0
    misclassified_msg = []
    while (print_count < PRINT_BOUND) and (i < iter_bound) and (i < idxs_cm[0].shape[0]-1):
        i += 1
        if idxs_cm[0][i] != idxs_cm[1][i]:
            print_count += 1
            misclassified_msg.append(f"{labels_inv_map[idxs_cm[0][i]]} was predicted as {labels_inv_map[idxs_cm[1][i]]}: {cm[idxs_cm[0][i], idxs_cm[1][i]]} times")
    
    return misclassified_msg