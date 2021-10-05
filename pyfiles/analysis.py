import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

ITER_BOUND = 100
PRINT_BOUND = 10

def get_labels_analysis(y_true, y_pred, labels_inv_map):
    """Function that generates a trained RandomForestClassifier model

    Parameters
    ==========

    y_true : numpy.ndarray

        Ground truth vector of shape (n_samples,).
        Integer class representation.

    y_pred : numpy.ndarray

        Prediction vector of shape (n_samples,).
        Integer class representation.
    
    labels_inv_map : dict

        Dictionary that maps from integer class represention
        to plant species representation.
        

    Returns
    =======

    labels_true : list
        
        Ground truth list of length n_samples.
        Plant species representation.

    labels_predict : list
        
        Prediction list of length n_samples.
        Plant species representation.

    labels_order : list

        List of length n_classes, containing all plant species.
        It sets an order for later analysis. 

    """

    labels_order = [labels_inv_map[i] for i in range(len(labels_inv_map))]
    labels_true = [labels_inv_map[i] for i in y_true]
    labels_predict = [labels_inv_map[i] for i in y_pred]
    return labels_true, labels_predict, labels_order


def visualize_report(y_true, y_pred, labels_inv_map):
    """Function that generates a trained RandomForestClassifier model

    Parameters
    ==========

    y_true : numpy.ndarray

        Ground truth vector of shape (n_samples,).
        Integer class representation.

    y_pred : numpy.ndarray

        Prediction vector of shape (n_samples,).
        Integer class representation.
    
    labels_inv_map : dict

        Dictionary that maps from integer class represention
        to plant species representation.
        

    Returns
    =======

    report : str
        Classification report from scikit-learn.

    """

    labels_true, labels_predict, labels_order = get_labels_analysis(y_true, y_pred, labels_inv_map)
    return classification_report(labels_true,labels_predict, labels =labels_order)
    

def inspect_misclassified(y_true, y_pred, labels_inv_map,
                         iter_bound=ITER_BOUND, print_bound=PRINT_BOUND):
    """Function that generates a trained RandomForestClassifier model

    Parameters
    ==========

    y_true : numpy.ndarray

        Ground truth vector.
        Integer class representation.

    y_pred : numpy.ndarray

        Prediction vector.
        Integer class representation.
    
    labels_inv_map : dict

        Dictionary that maps from integer class represention
        to plant species representation.
        
    iter_bound : int

        Maximum bound on number of iterations while searching for
        important misclassifications.
        It has an impact if it is small with respect to n_classes.

    print_bound : int
        Maximum bound on number of misclassified_msg elements.


    Returns
    =======

    misclassified_msg : list
        
        List containing messages strings highlighting where
        most plant species misclassifications occur.
        
    """

    iter_bound = min(iter_bound, len(labels_inv_map)**2 - 1)
    
    labels_true, labels_predict, labels_order = get_labels_analysis(y_true, y_pred, labels_inv_map)
    
    cm = confusion_matrix(labels_true, labels_predict, labels=labels_order)
    idxs_cm = np.unravel_index(np.argsort(cm, axis=None)[::-1], cm.shape)

    print_count = 0
    i = 0
    misclassified_msg = []
    while (print_count < print_bound) and (i < iter_bound):
        i += 1
        if idxs_cm[0][i] != idxs_cm[1][i]:
            print_count += 1
            misclassified_msg.append(f"{labels_inv_map[idxs_cm[0][i]]} was predicted as {labels_inv_map[idxs_cm[1][i]]}: {cm[idxs_cm[0][i], idxs_cm[1][i]]} times")
    
    return misclassified_msg