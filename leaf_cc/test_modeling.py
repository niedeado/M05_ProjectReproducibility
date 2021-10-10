import pytest
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from . import database
from . import analysis
from . import parameter_selection
from . import algorithm
from . import main_script


@pytest.fixture
def dataset_setup():
    data = [
        ["Pippo", 0.5, 0.2],
        ["Pippo", 0.6, 0.1],
        ["Pluto", 0.3, 0.7],
        ["Pluto", 0.8, 0.3],
    ]
    columns = ["species", "shape_0", "shape_1"]
    return pd.DataFrame(data=data, columns=columns)


def test_extract_data_array(dataset_setup):
    X, y, labels_inv_map, labels_map = database.extract_data_array(dataset_setup)
    assert np.equal(X, np.array([[0.5, 0.2], [0.6, 0.1], [0.3, 0.7], [0.8, 0.3]])).all()
    assert np.equal(y, np.array([0, 0, 1, 1])).all()
    assert labels_inv_map == {0: "Pippo", 1: "Pluto"}
    assert labels_map == {"Pippo": 0, "Pluto": 1}


def test_split_data(dataset_setup):
    X, y, _, _ = database.extract_data_array(dataset_setup)
    X_train, X_test, y_train, y_test = database.split_data(X, y, test_size=0.5)
    assert X_train.shape == (2, 2)
    assert X_test.shape == (2, 2)
    assert len(y_train) == 2
    assert len(y_test) == 2
    assert np.sum(y_train == 0) == 1
    assert np.sum(y_train == 1) == 1
    assert np.sum(y_test == 0) == 1
    assert np.sum(y_test == 1) == 1


def test_get_labels_analysis():
    y_true = np.array([1, 1, 0, 0, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 1, 0])
    labels_inv_map = {0: "Pippo", 1: "Pluto"}
    labels_true, labels_predict, labels_order = analysis.get_labels_analysis(
        y_true, y_pred, labels_inv_map
    )
    assert labels_true == [
        "Pluto",
        "Pluto",
        "Pippo",
        "Pippo",
        "Pluto",
        "Pippo",
        "Pippo",
    ]
    assert labels_predict == [
        "Pluto",
        "Pluto",
        "Pluto",
        "Pippo",
        "Pippo",
        "Pluto",
        "Pippo",
    ]
    assert labels_order == ["Pippo", "Pluto"]


def test_visualize_report():
    y_true = np.array([1, 1, 0, 0, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 1, 0])
    labels_inv_map = {0: "Pippo", 1: "Pluto"}
    report = analysis.visualize_report(y_true, y_pred, labels_inv_map)
    # filters out spaces and then empty strings in the report.
    report_split = [s for s in report.split(" ") if s != ""]
    accuracy_idx = report_split.index("accuracy") + 1
    assert report_split[accuracy_idx] == str(0.57)


def test_inspect_misclassified():
    y_true = np.array([1, 1, 0, 0, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 1, 0])
    labels_inv_map = {0: "Pippo", 1: "Pluto"}
    misclassified_msg = analysis.inspect_misclassified(y_true, y_pred, labels_inv_map)
    assert misclassified_msg == [
        "Pippo was predicted as Pluto: 2 times",
        "Pluto was predicted as Pippo: 1 times",
    ]


def test_train(dataset_setup):
    X, y, _, _ = database.extract_data_array(dataset_setup)
    hyperparameters = {"n_estimators": 3, "max_depth": None, "max_features": "auto"}
    model = algorithm.train(X, y, hyperparameters)
    assert isinstance(model, RandomForestClassifier)


def test_selection_criteria():
    parameters = [
        {"n_estimators": 3, "max_depth": None, "max_features": "auto"},
        {"n_estimators": 50, "max_depth": 10, "max_features": "sqrt"},
        {"n_estimators": 200, "max_depth": 20, "max_features": "log2"},
    ]
    validation_accs = [0.2, 0.7, 0.9]
    best_params = parameter_selection.selection_criteria(parameters, validation_accs)
    assert best_params == {"n_estimators": 200, "max_depth": 20, "max_features": "log2"}


def test_main_script():
    assert main_script.main()
