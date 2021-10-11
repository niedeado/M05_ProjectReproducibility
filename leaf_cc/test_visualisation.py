from unittest.mock import patch
import pandas as pd
import numpy as np
import ipywidgets as widgets
from . import data_visualisation
from . import parameter_selection
import string


def test_load_widgets():
    x_widget, mean_widget, std_widget, button_widget = data_visualisation.load_widgets()
    assert isinstance(x_widget, widgets.widget_selection.SelectMultiple)
    assert x_widget.value[0] == "Magnolia Heptapeta"
    assert isinstance(mean_widget, widgets.widget_bool.Checkbox)
    assert isinstance(mean_widget.value, bool)
    assert isinstance(std_widget, widgets.widget_bool.Checkbox)
    assert isinstance(std_widget.value, bool)
    assert isinstance(button_widget, widgets.widget_button.Button)


def test_run_pca():
    x_widget = widgets.SelectMultiple(
        options=["Magnolia Heptapeta", "Ilex Cornuta"],
        value=["Magnolia Heptapeta"],
        description="Species\n",
        disabled=False,
        rows=8,
    )
    true_widget = widgets.Checkbox(value=True, description="Test")
    false_widget = widgets.Checkbox(value=False, description="Test")

    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))
    cols = ["PC1", "PC2", "PC3", "PC4"]
    df.columns = cols + df.columns.tolist()[len(cols) :]
    X_train = df.values

    y_train = np.random.choice([0, 1], size=len(df))
    df["Species"] = pd.Series(
        np.random.choice(
            ["Magnolia Heptapeta", "Ilex Cornuta"], size=len(df), p=[0.5, 0.5]
        ),
        index=df.index,
    )
    labels_map = {"Magnolia Heptapeta": 0, "Ilex Cornuta": 1}
    labels_inv_map = {"0": "Magnolia Heptapeta", "1": "Ilex Cornuta"}
    df_out = data_visualisation.run_pca(
        X_train,
        y_train,
        false_widget,
        false_widget,
        x_widget,
        labels_map,
        labels_inv_map,
    )
    df_out_mean = data_visualisation.run_pca(
        X_train,
        y_train,
        true_widget,
        false_widget,
        x_widget,
        labels_map,
        labels_inv_map,
    )
    df_out_std = data_visualisation.run_pca(
        X_train,
        y_train,
        false_widget,
        true_widget,
        x_widget,
        labels_map,
        labels_inv_map,
    )
    df_out_all = data_visualisation.run_pca(
        X_train, y_train, true_widget, true_widget, x_widget, labels_map, labels_inv_map
    )
    assert isinstance(df_out, pd.DataFrame)
    assert set(df.columns).intersection(set(df_out.columns))
    assert not np.all(df_out.values == df_out_mean.values)
    assert not np.all(df_out.values == df_out_std.values)
    assert not np.all(df_out.values == df_out_all.values)


@patch("%s.data_visualisation.Idp" % __name__)
def test_run_all_below(mock_disp):
    button_widget = widgets.Button(
        description="Update", disabled=False, button_style="primary", tooltip="Update"
    )
    button_widget.on_click(data_visualisation.run_all_below)
    assert mock_disp.display.assert_called
    assert True


@patch("%s.data_visualisation.plt" % __name__)
def test_plot_pca_variance(mock_plt):
    ncol = 25
    alphabet_string = list(string.ascii_lowercase)
    df = pd.DataFrame(
        np.random.randint(0, 100, size=(100, ncol)), columns=alphabet_string[:ncol]
    )
    _ = data_visualisation.plot_pca_variance(df)
    assert mock_plt.figure.called
    assert mock_plt.show.called


@patch("%s.data_visualisation.plt" % __name__)
def test_plot_pca(mock_plt):
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))
    cols = ["PC1", "PC2", "PC3", "PC4"]
    df.columns = cols + df.columns.tolist()[len(cols) :]
    df["Species"] = pd.Series(
        np.random.choice(
            ["Magnolia Heptapeta", "Ilex Cornuta"], size=len(df), p=[0.5, 0.5]
        ),
        index=df.index,
    )
    _ = data_visualisation.plot_pca(df)
    assert mock_plt.ticklabel_format.assert_called_once
    assert mock_plt.show.called


@patch("%s.parameter_selection.pickle" % __name__)
def test_hyperparam_tuning(mock_pickle):

    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))
    cols = ["PC1", "PC2", "PC3", "PC4"]
    df.columns = cols + df.columns.tolist()[len(cols) :]
    X_train = df.values

    y_train = np.random.choice([0, 1], size=len(df))

    parameter_selection.hyperparam_tuning(
            X_train, y_train, True
        )
    assert mock_pickle.dump.called