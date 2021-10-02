import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
import ipywidgets as widgets
import data_visualisation

@pytest.fixture # not used yet
def get_testdata():
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    cols = ['PC1', 'PC2', 'PC3', 'PC4']
    df.columns = cols + df.columns.tolist()[len(cols):]
    return df

def test_load_widgets():
    x_widget, mean_widget, std_widget, button_widget = data_visualisation.load_widgets()
    assert isinstance(x_widget, widgets.widget_selection.SelectMultiple)
    assert x_widget.value[0] == "Magnolia Heptapeta"
    assert isinstance(mean_widget, widgets.widget_bool.Checkbox)
    assert isinstance(mean_widget.value,bool)
    assert isinstance(std_widget, widgets.widget_bool.Checkbox)
    assert isinstance(std_widget.value, bool)
    assert isinstance(button_widget, widgets.widget_button.Button)


#@pytest.mark.skip(reason="not implemented yet")
def test_run_pca():
    x_widget = widgets.SelectMultiple(options=["Magnolia Heptapeta", 'Ilex Cornuta'], value=["Magnolia Heptapeta"],
                                      description="Species\n", disabled=False,
                                      rows=8 )
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    cols = ['PC1', 'PC2', 'PC3', 'PC4']
    df.columns = cols + df.columns.tolist()[len(cols):]
    X_train = df.values

    y_train = np.random.choice([0,1], size = len(df))
    df['Species'] = pd.Series(np.random.choice(["Magnolia Heptapeta", 'Ilex Cornuta'], size=len(df), p=[0.5, 0.5]),
                              index=df.index)
    labels_map = {'Magnolia Heptapeta': 0, 'Ilex Cornuta':1}
    labels_inv_map = {"0" : 'Magnolia Heptapeta',  "1": 'Ilex Cornuta'}
    df_out = data_visualisation.run_pca(X_train, y_train, True, True, x_widget, labels_map, labels_inv_map)
    assert isinstance(df_out, pd.DataFrame)
    assert set(df.columns).intersection(set(df_out.columns))


def test_run_all_below():
    button_widget = widgets.Button(description='Update',
                                   disabled=False,
                                   button_style='primary',
                                   tooltip='Update')
    button_widget.on_click(data_visualisation.run_all_below)
    assert True


@patch("matplotlib.pyplot.show")
def test_plot_pca(mock_show):
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    cols = ['PC1', 'PC2', 'PC3', 'PC4']
    df.columns = cols + df.columns.tolist()[len(cols):]
    df['Species'] = pd.Series(np.random.choice(["Magnolia Heptapeta", 'Ilex Cornuta'], size=len(df), p =[0.5, 0.5]), index=df.index)
    data_visualisation.plot_pca(df)


