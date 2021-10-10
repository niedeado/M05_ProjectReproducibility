import numpy as np
import pandas as pd
from . import database
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as Idp
from IPython.display import Javascript
import ipywidgets as widgets

# initialize some variables for using them as default input values for
# some functions defined below
dataset = database.load()
X, y, labels_inv_map, labels_map = database.extract_data_array(dataset)
X_train, X_test, y_train, y_test = database.split_data(X, y)


def run_pca(
    X_train,
    y_train,
    mean_widget,
    std_widget,
    x_widget,
    labels_map=labels_map,
    labels_inv_map=labels_inv_map,
):
    """Runs PCA on the passed data based on the defined parameters and returns a
    pandas Dataframe. Consider the PCA is always fitted on the whole dataset X_train
    and the returned Dataframe isdependable on the values from the x_widget object.

    Parameters
    ==========

    X_train : numpy.ndarray
        Data matrix to run PCA on it

    y_train : numpy.ndarray
         Ground truth vector with integer class labels

    mean_widget : ipywidgets.widgets.widget_bool.Checkbox
        Widgets that indicates to center the data before scaling

    std_widget : ipywidgets.widgets.widget_bool.Checkbox
        Widget that indicates to scale the data to unit variance

    x_widget : ipywidgets.widgets.widget_selection.SelectMultiple
        Widget that defines, which data observation is returned,
        based on the containing labels in the widget object

    labels_map : dict
        Dictionary that maps from plant species representation
        to integer class represention.

    labels_inv_map : dict
        Dictionary that maps from integer class represention
        to plant species representation.


    Returns
    =======

    pc_df : pandas.DataFrame
        Data matrix with 4 PCA-Components and the regarding
        label entry as 'Species' in plant species representation .


    """

    ss = StandardScaler(with_mean=mean_widget.value, with_std=std_widget.value)
    train_data = ss.fit_transform(X_train)

    pca = decomposition.PCA(n_components=4)
    _ = pca.fit_transform(train_data)

    chosen_labels = np.array([labels_map.get(name) for name in x_widget.value])
    ix_true = np.argwhere(np.in1d(y_train, chosen_labels)).flatten()
    pc = pca.transform(X_train[ix_true, ...])

    pc_df = pd.DataFrame(data=pc, columns=["PC1", "PC2", "PC3", "PC4"])
    pc_df["Species"] = np.array(
        [labels_inv_map.get(label_nr) for label_nr in y_train[ix_true]]
    )

    return pc_df


def plot_pca(data):
    """Creates the plot figure of the passed data of the observations in 2d and
    returns the figure object.
    Consider that the method matplotlib.pyplot.show is called inside this function
    and the plot is showed if the regarding executing environment supports to show
    figures.


    Parameters
    ==========

    data : pandas.DataFrame
        Pandas dataframe with at least two PCA-Components
        named 'PC1' and 'PC2' and the Column 'Species'
        to indicate the label of the observations in the
        plot.


    Returns
    =======

    fig : matplotlib.figure.Figure
        Figure instance that contains all the plot properties

    """

    fig = sns.lmplot(
        x="PC1",
        y="PC2",
        data=data,
        fit_reg=False,
        hue="Species",  # color by cluster
        legend=True,
        scatter_kws={"s": 32},
        height=6.3,
        aspect=11.2 / 6.3,
    )
    plt.ticklabel_format(style="sci", axis="y")

    plt.xlabel("PC1", size=14)
    plt.ylabel("PC2", size=14)
    plt.title("\nPCA Plant Species\n", size=20)
    plt.show()
    return fig


def plot_pca_variance(data):
    """Runs PCA on passed data and creates the plot figure to visualize the
    variance corresponding to each component and returns finally the figure object.
    Consider that the method matplotlib.pyplot.show is called inside this function
    and the plot is showed if the regarding executing environment supports to show
    figures.


    Parameters
    ==========

    data : numpy.ndarray
        Data matrix to run PCA on it.
        The shape on axis =1 has to be
        at least 19.


    Returns
    =======

    fig : matplotlib.figure.Figure
        Figure instance that contains all the plot properties

    """

    n_comp = 20
    pca = decomposition.PCA(n_components=n_comp - 1)
    _ = pca.fit_transform(data)
    var_exp = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_exp)
    scale = 0.8
    fig = plt.figure(figsize=(11.2 * scale, 6.3 * scale))
    plt.bar(
        range(1, n_comp),
        var_exp,
        alpha=0.75,
        align="center",
        label="Individual explanatory variance",
    )

    plt.step(
        range(1, n_comp),
        cum_var_exp,
        where="mid",
        label="Cumulative explanatory variance",
        color=(0.867, 0.52, 0.32),
    )

    plt.ylim(0, 1.1)
    plt.xlabel("Principal components")
    plt.ylabel("Amount of explanatory variance")
    plt.title(
        "\nExplanatory variance of the principal components " "from the features\n",
        size=14,
    )
    plt.legend(loc="upper right", bbox_to_anchor=(1, 0.85))
    plt.xticks(list(range(1, n_comp, 2)))
    plt.yticks([i * 0.1 for i in range(0, int((n_comp / 2) + 1))])
    plt.show()
    return fig


def run_all_below(ev):
    """Run all cells below the current cell form a Jupyter notebook,
    without executing the cell that has this button from where this
    function is called.


    Parameters
    ==========

    ev :
        unused pseudo variable which is needed to pass this function
        later


    Returns
    =======

    None

    """

    Idp.display(
        Javascript(
            "IPython.notebook.execute_cell_range(IPython.notebook.get_selected_index()+1, "
            "IPython.notebook.ncells())"
        )
    )
    return None


def load_widgets():
    """Load and display the Jupyter widgets needed in the notebook

    Returns
    =======

    x_widget : ipywidgets.widgets.widget_selection.SelectMultiple
        SelectMultiple widget, that contains in the attribute 'value'
        a list with all the selected species.

    mean_widget : ipywidgets.widgets.widget_bool.Checkbox
        Checkbox widget, that contains in the attribute 'value'
        a bool value.

    std_widget : ipywidgets.widgets.widget_bool.Checkbox
        Checkbox widget, that contains in the attribute 'value'
        a bool value.

    button_widget : ipywidgets.widgets.widget_button.Button
        Button which is connected to the function 'run_all_below'
        By clicking on that button, all cells below it in the
        Jupyter notebook file will be executed.

    """

    species = list(sorted(set(dataset.species)))

    x_widget = widgets.SelectMultiple(
        options=species,
        value=["Magnolia Heptapeta"],
        description="Species\n",
        disabled=False,
        rows=8,
    )

    mean_widget = widgets.Checkbox(value=True, description="Scale by Mean")

    std_widget = widgets.Checkbox(value=True, description="Scale by Std")

    button_widget = widgets.Button(
        description="Update", disabled=False, button_style="primary", tooltip="Update"
    )

    box_layout = widgets.Layout(
        display="flex", flex_flow="column", align_items="center", width="34%"
    )

    box = widgets.HBox(children=[button_widget], layout=box_layout)

    button_widget.on_click(run_all_below)

    # Display all widgets
    Idp.display(x_widget)
    Idp.display(mean_widget)
    Idp.display(std_widget)
    Idp.display(box)

    return x_widget, mean_widget, std_widget, button_widget


if __name__ == "__main__":
    x_widget, mean_widget, std_widget, button_widget = load_widgets()
