import sys
sys.path.append('./M05_ProjectReproducibility/pyfiles')
from database import *
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Javascript
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, Layout

#print(os.getcwd())
#print(sys.argv[0])
#print(os.path.dirname(os.path.realpath('__file__')))

dataset = load()

X, y, labels_inv_map, labels_map = extract_data_array(dataset)

X_train, X_test, y_train, y_test = split_data(X,y)


def run_pca(X_train, y_train, mean_widget, std_widget, x_widget, labels_map=labels_map, labels_inv_map=labels_inv_map):
    ss = StandardScaler(with_mean=mean_widget, with_std=std_widget)
    train_data = ss.fit_transform(X_train)

    pca = decomposition.PCA(n_components=4)
    _ = pca.fit_transform(train_data)
    chosen_labels = np.array([labels_map.get(name) for name in x_widget.value])
    ix_true = np.argwhere(np.in1d(y_train, chosen_labels)).flatten()

    pc = pca.transform(X_train[ix_true, ...])

    pc_df = pd.DataFrame(data=pc,
                         columns=['PC1', 'PC2', 'PC3', 'PC4'])
    pc_df['Species'] = np.array([labels_inv_map.get(label_nr) for label_nr in y_train[ix_true]])

    return pc_df

def plot_pca(data):
    sns.lmplot(x="PC1", y="PC2",
               data=data,
               fit_reg=False,
               hue='Species',  # color by cluster
               legend=True,
               scatter_kws={"s": 32},
               height=6.3, aspect=11.2 / 6.3)
    plt.ticklabel_format(style='sci', axis='y')

    plt.xlabel("PC1", size=14)
    plt.ylabel("PC2", size=14)
    plt.title("\nPCA Plant Species\n", size=20)
    plt.show()


def run_all_below(ev):
    display(Javascript(
        'IPython.notebook.execute_cell_range(IPython.notebook.get_selected_index()+1, IPython.notebook.ncells())'))
    return None

def load_widgets():
    """
    Load and display the Jupyter widgets
    """

    species = list(sorted(set(dataset.species)))

    # Setting options
    x_widget = widgets.SelectMultiple(options=species, value=["Magnolia Heptapeta"],
                                      description="Species\n", disabled=False,
                                      rows=8
    )

                                      #layout = Layout( display="flex",flex_flow='column'))

    mean_widget = widgets.Checkbox(value=True, description='Scale by Mean')
    std_widget = widgets.Checkbox(value=True, description='Scale by Std')

    button_widget = widgets.Button(description='Update',
                                   disabled=False,
                                   button_style='primary',
                                   tooltip='Update')

    """button_widget = widgets.Button(description='Update',
                                   disabled=False)
"""
    box_layout = widgets.Layout(display='flex',
                                flex_flow='column',
                                align_items='center',
                                width='34%')
    box = widgets.HBox(children=[button_widget], layout=box_layout)





    button_widget.on_click(run_all_below)

    # Display
    display(x_widget)

    display(mean_widget)
    display(std_widget)
    """display(widgets.VBox([x_widget,mean_widget, std_widget]))"""
    display(box)

    return x_widget, mean_widget, std_widget, button_widget

if __name__ == "__main__":
    x_widget, mean_widget, std_widget, button_widget= load_widgets()