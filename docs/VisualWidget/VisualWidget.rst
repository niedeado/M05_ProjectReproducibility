.. _activities_visualwidget:

Visualisation of Plant Species after applying PCA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import sys
    from leaf_cc import database as db
    from leaf_cc import data_visualisation as dv
    import matplotlib.pyplot as plt

Load the data, apply split of the data in test and training set and show
the explonatory variance of the first 19 components after running PCA.

.. code:: ipython3

    dataset = db.load()
    X, y, labels_inv_map, labels_map = db.extract_data_array(dataset)
    
    X_train, X_test, y_train, y_test = db.split_data(X,y)
    fig_var = dv.plot_pca_variance(X_train)
    plt.show()



.. image:: output_3_0.png


.. code:: ipython3

    x_widget, mean_widget, std_widget, button_widget= dv.load_widgets()



.. image:: output_3_1.png

Show the chosen plant species in the 2D PCA domain.

.. code:: ipython3

    data = dv.run_pca(X_train, y_train, mean_widget, std_widget, x_widget)
    rfig_pca = dv.plot_pca(data)
    plt.show()



.. image:: output_6_0.png


