.. _activities_set_up_visual:

Start Jupyter Notebook with Visualisation File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you followed the :ref:`installation guide <activities_installation>`
to set set up the package and it is installed successfully, it is now also possible
to start the jupyter notebook to visualize the data after applying a PCA. How the PCA is
applied, please take a look at the documentation of :meth:`leaf_cc.data_visualisation.run_pca`.

If the package is cloned from git and set up with conda, it is possible to run the
notebook from command line as follow. Please be sure that the current working directory
is in the folder leaf_cc, which is ensured if you have followed the installation
instructions.

.. code:: sh

    $ jupyter notebook leaf_cc/notebooks/VisualWidget.ipynb


The second possibility is to run the model and activate the option to start the jupyter
notebook file by adding ``-n=True`` to the running statement.
Please consider if you choose this option, the jupyter notebook file is called from the
package itself, so any modifications are applied into the file in the package.


.. code:: sh

    $ $ leaf_cc-run_model -n=True


Once the statement is executed and the notebook appeared, all the cells should be restarted.
How to do that, please refer to the
`Jupyter Notebook user guide <https://jupyter.readthedocs.io/en/latest/install.html>`__.

The corresponding file should look like in this appended :ref:`example <activities_visualwidget>`.

To chose multiple species in the dropdown menu, ``CTRL`` has to be pressed while clicking
on additional species for selecting.
After that, by pressing the ``Update`` button, all the cells below are refreshed.



