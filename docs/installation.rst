.. _miniconda: https://docs.conda.io/en/latest/miniconda.html

.. _activities_installation:


==============
 Installation
==============

It is possible to install the project and its related package with pip or with conda in a
virtual environment. Both options are shown below. If it is intended to also use the
provided visualisation tool in a jupyter notebook, it is recommended to use the option by using
conda environment. So it is possible to access the notebook file outside of the package.

To ensure everything is set up correctly, please run after the installation in both cases
the provided test shown in the code blocks below.


Using Miniconda
---------------


The package depends on multiple packages to run properly. Please install the
version of these packages by creating the environment with the help of the packages listed
in the requirements_pkg.txt file. To use conda environments, anaconda or miniconda has to be installed.
To set up the package, we propose to use miniconda_.

.. code-block:: sh

    $ git clone https://github.com/niedeado/M05_ProjectReproducibility.git leaf_src
    $ cd leaf_src/
    $ conda create --name leaf --file requirements_pkg.txt
    $ conda activate leaf
    (leaf) $ pip install -e .
    (leaf) $ leaf_cc-run_test




Using pip
---------

It is also possible to install the package directly with pip, as shown below.

.. code-block:: sh

    $ pip install git+https://github.com/niedeado/M05_ProjectReproducibility.git@main
    $ leaf_cc-run_test
    $ leaf_cc-run_model # run model with default configuration