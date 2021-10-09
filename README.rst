.. image:: https://coveralls.io/repos/github/niedeado/M05_ProjectReproducibility/badge.svg?branch=main
   :target: https://coveralls.io/github/niedeado/M05_ProjectReproducibility?branch=main
.. image:: https://github.com/niedeado/M05_ProjectReproducibility/actions/workflows/ci_testing.yml/badge.svg?
   :target: https://github.com/niedeado/M05_ProjectReproducibility/actions/workflows/ci_testing.yml?branch=main
.. image:: https://img.shields.io/badge/docs-latest-orange.svg
   :target: https://niedeado.github.io/M05_ProjectReproducibility/
.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT

============================================================
 Plant Species Classification based on Leaves Shape Features
============================================================

---------------------------------------------------------
Summary
---------------------------------------------------------

The goal of this project is to apply reproducible research principles
to the problem of plant species classification from leaves shape features.

Our hypothesis is that is possible to train a classifier that is able to
achieve a better than random accuracy.

-----------
Data Source
-----------

The dataset used for the current project can be retrieved at the following link:
https://archive.ics.uci.edu/ml/datasets/One-hundred+plant+species+leaves+data+set


---------------------------
Installation & Instructions
---------------------------


The installation and usage instructions can be accessed `here <https://niedeado.github.io/M05_ProjectReproducibility/>`__.

The Project and the corresponding model can be installed by pip as mentioned below.
Please run after the installation the provided test, to ensure everything is set up
correclty.


.. code-block:: rst
   
   $ pip install git+https://github.com/niedeado/M05_ProjectReproducibility.git@main
   
   $ leaf_cc-run_test
   
   $ leaf_cc-run_model


----------
References
----------

Original paper::

   @inproceedings{Mallah2013PLANTLC,
      title={PLANT LEAF CLASSIFICATION USING PROBABILISTIC INTEGRATION OF SHAPE, TEXTURE AND MARGIN FEATURES},
      author={Charles D. Mallah and James S. Cope and James Orwell},
      year={2013}
   }


UCI Machine Learning Repository::

   @misc{Dua:2019 ,
      author = "Dua, Dheeru and Graff, Casey",
      year = "2017",
      title = "{UCI} Machine Learning Repository",
      url = "http://archive.ics.uci.edu/ml",
      institution = "University of California, Irvine, School of Information and Computer Sciences"
   }
