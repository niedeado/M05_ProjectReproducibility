============================================================
 Plant Species Classification based on Leaves Shape Features
============================================================


Introduction
------------

The goal of this project is to apply reproducible research principles
to the problem of plant species classification from leaves shape features.

The specification of the problem at hand can be summarized by the following
basic statistics on the shape features dataset:

   * Number of samples: 1600
   * Number of classes: 100
   * Number of features: 64

Moreover, the samples are evenly distributed among classes, 
meaning there are 16 samples per class.

Our scientific hypothesis is that based on the shape features
we can train a classifier that achieves a better than random accuracy.


Method
------

The task at hand is challenging, due to the unfavorable dataset statistics, 
i.e. small ratio number of samples over number of classes.

The adopted workflow is summarized by the image below:

.. image:: /pictures/workflow.png

To approach the problem, we decided to adopt a random forest classifier, 
a machine learning algorithm that thanks to his characteristics, most notably 
bagging, may be able to handle such a case.

Train-test and cross-validation splits are conducted 
in such a way to preserve the balance among classes.

Hyperparameter tuning is conducted via cross-validation
and the following parameters are kept as default 
(unspecified parameters are the default ones from scikit-learn):

.. code-block:: python

   n_estimators = 200
   max_depth = None
   max_features = 'log2'


Results
-------

Following the described workflow the test set accuracy
was found to be **0.65**.

For a more detailed description on how to obtain this result,
please refer to the :ref:`user guide <activities_userguide>`.


Conclusion
----------

Based on the obtained results we cannot reject our initial hypothesis.


Data Source
-----------

The dataset used for the current project can be retrieved at the following link:
https://archive.ics.uci.edu/ml/datasets/One-hundred+plant+species+leaves+data+set



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


Documentation
-------------

.. toctree::

   installation
   guide
   api
   license


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

