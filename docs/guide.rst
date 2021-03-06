.. _activities_userguide:

============
 User Guide
============

This section shows how to reproduce the classification results.

From the main directory launch the following command:

.. code-block:: sh

   $ leaf_cc-run_model


The classification report, misclassified inspection and score are reported below.

To get to know more about how to run the model with different options, the argument ``-h`` can be passed to the
main statement to get the help messages.

.. code-block:: sh

   $ leaf_cc-run_model -h


Test classification Report
--------------------------

(shortened output)

+----------------+-------------+----------+------------+-----------+
|                | *precision* | *recall* | *f1-score* | *support* |
+================+=============+==========+============+===========+
| *accuracy*     |             |          | 0.65       | 320       |
+----------------+-------------+----------+------------+-----------+
| *macro avg*    | 0.67        | 0.65     | 0.62       | 320       |
+----------------+-------------+----------+------------+-----------+
| *weighted avg* | 0.68        | 0.65     | 0.63       | 320       |
+----------------+-------------+----------+------------+-----------+


Misclassification inspection
----------------------------

* Quercus Castaneifolia was predicted as Quercus Imbricaria: 3 times
* Olea Europaea was predicted as Quercus Crassipes: 2 times
* Quercus Imbricaria was predicted as Quercus Castaneifolia: 2 times
* Quercus Infectoria sub was predicted as Salix Intergra: 2 times
* Quercus Canariensis was predicted as Lithocarpus Cleistocarpus: 2 times
* Populus Nigra was predicted as Populus Grandidentata: 2 times
* Viburnum Tinus was predicted as Quercus Phillyraeoides: 2 times
* Morus Nigra was predicted as Liriodendron Tulipifera: 2 times
* Quercus Rhysophylla was predicted as Quercus Imbricaria: 2 times
* Acer Campestre was predicted as Acer Rubrum: 2 times


Score
-----

* Training accuracy: 1.0
* Test accuracy: 0.653125


Visualisation of Species
------------------------

.. toctree::
    :glob:

    set_up_visual

    VisualWidget/*