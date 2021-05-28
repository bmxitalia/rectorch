.. _config-format:

Configuration files format
==========================

In **rectorch**, the configuration of the data set pre-processing, and the models' training/test 
is performed via `.json <https://www.json.org/json-en.html>`_ files.


Data configuration file
-----------------------

The data configuration file defines how the data set must be pre-processed.
The pre-processing comprehends the reading, clean up, and the partition of the data set.
The `.json <https://www.json.org/json-en.html>`_ data configuration file must have the following key-value pairs:

* ``processing``: dictionary with the pre-processing configurations;
* ``splitting``: dictionary with the splitting configurations.

The ``processing`` options are the following:

* ``data_path``: string representing the path to the data set `.csv <https://it.wikipedia.org/wiki/Comma-separated_values>`_ file;
* ``threshold``: float cut-off value for converting explicit feedback to implicit feedback;
* ``separator``: string delimiter used in the `.csv <https://it.wikipedia.org/wiki/Comma-separated_values>`_ file;
* ``header``: number of rows of the header (if no header set to ``None`` or ``null`` if in *json*);
* ``u_min``: integer minimum number of items for a user to be kept in the data set;
* ``i_min``: integer minimum number of users for an item to be kept in the data set;

The ``splitting`` options are the following:

* ``split_type``: string in the set {``vertical``, ``horizontal``} that indicates the type of splitting. Vertical splitting means that the heldout (validation and test) set considers users that are not in the training set. Horizontal splitting instead uses part of the ratings of all users as training and the rest as validation/test set/s;
* ``sort_by``: string that indicates the column to use for sorting the ratings. If the header is missing use the column index. If no sorting is required set to ``None`` or``null`` if in *json*;
* ``seed``: integer random seed used for both the training/validation/test division as well as for shuffling the data;
* ``shuffle``:  binary integer value which states whether to shuffle the ratings or not;
* ``valid_size``: if float is considered as the portion of ratings (if horizontal split) or users (if vertical split) to consider in the validation set. In the case of vertical splitting the value can be also an integer that indicates the number of users in the validation set;
* ``test_size``: if float is considered as the portion of ratings (if horizontal split) or users (if vertical split) to consider in the test set. In the case of vertical splitting the value can be also an integer that indicates the number of users in the test set;
* ``test_prop``: (used only in the case of vertical splitting) float in the range (0,1) which represents the proportion of items of the test users that are considered as test items (optional, default 0.2).
* ``cv``: whether to split the dataset in a cross-validationn fashion. If 'cv' is provided and it is > 1 then the splitting procedure returns a list of datasets. Note that in this case when the horizontal splitting is performed the default values of both 'shuffle' and 'sort_by' are used.

This is an example of a valid data configuration file:

.. code-block:: json

    {
        "processing": {
            "data_path": "./ml-100k/u.data",
            "threshold": 3.5,
            "separator": ",",
            "header": null,
            "u_min": 3,
            "i_min": 0
        },
        "splitting": {
            "split_type": "vertical",
            "sort_by": null,
            "seed": 98765,
            "shuffle": 1,
            "valid_size": 100,
            "test_size": 100,
            "test_prop": 0.2,
            "cv": 1
        }
    }



The example above is a valid configuration for the `Movielens 100k dataset <https://grouplens.org/datasets/movielens/100k/>`_
where ratings less than 3.5 stars are discarded as well as users with less than 3 ratings.
The splitting type is vertical and the heldout set has size 100 users (100 for validation set and 100 for the test set).
Ratings are not sorted but shuffled. The portion of ratings kept in the testing part of each users is 20%. Top-N is the task so the
remaining positive ratings are set equal to 1. Some examples of data configuration files are
available in `GitHub <https://github.com/makgyver/rectorch/tree/master/config>`_.


Neural Collaborative Reasoning data configuration file
------------------------

The data configuration file for Neural Collaborative Reasoning (`NCR <https://grouplens.org/datasets/movielens/100k/>`_) is different from the standard configuration file
since NCR uses a proprietary pre-processing.
In particular, the pre-processing includes the following steps:

1. the interactions are divided into positive and negative interactions based on a threshold. The interactions equal to
or higher than the threshold are mapped to 1, while the remaining interactions are mapped to 0;
2. all the interactions are ordered by timestamp;
3. the dataset is splitted into train, validation and test set using the leave-one procedure reported in the NCR paper;
4. the logical expressions are generated for train, validation, and test sets using the procedure explained in the NCR
paper.

The `.json <https://www.json.org/json-en.html>`_ data configuration file for NCR must have the following key-value pairs:

* ``processing``: dictionary with the pre-processing configurations;
* ``splitting``: dictionary with the splitting configurations.

The ``processing`` options are the following:

* ``data_path``: string representing the path to the data set `.csv <https://it.wikipedia.org/wiki/Comma-separated_values>`_ file;
* ``rating_threshold``: float value for converting explicit feedback to implicit feedback. All the ratings equal to or higher
than ``rating_threshold`` are mapped to 1, while the remaining ratings are mapped to 0;
* ``separator``: string delimiter used in the `.csv <https://it.wikipedia.org/wiki/Comma-separated_values>`_ file;
* ``header``: number of rows of the header (if no header set to ``None`` or ``null`` if in *json*);
* ``rating_order``: flag indicating whether the dataset has to be ordered by timestamp or not before splitting it using
the procedure explained in the NCR paper;
* ``max_history_length``: integer maximum number of items in the premise of the logical expressions;
* ``premise_threshold``: integer threshold used to cut-off logical expressions from the dataset based on the number of
premises. All the logical expressions with a number of premises equal to or lower than premise_threshold are removed
from the dataset.

The ``splitting`` options are the following:

* ``leave_n``: number of positive interactions that have to be held-out for validation and test sets. For example, if
``leave_n`` is set to 2, then for each user two positive interactions are put in validation set and 2 positive interactions
are put in test set;
* ``keep_n``: minimum number of positive interactions that must be put in the training set for each user.

This is an example of a valid data configuration file for NCR:

.. code-block:: json

    {
        "processing": {
        "data_path": "./ml-100k/movielens_100k.csv",
        "separator": ",",
        "header": 0,
        "rating_order": 1,
        "rating_threshold": 4,
        "max_history_length": 5,
        "premise_threshold": 0
    },
        "splitting": {
            "leave_n": 1,
            "keep_n": 5
        }
    }



The example above is a valid configuration for the `Movielens 100k dataset <https://grouplens.org/datasets/movielens/100k/>`_
where ratings equal to or higher than 4 stars are considered as positive, while ratings lower than 4 are considered as negative.
The ratings are ordered by timestamp before splitting the dataset into train, validation, and test set. The dataset is then
splitted into train, validation, and test sets using the parameters ``leave_n`` and ``keep_n``. Finally, the logical expressions
for the three folds are generated according to parameters ``max_history_length`` and ``premise_threshold``.
Some examples of data configuration files are
available in `GitHub <https://github.com/makgyver/rectorch/tree/master/config>`_.

Model configuration file
------------------------

The model configuration file defines the model and training hyper-parameters.
The `.json <https://www.json.org/json-en.html>`_ model configuration file must have the following key-value pairs:

* ``model``: dictionary with the values of model hyper-parameters. The name of the hyper-parameter must match the signature of the model as in :mod:`models`;
* ``train``: dictionary with the training parameters such as, number of epochs or the validation metrics. The name of the parameters must match the signature of the model's training method (see :mod:`models`);
* ``test``: dictionary with the test parameters. Up to now "metrics" is the only parameters to which a list of strings has to be associated. Metric names must follow the convention as defined in :mod:`metrics`;
* ``sampler``: dictionary with sampler parameters. The name of the parameters must match the signature of the sampler (see :mod:`sampler`);

.. code-block:: json

    {
        "model": {
            "beta" : 0.2,
            "anneal_steps" : 100000,
            "learning_rate": 0.001
        },
        "train": {
            "num_epochs": 200,
            "verbose": 1,
            "best_path": "chkpt_best.pth",
            "valid_metric": "ndcg@100"
        },
        "test":{
            "metrics": ["ndcg@100", "ndcg@10", "recall@20", "recall@50"]
        },
        "sampler": {
            "batch_size": 250
        }
    }

Some examples of model configuration files are available in
`GitHub <https://github.com/makgyver/rectorch/tree/master/config>`_.