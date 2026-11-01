Command line interface
^^^^^^^^^^^^^^^^^^^^^^^^^

A command line interface (CLI) is provided for running multiple federated learning experiments.
The only argument is the path to the configuration file (in YAML format) for the experiments.
Examples of configuration files can be found in the `example-configs <https://github.com/wenh06/fl-sim/tree/master/example-configs>`_ directory.
For example, in the `all-alg-fedprox-femnist.yml <https://github.com/wenh06/fl-sim/tree/master/example-configs/all-alg-fedprox-femnist.yml>`_ file, we have

.. dropdown:: [example-configs/all-alg-fedprox-femnist.yml]
    :animate: fade-in-slide-down

    .. literalinclude:: ../../example-configs/all-alg-fedprox-femnist.yml
        :language: yaml

|

The ``strategy`` (optional) section specifies the grid search strategy;
the ``algorithm`` section specifies the hyperparameters of the federated learning algorithm:
``name`` is the name of the algorithm, ``server`` specifies the hyperparameters of the server,
and ``client`` specifies the hyperparameters of the client;
the ``dataset`` section specifies the dataset, and the ``model`` section specifies
the named model (ref. the ``candidate_models`` property of the dataset classes) to be used.
