Command line interface
^^^^^^^^^^^^^^^^^^^^^^^^^

A command line interface (CLI) is provided for running multiple federated learning experiments.
The only argument is the path to the configuration file (in YAML format) for the experiments.
Examples of configuration files can be found in the `example-configs <https://github.com/wenh06/fl-sim/example-configs>`_ directory.
For example, in the `all-alg-fedprox-femnist.yml <https://github.com/wenh06/fl-sim/example-configs/all-alg-fedprox-femnist.yml>`_ file, we have

.. collapse:: [example-configs/all-alg-fedprox-femnist.yml]

    .. code-block:: yaml

        # Example config file for fl-sim command line interface

        strategy:
        matrix:
            algorithm:
            - Ditto
            - FedDR
            - FedAvg
            - FedAdam
            - FedProx
            - FedPD
            - FedSplit
            - IFCA
            - pFedMac
            - pFedMe
            - ProxSkip
            - SCAFFOLD
            clients_sample_ratio:
            - 0.1
            - 0.3
            - 0.7
            - 1.0

        algorithm:
        name: ${{ matrix.algorithm }}
        server:
            num_clients: null
            clients_sample_ratio: ${{ matrix.clients_sample_ratio }}
            num_iters: 100
            p: 0.3  # for FedPD, ProxSkip
            lr: 0.03  # for SCAFFOLD
            num_clusters: 10  # for IFCA
            log_dir: all-alg-fedprox-femnist
        client:
            lr: 0.03
            num_epochs: 10
            batch_size: null  # null for default batch size
            scheduler:
            name: step  # StepLR
            step_size: 1
            gamma: 0.99
        dataset:
        name: FedProxFEMNIST
        datadir: null  # default dir
        transform: none  # none for static transform (only normalization, no augmentation)
        model:
        name: cnn_femmist_tiny
        seed: 0

|

The ``strategy`` (optional) section specifies the grid search strategy;
the ``algorithm`` section specifies the hyperparameters of the federated learning algorithm:
``name`` is the name of the algorithm, ``server`` specifies the hyperparameters of the server,
and ``client`` specifies the hyperparameters of the client;
the ``dataset`` section specifies the dataset, and the ``model`` section specifies
the named model (ref. the ``candidate_models`` property of the dataset classes) to be used.
