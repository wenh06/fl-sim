.. _config_reference:

Algorithm Configuration Reference
=================================

.. note::

   This page is **auto-generated** at documentation build time from the
   (pydantic) JSON schemas of the built-in algorithms' configs -- do not
   edit it by hand; the generator is ``docs/gen_config_reference.py``.

All server and client configs accept additional keyword arguments beyond
the listed fields (``extra="allow"``); extras are stored as attributes and
are available to the algorithm implementation. Positional construction maps
arguments onto the fields in their declaration order, i.e. the order of the
tables below; a ``*`` in the Default column marks a required field.


.. _config-apfl:

APFL
----

**Server config:** ``APFLServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``tau``
     - int
     - 1

.. dropdown:: ``APFLServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the APFL algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float\n    The ratio of clients to sample for each iteration.\ntau : int, default 1,\n    synchronization gap.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "tau": {
          "default": 1,
          "title": "Tau",
          "type": "integer"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio"
      ],
      "title": "APFLServerConfig",
      "type": "object"
    }

**Client config:** ``APFLClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "SGD"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""
   * - ``mixture_weight``
     - float
     - 0.5

.. dropdown:: ``APFLClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the APFL algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\noptimizer : str, default \"SGD\"\n    The name of the optimizer to solve the local (inner) problem.\nlr : float, default 1e-2\n    The learning rate.\nmixture_weight : float, default 0.5\n    The weight of the local model parameters and the global model parameters.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "SGD",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "mixture_weight": {
          "default": 0.5,
          "title": "Mixture Weight",
          "type": "number"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "APFLClientConfig",
      "type": "object"
    }


.. _config-ditto:

Ditto
-----

**Server config:** ``DittoServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""

.. dropdown:: ``DittoServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the Ditto algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float\n    The ratio of clients to sample for each iteration.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio"
      ],
      "title": "DittoServerConfig",
      "type": "object"
    }

**Client config:** ``DittoClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "SGD"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""
   * - ``optimizer_per``
     - str
     - "ProxSGD"
   * - ``prox``
     - float
     - 0.01
   * - ``lr_per``
     - Optional[float]
     - None

.. dropdown:: ``DittoClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the Ditto algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\noptimizer : str, default \"SGD\"\n    The name of the optimizer to solve the local (inner) problem.\noptimizer_per : str, default \"ProxSGD\"\n    The name of the optimizer to solve the personalization problem.\nprox : float, default 0.01\n    Coefficient of the proximal term.\nlr : float, default 1e-2\n    The learning rate.\nlr_per : float, optional\n    The learning rate for personalization.\n    If not specified, ``lr_per`` will be set to ``lr``.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "SGD",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "optimizer_per": {
          "default": "ProxSGD",
          "title": "Optimizer Per",
          "type": "string"
        },
        "prox": {
          "default": 0.01,
          "title": "Prox",
          "type": "number"
        },
        "lr_per": {
          "anyOf": [
            {
              "type": "number"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Lr Per"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "DittoClientConfig",
      "type": "object"
    }


.. _config-fpfc:

FPFC
----

**Server config:** ``FPFCServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``optimizer``
     - str
     - "Avg"
   * - ``lr``
     - float
     - 1
   * - ``betas``
     - list
     - [0, 1]
   * - ``tau``
     - float
     - 1

.. dropdown:: ``FPFCServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the FPFC algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float, default 1\n    The ratio of clients to participate in each round.\nrho : float, default 1.0\n    The penalty parameter :math:`\\rho` of the augmented Lagrangian (ADMM).\nxi : float, default 1e-4\n    The smoothing parameter :math:`\\xi` of the (smoothed SCAD-type) fusion penalty.\nlam : float, default 1.0\n    The penalty coefficient :math:`\\lambda` of the pairwise fusion penalty.\na : float, default 3.7\n    The parameter :math:`a` of the fusion penalty, should be larger than 1.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "optimizer": {
          "default": "Avg",
          "title": "Optimizer",
          "type": "string"
        },
        "lr": {
          "default": 1,
          "title": "Lr",
          "type": "number"
        },
        "betas": {
          "default": [
            0,
            1
          ],
          "maxItems": 2,
          "minItems": 2,
          "prefixItems": [
            {
              "type": "number"
            },
            {
              "type": "number"
            }
          ],
          "title": "Betas",
          "type": "array"
        },
        "tau": {
          "default": 1,
          "title": "Tau",
          "type": "number"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio"
      ],
      "title": "FPFCServerConfig",
      "type": "object"
    }

**Client config:** ``FPFCClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "SGD"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""

.. dropdown:: ``FPFCClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the FPFC algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of (local) epochs, i.e. :math:`T_i` in the paper.\nlr : float, default 1e-2\n    The learning rate, i.e. the stepsize :math:`\\alpha` in the paper.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "SGD",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "FPFCClientConfig",
      "type": "object"
    }


.. _config-fedadagrad:

FedAdagrad
----------

**Server config:** ``FedAdagradServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``optimizer``
     - str
     - "Adagrad"
   * - ``lr``
     - float
     - 0.01
   * - ``betas``
     - list
     - [0.0, 0.99]
   * - ``tau``
     - float
     - 1e-05

.. dropdown:: ``FedAdagradServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the FedOpt algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float\n    The ratio of clients to sample for each iteration.\nlr : float, default 1e-2\n    The learning rate.\nbetas : Sequence[float], default (0.9, 0.99)\n    The beta values for the optimizer.\ntau : float, default 1e-5\n    The tau value for the optimizer,\n    which controls the degree of adaptivity of the algorithm.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "optimizer": {
          "default": "Adagrad",
          "title": "Optimizer",
          "type": "string"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "betas": {
          "default": [
            0.0,
            0.99
          ],
          "maxItems": 2,
          "minItems": 2,
          "prefixItems": [
            {
              "type": "number"
            },
            {
              "type": "number"
            }
          ],
          "title": "Betas",
          "type": "array"
        },
        "tau": {
          "default": 1e-05,
          "title": "Tau",
          "type": "number"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio"
      ],
      "title": "FedAdagradServerConfig",
      "type": "object"
    }

**Client config:** ``FedAdagradClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "SGD"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""

.. dropdown:: ``FedAdagradClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the FedAdagrad algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 1e-2\n    The learning rate.\noptimizer : str, default \"SGD\"\n    The name of the optimizer to solve the local (inner) problem.\n**kwargs : dict, optional\n    Additional keyword arguments for specific algorithms\n    (FedAvg, FedAdagrad, FedYogi, FedAdam). And\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "SGD",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "FedAdagradClientConfig",
      "type": "object"
    }


.. _config-fedadam:

FedAdam
-------

**Server config:** ``FedAdamServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``optimizer``
     - str
     - "Adam"
   * - ``lr``
     - float
     - 0.01
   * - ``betas``
     - list
     - [0.9, 0.99]
   * - ``tau``
     - float
     - 1e-05

.. dropdown:: ``FedAdamServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the FedOpt algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float\n    The ratio of clients to sample for each iteration.\nlr : float, default 1e-2\n    The learning rate.\nbetas : Sequence[float], default (0.9, 0.99)\n    The beta values for the optimizer.\ntau : float, default 1e-5\n    The tau value for the optimizer,\n    which controls the degree of adaptivity of the algorithm.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "optimizer": {
          "default": "Adam",
          "title": "Optimizer",
          "type": "string"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "betas": {
          "default": [
            0.9,
            0.99
          ],
          "maxItems": 2,
          "minItems": 2,
          "prefixItems": [
            {
              "type": "number"
            },
            {
              "type": "number"
            }
          ],
          "title": "Betas",
          "type": "array"
        },
        "tau": {
          "default": 1e-05,
          "title": "Tau",
          "type": "number"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio"
      ],
      "title": "FedAdamServerConfig",
      "type": "object"
    }

**Client config:** ``FedAdamClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "SGD"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""

.. dropdown:: ``FedAdamClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the FedAdam algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 1e-2\n    The learning rate.\noptimizer : str, default \"SGD\"\n    The name of the optimizer to solve the local (inner) problem.\n**kwargs : dict, optional\n    Additional keyword arguments for specific algorithms\n    (FedAvg, FedAdagrad, FedYogi, FedAdam). And\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "SGD",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "FedAdamClientConfig",
      "type": "object"
    }


.. _config-fedavg:

FedAvg
------

**Server config:** ``FedAvgServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``optimizer``
     - str
     - "Avg"
   * - ``lr``
     - float
     - 1
   * - ``betas``
     - list
     - [0, 1]
   * - ``tau``
     - float
     - 1

.. dropdown:: ``FedAvgServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the FedOpt algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float\n    The ratio of clients to sample for each iteration.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "optimizer": {
          "default": "Avg",
          "title": "Optimizer",
          "type": "string"
        },
        "lr": {
          "default": 1,
          "title": "Lr",
          "type": "number"
        },
        "betas": {
          "default": [
            0,
            1
          ],
          "maxItems": 2,
          "minItems": 2,
          "prefixItems": [
            {
              "type": "number"
            },
            {
              "type": "number"
            }
          ],
          "title": "Betas",
          "type": "array"
        },
        "tau": {
          "default": 1,
          "title": "Tau",
          "type": "number"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio"
      ],
      "title": "FedAvgServerConfig",
      "type": "object"
    }

**Client config:** ``FedAvgClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "SGD"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""

.. dropdown:: ``FedAvgClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the FedAvg algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 1e-2\n    The learning rate.\noptimizer : str, default \"SGD\"\n    The name of the optimizer to solve the local (inner) problem.\n**kwargs : dict, optional\n    Additional keyword arguments for specific algorithms\n    (FedAvg, FedAdagrad, FedYogi, FedAdam). And\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "SGD",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "FedAvgClientConfig",
      "type": "object"
    }


.. _config-fedcr:

FedCR
-----

**Server config:** ``FedCRServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - 1
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``optimizer``
     - str
     - "Avg"
   * - ``lr``
     - float
     - 1
   * - ``betas``
     - list
     - [0, 1]
   * - ``tau``
     - float
     - 1
   * - ``beta``
     - float
     - 0.005

.. dropdown:: ``FedCRServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the FedCR algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float, default 1\n    The ratio of clients to participate in each round.\nbeta : float, default 0.005\n    The coefficient :math:`\\beta` of the CMI (KL alignment) regularizer.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "default": 1,
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "optimizer": {
          "default": "Avg",
          "title": "Optimizer",
          "type": "string"
        },
        "lr": {
          "default": 1,
          "title": "Lr",
          "type": "number"
        },
        "betas": {
          "default": [
            0,
            1
          ],
          "maxItems": 2,
          "minItems": 2,
          "prefixItems": [
            {
              "type": "number"
            },
            {
              "type": "number"
            }
          ],
          "title": "Betas",
          "type": "array"
        },
        "tau": {
          "default": 1,
          "title": "Tau",
          "type": "number"
        },
        "beta": {
          "default": 0.005,
          "title": "Beta",
          "type": "number"
        }
      },
      "required": [
        "num_iters"
      ],
      "title": "FedCRServerConfig",
      "type": "object"
    }

**Client config:** ``FedCRClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "SGD"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""

.. dropdown:: ``FedCRClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the FedCR algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of (local) epochs.\nlr : float, default 1e-2\n    The learning rate.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "SGD",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "FedCRClientConfig",
      "type": "object"
    }


.. _config-feddc:

FedDC
-----

**Server config:** ``FedDCServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - 1
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``optimizer``
     - str
     - "Avg"
   * - ``lr``
     - float
     - 1
   * - ``betas``
     - list
     - [0, 1]
   * - ``tau``
     - float
     - 1
   * - ``alpha``
     - float
     - 0.01

.. dropdown:: ``FedDCServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the FedDC algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float, default 1\n    The ratio of clients to participate in each round.\nalpha : float, default 0.01\n    The coefficient :math:`\\alpha` of the parameter correction\n    (penalized) term :math:`\\frac{\\alpha}{2} \\lVert \\theta_i - (w - h_i) \\rVert^2`.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "default": 1,
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "optimizer": {
          "default": "Avg",
          "title": "Optimizer",
          "type": "string"
        },
        "lr": {
          "default": 1,
          "title": "Lr",
          "type": "number"
        },
        "betas": {
          "default": [
            0,
            1
          ],
          "maxItems": 2,
          "minItems": 2,
          "prefixItems": [
            {
              "type": "number"
            },
            {
              "type": "number"
            }
          ],
          "title": "Betas",
          "type": "array"
        },
        "tau": {
          "default": 1,
          "title": "Tau",
          "type": "number"
        },
        "alpha": {
          "default": 0.01,
          "title": "Alpha",
          "type": "number"
        }
      },
      "required": [
        "num_iters"
      ],
      "title": "FedDCServerConfig",
      "type": "object"
    }

**Client config:** ``FedDCClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "SGD"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""

.. dropdown:: ``FedDCClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the FedDC algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of (local) epochs.\nlr : float, default 1e-2\n    The learning rate :math:`\\eta`.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "SGD",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "FedDCClientConfig",
      "type": "object"
    }


.. _config-feddr:

FedDR
-----

**Server config:** ``FedDRServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``eta``
     - float
     - 1.0
   * - ``alpha``
     - float
     - 1.9
   * - ``reg_type``
     - str
     - "l1_norm"

.. dropdown:: ``FedDRServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the FedDR algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float\n    The ratio of clients to sample for each iteration.\neta : float, default 1.0\n    The eta (regularization) parameter for the FedDR algorithm.\nalpha : float, default 1.9\n    The alpha parameter for the FedDR algorithm.\nreg_type : str, default \"l1_norm\"\n    The type of regularizer to use for the FedDR algorithm.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "eta": {
          "default": 1.0,
          "title": "Eta",
          "type": "number"
        },
        "alpha": {
          "default": 1.9,
          "title": "Alpha",
          "type": "number"
        },
        "reg_type": {
          "default": "l1_norm",
          "title": "Reg Type",
          "type": "string"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio"
      ],
      "title": "FedDRServerConfig",
      "type": "object"
    }

**Client config:** ``FedDRClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "FedDR"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""
   * - ``eta``
     - float
     - 1.0
   * - ``alpha``
     - float
     - 1.9

.. dropdown:: ``FedDRClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the FedDR algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 1e-2\n    The learning rate.\neta : float, default 1.0\n    The eta (regularization) parameter for the FedDR algorithm.\nalpha : float, default 1.9\n    The alpha parameter for the FedDR algorithm.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "FedDR",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "eta": {
          "default": 1.0,
          "title": "Eta",
          "type": "number"
        },
        "alpha": {
          "default": 1.9,
          "title": "Alpha",
          "type": "number"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "FedDRClientConfig",
      "type": "object"
    }


.. _config-feddyn:

FedDyn
------

**Server config:** ``FedDynServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``mu``
     - float
     - 0.1

.. dropdown:: ``FedDynServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the FedDyn algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float\n    The ratio of clients to sample for each iteration.\nmu : float, default 1 / 10\n    The coefficient of the \"proximal\" term.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "mu": {
          "default": 0.1,
          "title": "Mu",
          "type": "number"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio"
      ],
      "title": "FedDynServerConfig",
      "type": "object"
    }

**Client config:** ``FedDynClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - ""
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""

.. dropdown:: ``FedDynClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the FedDyn algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 1e-2\n    The learning rate.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "FedDynClientConfig",
      "type": "object"
    }


.. _config-fedopt:

FedOpt
------

**Server config:** ``FedOptServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``optimizer``
     - str
     - "Adam"
   * - ``lr``
     - float
     - 0.01
   * - ``betas``
     - list
     - [0.9, 0.99]
   * - ``tau``
     - float
     - 1e-05

.. dropdown:: ``FedOptServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the FedOpt algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float\n    The ratio of clients to sample for each iteration.\noptimizer : {\"SGD\", \"Adam\", \"Adagrad\", \"Yogi\"}, default \"Adam\"\n    The optimizer to use, case insensitive.\nlr : float, default 1e-2\n    The learning rate.\nbetas : Sequence[float], default (0.9, 0.99)\n    The beta values for the optimizer.\ntau : float, default 1e-5\n    The tau value for the optimizer,\n    which controls the degree of adaptivity of the algorithm.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "optimizer": {
          "default": "Adam",
          "title": "Optimizer",
          "type": "string"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "betas": {
          "default": [
            0.9,
            0.99
          ],
          "maxItems": 2,
          "minItems": 2,
          "prefixItems": [
            {
              "type": "number"
            },
            {
              "type": "number"
            }
          ],
          "title": "Betas",
          "type": "array"
        },
        "tau": {
          "default": 1e-05,
          "title": "Tau",
          "type": "number"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio"
      ],
      "title": "FedOptServerConfig",
      "type": "object"
    }

**Client config:** ``FedOptClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "SGD"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""

.. dropdown:: ``FedOptClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the FedOpt algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 1e-2\n    The learning rate.\noptimizer : str, default \"SGD\"\n    The name of the optimizer to solve the local (inner) problem.\n**kwargs : dict, optional\n    Additional keyword arguments for specific algorithms\n    (FedAvg, FedAdagrad, FedYogi, FedAdam). And\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "SGD",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "FedOptClientConfig",
      "type": "object"
    }


.. _config-fedpd:

FedPD
-----

**Server config:** ``FedPDServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - 1
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``p``
     - float
     - *
   * - ``stochastic``
     - bool
     - False
   * - ``vr``
     - bool
     - False

.. dropdown:: ``FedPDServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the FedPD algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\np : float\n    Probability of skipping communication.\nstochastic : bool, default False\n    Skip communication in a stochastic manner or not.\nvr : bool, default False\n    Whether to use variance reduction.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "default": 1,
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "p": {
          "title": "P",
          "type": "number"
        },
        "stochastic": {
          "default": false,
          "title": "Stochastic",
          "type": "boolean"
        },
        "vr": {
          "default": false,
          "title": "Vr",
          "type": "boolean"
        }
      },
      "required": [
        "num_iters",
        "p"
      ],
      "title": "FedPDServerConfig",
      "type": "object"
    }

**Client config:** ``FedPDClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - ""
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""
   * - ``mu``
     - float
     - 0.1
   * - ``vr``
     - bool
     - False
   * - ``dual_rand_init``
     - bool
     - False

.. dropdown:: ``FedPDClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the FedPD algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 1e-2\n    The learning rate.\nmu : float, default 1 / 10\n    The coefficient of the proximal term.\nvr : bool, default False\n    Whether to use variance reduction.\ndual_rand_init : bool, default False\n    Whether to use random initialization for dual variables.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "mu": {
          "default": 0.1,
          "title": "Mu",
          "type": "number"
        },
        "vr": {
          "default": false,
          "title": "Vr",
          "type": "boolean"
        },
        "dual_rand_init": {
          "default": false,
          "title": "Dual Rand Init",
          "type": "boolean"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "FedPDClientConfig",
      "type": "object"
    }


.. _config-fedprox:

FedProx
-------

**Server config:** ``FedProxServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``vr``
     - bool
     - False

.. dropdown:: ``FedProxServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the FedProx algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float\n    The ratio of clients to sample for each iteration.\nvr : bool, default False\n    Whether to use variance reduction.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "vr": {
          "default": false,
          "title": "Vr",
          "type": "boolean"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio"
      ],
      "title": "FedProxServerConfig",
      "type": "object"
    }

**Client config:** ``FedProxClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - ""
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""
   * - ``mu``
     - float
     - 0.01
   * - ``vr``
     - bool
     - False

.. dropdown:: ``FedProxClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the FedProx algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 1e-2\n    The learning rate.\nmu : float, default 0.01\n    Coefficient for the proximal term.\nvr : bool, default False\n    Whether to use variance reduction.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "mu": {
          "default": 0.01,
          "title": "Mu",
          "type": "number"
        },
        "vr": {
          "default": false,
          "title": "Vr",
          "type": "boolean"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "FedProxClientConfig",
      "type": "object"
    }


.. _config-fedsplit:

FedSplit
--------

**Server config:** ``FedSplitServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""

.. dropdown:: ``FedSplitServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the FedSplit algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float\n    The ratio of clients to sample for each iteration.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio"
      ],
      "title": "FedSplitServerConfig",
      "type": "object"
    }

**Client config:** ``FedSplitClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "ProxSGD"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""
   * - ``s``
     - float
     - 10.0

.. dropdown:: ``FedSplitClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the FedSplit algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 1e-2\n    The learning rate.\ns : float, default 10.0\n    Reciprocal of the proximal parameter.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "ProxSGD",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "s": {
          "default": 10.0,
          "title": "S",
          "type": "number"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "FedSplitClientConfig",
      "type": "object"
    }


.. _config-fedyogi:

FedYogi
-------

**Server config:** ``FedYogiServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``optimizer``
     - str
     - "Yogi"
   * - ``lr``
     - float
     - 0.01
   * - ``betas``
     - list
     - [0.9, 0.99]
   * - ``tau``
     - float
     - 1e-05

.. dropdown:: ``FedYogiServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the FedOpt algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float\n    The ratio of clients to sample for each iteration.\nlr : float, default 1e-2\n    The learning rate.\nbetas : Sequence[float], default (0.9, 0.99)\n    The beta values for the optimizer.\ntau : float, default 1e-5\n    The tau value for the optimizer,\n    which controls the degree of adaptivity of the algorithm.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "optimizer": {
          "default": "Yogi",
          "title": "Optimizer",
          "type": "string"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "betas": {
          "default": [
            0.9,
            0.99
          ],
          "maxItems": 2,
          "minItems": 2,
          "prefixItems": [
            {
              "type": "number"
            },
            {
              "type": "number"
            }
          ],
          "title": "Betas",
          "type": "array"
        },
        "tau": {
          "default": 1e-05,
          "title": "Tau",
          "type": "number"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio"
      ],
      "title": "FedYogiServerConfig",
      "type": "object"
    }

**Client config:** ``FedYogiClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "SGD"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""

.. dropdown:: ``FedYogiClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the FedYogi algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 1e-2\n    The learning rate.\noptimizer : str, default \"SGD\"\n    The name of the optimizer to solve the local (inner) problem.\n**kwargs : dict, optional\n    Additional keyword arguments for specific algorithms\n    (FedAvg, FedAdagrad, FedYogi, FedAdam). And\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "SGD",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "FedYogiClientConfig",
      "type": "object"
    }


.. _config-ifca:

IFCA
----

**Server config:** ``IFCAServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - 1
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``optimizer``
     - str
     - "Avg"
   * - ``lr``
     - float
     - 1
   * - ``betas``
     - list
     - [0, 1]
   * - ``tau``
     - float
     - 1
   * - ``num_clusters``
     - int
     - *

.. dropdown:: ``IFCAServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the IFCA algorithm.\n\nParameters\n----------\nnum_clusters : int\n    The number of clusters.\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float, default 1\n    The ratio of clients to participate in each round.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "default": 1,
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "optimizer": {
          "default": "Avg",
          "title": "Optimizer",
          "type": "string"
        },
        "lr": {
          "default": 1,
          "title": "Lr",
          "type": "number"
        },
        "betas": {
          "default": [
            0,
            1
          ],
          "maxItems": 2,
          "minItems": 2,
          "prefixItems": [
            {
              "type": "number"
            },
            {
              "type": "number"
            }
          ],
          "title": "Betas",
          "type": "array"
        },
        "tau": {
          "default": 1,
          "title": "Tau",
          "type": "number"
        },
        "num_clusters": {
          "title": "Num Clusters",
          "type": "integer"
        }
      },
      "required": [
        "num_iters",
        "num_clusters"
      ],
      "title": "IFCAServerConfig",
      "type": "object"
    }

**Client config:** ``IFCAClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "SGD"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""

.. dropdown:: ``IFCAClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the IFCA algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 1e-2\n    The learning rate.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "SGD",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "IFCAClientConfig",
      "type": "object"
    }


.. _config-proxskip:

ProxSkip
--------

**Server config:** ``ProxSkipServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - 1
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``p``
     - float
     - *
   * - ``vr``
     - bool
     - False

.. dropdown:: ``ProxSkipServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the ProxSkip algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\np : float\n    Probability of skipping communication.\nvr : bool, default False\n    Whether to use variance reduction.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "default": 1,
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "p": {
          "title": "P",
          "type": "number"
        },
        "vr": {
          "default": false,
          "title": "Vr",
          "type": "boolean"
        }
      },
      "required": [
        "num_iters",
        "p"
      ],
      "title": "ProxSkipServerConfig",
      "type": "object"
    }

**Client config:** ``ProxSkipClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "SCAFFOLD"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""
   * - ``vr``
     - bool
     - False

.. dropdown:: ``ProxSkipClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the ProxSkip algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 1e-2\n    The learning rate.\nvr : bool, default False\n    Whether to use variance reduction.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "SCAFFOLD",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "vr": {
          "default": false,
          "title": "Vr",
          "type": "boolean"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "ProxSkipClientConfig",
      "type": "object"
    }


.. _config-scaffold:

SCAFFOLD
--------

**Server config:** ``SCAFFOLDServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``lr``
     - float
     - *

.. dropdown:: ``SCAFFOLDServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the SCAFFOLD algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float\n    The ratio of clients to sample for each iteration.\nlr : float\n    The learning rate.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "lr": {
          "title": "Lr",
          "type": "number"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio",
        "lr"
      ],
      "title": "SCAFFOLDServerConfig",
      "type": "object"
    }

**Client config:** ``SCAFFOLDClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "SGD"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""
   * - ``control_variate_update_rule``
     - int
     - 1

.. dropdown:: ``SCAFFOLDClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the SCAFFOLD algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 1e-2\n    The learning rate.\ncontrol_variate_update_rule : int, default 1\n    The update rule for the control variates.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "SGD",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "control_variate_update_rule": {
          "default": 1,
          "title": "Control Variate Update Rule",
          "type": "integer"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "SCAFFOLDClientConfig",
      "type": "object"
    }


.. _config-pfedmac:

pFedMac
-------

**Server config:** ``pFedMacServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``beta``
     - float
     - 1.0
   * - ``vr``
     - bool
     - False

.. dropdown:: ``pFedMacServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the pFedMac algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float\n    The ratio of clients to sample for each iteration.\nbeta : float, default 1.0\n    The beta (inertia) parameter for pFedMac.\nvr : bool, default False\n    Whether to use variance reduction.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "beta": {
          "default": 1.0,
          "title": "Beta",
          "type": "number"
        },
        "vr": {
          "default": false,
          "title": "Vr",
          "type": "boolean"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio"
      ],
      "title": "pFedMacServerConfig",
      "type": "object"
    }

**Client config:** ``pFedMacClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.01
   * - ``optimizer``
     - str
     - "pFedMac"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""
   * - ``lam``
     - float
     - 15.0
   * - ``vr``
     - bool
     - False

.. dropdown:: ``pFedMacClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the pFedMac algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 1e-2\n    The learning rate.\nlam : float, default 15.0\n    The lambda parameter for pFedMac.\nvr : bool, default False\n    Whether to use variance reduction.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.01,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "pFedMac",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "lam": {
          "default": 15.0,
          "title": "Lam",
          "type": "number"
        },
        "vr": {
          "default": false,
          "title": "Vr",
          "type": "boolean"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "pFedMacClientConfig",
      "type": "object"
    }


.. _config-pfedme:

pFedMe
------

**Server config:** ``pFedMeServerConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``num_iters``
     - int
     - *
   * - ``num_clients``
     - Optional[int]
     - None
   * - ``clients_sample_ratio``
     - float
     - *
   * - ``log_dir``
     - Optional[str]
     - None
   * - ``txt_logger``
     - bool
     - True
   * - ``json_logger``
     - bool
     - True
   * - ``eval_every``
     - int
     - 1
   * - ``visiable_gpus``
     - Optional[list[int]]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``seed``
     - int
     - 0
   * - ``tag``
     - Optional[str]
     - None
   * - ``verbose``
     - int
     - 1
   * - ``gpu_proportion``
     - float
     - 0.2
   * - ``algorithm``
     - str
     - ""
   * - ``beta``
     - float
     - 1.0

.. dropdown:: ``pFedMeServerConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Server config for the pFedMe algorithm.\n\nParameters\n----------\nnum_iters : int\n    The number of (outer) iterations.\nnum_clients : int\n    The number of clients.\nclients_sample_ratio : float\n    The ratio of clients to sample for each iteration.\nbeta : float, default 1.0\n    The beta (inertia) parameter for model aggregation.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``log_dir`` : str or pathlib.Path, optional\n        The log directory.\n        If not specified, will use the default log directory.\n        If not absolute, will be relative to the default log directory.\n    - ``txt_logger`` : bool, default True\n        Whether to use txt logger.\n    - ``json_logger`` : bool, default True\n        Whether to use json logger.\n    - ``eval_every`` : int, default 1\n        The number of iterations to evaluate the model.\n    - ``visiable_gpus`` : Sequence[int], optional\n        Visable GPU IDs for allocating devices for clients.\n        Defaults to use all GPUs if available.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training.\n    - ``seed`` : int, default 0\n        The random seed.\n    - ``tag`` : str, optional\n        The tag of the experiment.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``gpu_proportion`` : float, default 0.2\n        The proportion of clients to use GPU.\n        Used to similate the system heterogeneity of the clients.\n        Not used in the current version, reserved for future use.",
      "properties": {
        "num_iters": {
          "title": "Num Iters",
          "type": "integer"
        },
        "num_clients": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Num Clients"
        },
        "clients_sample_ratio": {
          "title": "Clients Sample Ratio",
          "type": "number"
        },
        "log_dir": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "format": "path",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Log Dir"
        },
        "txt_logger": {
          "default": true,
          "title": "Txt Logger",
          "type": "boolean"
        },
        "json_logger": {
          "default": true,
          "title": "Json Logger",
          "type": "boolean"
        },
        "eval_every": {
          "default": 1,
          "title": "Eval Every",
          "type": "integer"
        },
        "visiable_gpus": {
          "anyOf": [
            {
              "items": {
                "type": "integer"
              },
              "type": "array"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Visiable Gpus"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "seed": {
          "default": 0,
          "title": "Seed",
          "type": "integer"
        },
        "tag": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Tag"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "gpu_proportion": {
          "default": 0.2,
          "title": "Gpu Proportion",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "beta": {
          "default": 1.0,
          "title": "Beta",
          "type": "number"
        }
      },
      "required": [
        "num_iters",
        "clients_sample_ratio"
      ],
      "title": "pFedMeServerConfig",
      "type": "object"
    }

**Client config:** ``pFedMeClientConfig``

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Field
     - Type
     - Default
   * - ``batch_size``
     - int
     - *
   * - ``num_epochs``
     - int
     - *
   * - ``lr``
     - float
     - 0.005
   * - ``optimizer``
     - str
     - "pFedMe"
   * - ``scheduler``
     - Optional[dict]
     - None
   * - ``extra_observes``
     - list[str]
     - (factory)
   * - ``verbose``
     - int
     - 1
   * - ``latency``
     - float
     - 0.0
   * - ``algorithm``
     - str
     - ""
   * - ``num_steps``
     - int
     - 30
   * - ``lamda``
     - float
     - 15.0
   * - ``eta``
     - float
     - 0.001
   * - ``mu``
     - float
     - 0.001

.. dropdown:: ``pFedMeClientConfig`` JSON schema
   :animate: fade-in-slide-down

   .. code-block:: json

    {
      "additionalProperties": true,
      "description": "Client config for the pFedMe algorithm.\n\nParameters\n----------\nbatch_size : int\n    The batch size.\nnum_epochs : int\n    The number of epochs.\nlr : float, default 5e-3\n    The learning rate for personalized model training.\nnum_steps : int, default 30\n    The number of steps for each epoch.\nlamda : float, default 15.0\n    The lambda parameter for pFedMe,\n    i.e. the coefficient of the proximal term.\neta : float, default 1e-3\n    The eta (learning rate) parameter for pFedMe.\nmu : float, default 1e-3\n    The mu (momentum) parameter for pFedMe.\n**kwargs : dict, optional\n    Additional keyword arguments:\n\n    - ``scheduler`` : dict, optional\n        The scheduler config.\n        None for no scheduler, using constant learning rate.\n    - ``extra_observes`` : List[str], optional\n        Extra attributes to observe during training,\n        which would be recorded in evaluated metrics,\n        sent to the server, and written to the log file.\n    - ``verbose`` : int, default 1\n        The verbosity level.\n    - ``latency`` : float, default 0.0\n        The latency of the client.\n        Not used in the current version, reserved for future use.\n\n\nReferences\n----------\n.. [1] https://github.com/CharlieDinh/pFedMe/blob/master/FLAlgorithms/users/userpFedMe.py\n\n.. note::\n\n    1. `lr` is the `personal_learning_rate` in the original implementation\n    2. `eta` is the `learning_rate` in the original implementation\n    3. `mu` is the momentum factor in the original implemented optimzer",
      "properties": {
        "batch_size": {
          "title": "Batch Size",
          "type": "integer"
        },
        "num_epochs": {
          "title": "Num Epochs",
          "type": "integer"
        },
        "lr": {
          "default": 0.005,
          "title": "Lr",
          "type": "number"
        },
        "optimizer": {
          "default": "pFedMe",
          "title": "Optimizer",
          "type": "string"
        },
        "scheduler": {
          "anyOf": [
            {
              "additionalProperties": true,
              "type": "object"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Scheduler"
        },
        "extra_observes": {
          "items": {
            "type": "string"
          },
          "title": "Extra Observes",
          "type": "array"
        },
        "verbose": {
          "default": 1,
          "title": "Verbose",
          "type": "integer"
        },
        "latency": {
          "default": 0.0,
          "title": "Latency",
          "type": "number"
        },
        "algorithm": {
          "default": "",
          "title": "Algorithm",
          "type": "string"
        },
        "num_steps": {
          "default": 30,
          "title": "Num Steps",
          "type": "integer"
        },
        "lamda": {
          "default": 15.0,
          "title": "Lamda",
          "type": "number"
        },
        "eta": {
          "default": 0.001,
          "title": "Eta",
          "type": "number"
        },
        "mu": {
          "default": 0.001,
          "title": "Mu",
          "type": "number"
        }
      },
      "required": [
        "batch_size",
        "num_epochs"
      ],
      "title": "pFedMeClientConfig",
      "type": "object"
    }
