Installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`fl-sim` requires **Python 3.6+** and is available through pip:

.. code-block:: bash
    
    pip install git+https://github.com/wenh06/fl-sim.git

or clone the repository and run

.. code-block:: bash

    pip install -e .

Alternatively, one can use the Docker image `wenh06/fl-sim <https://hub.docker.com/r/wenh06/fl-sim>`_ to run the code.
The image is built with the `Docker Image CI action <https://github.com/wenh06/fl-sim/actions/workflows/docker-image.yml>`_.
To pull the image, run the following command:

.. code-block:: bash

    docker pull wenh06/fl-sim

For the usage (interactive mode), run the following command:

.. code-block:: bash

    docker run -it wenh06/fl-sim bash

For more advanced usages (e.g., run a script), refer to the `Docker official documentation <https://docs.docker.com/engine/reference/commandline/run/>`_.
