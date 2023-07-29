Visualization subsystem
^^^^^^^^^^^^^^^^^^^^^^^^^

`fl-sim` implements a simple visualization subsystem named `Panel` along with a logging system.
`Panel` is based on Jupyter Widgets (`ipywidgets <https://ipywidgets.readthedocs.io/en/latest/>`_), `matplotlib <https://matplotlib.org/>`_ and `seaborn <https://seaborn.pydata.org/>`_.
It runs in a Jupyter Notebook or similar environment (Jupyter Lab, Google Colab, Amazon SageMaker, etc.).
`Panel` has the following features:

- It automatically searches, lists the log files of completed simulations (numerical experiments) in the specified directory, and displays them in a multi-select list box.
- It automatically decodes the log files and plots the curves of the specified metrics.
- It supports interactive operations on the plotted curves, such as zooming, smoothing, font family and size adjustment, etc.
- It supports saving the plotted curves to a file in PDF/SVG/PNG/JPEG/PS formats.
- It supports curve merging via tags into mean-value curves with error bars (optional). The error bounds have 4 options:

    - standard deviation (STD)
    - standard error of the mean (SEM)
    - quantile (QTL)
    - interquartile range (IQR)

The following GIF (created using `ScreenToGif <https://github.com/NickeManarin/ScreenToGif>`_) shows a demo of the visualization `Panel`:

.. figure:: ./_static/images/panel-demo.gif
   :align: center
   :width: 100%
   :alt: FL-SIM Panel Demo GIF

**NOTE:** to use Windows fonts on a Linux machine (e.g. Ubuntu), one can execute the following commands:

.. code-block:: bash

    $ sudo install ttf-mscorefonts-installer
    $ sudo fc-cache -fv
