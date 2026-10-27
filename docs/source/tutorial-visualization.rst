.. _tutorial-visualization:

Tutorial: visualize and export results
======================================

Every run writes ``.json`` logs (see :doc:`quickstart`); fl-sim ships a
visualization layer that turns them into publication-ready curves —
interactively in Jupyter, or programmatically from plain Python. This
tutorial does both: merge a multi-seed experiment into one mean curve
with error bounds, export it to PDF, and open the interactive dashboard.

The full script lives at ``test-files/tutorial_visualization.py`` and is
exercised in CI (everything except the dashboard rendering, which needs
Jupyter). Prerequisites: :doc:`tutorial-first-experiment`.

The reference for all knobs (fonts, smoothing, widgets) is
:doc:`viz`; this page is the workflow.

Get some logs
-------------

The script trains the same FedAvg experiment under two seeds — the
minimal "multi-seed" setup. Note the per-run ``tag``: log file names
carry it, so the two runs cannot overwrite each other (the same
reasoning as in :doc:`tutorial-cli-matrix`):

.. literalinclude:: ../../test-files/tutorial_visualization.py
   :language: python
   :start-after: # --- begin: setup ---
   :end-before: # --- end: setup ---

Merge seeds into one curve
--------------------------

``find_log_files`` collects the JSON logs; ``aggregate_results_from_json_log_with_custom_axis``
extracts a curve (``step`` on x, ``acc`` on y here — any recorded metric
works); ``plot_mean_curve_with_error_bounds`` merges the curves into a
mean with an uncertainty band (``"sem"``, ``"std"``, ``"quartile"`` or
``"iqr"``). Saving the figure is plain matplotlib:

.. literalinclude:: ../../test-files/tutorial_visualization.py
   :language: python
   :start-after: # --- begin: plot-mean ---
   :end-before: # --- end: plot-mean ---

This writes ``fedavg-multi-seed.pdf`` next to your working directory.
Two things worth noticing:

- The x axis is the **cumulative communication count** — the metric
  :doc:`tutorial-first-experiment` dissected. Plotting against
  communications (rather than rounds) is what makes different
  communication-efficiency algorithms comparable in one figure.
- With only two seeds the band is a range, not a statistic — use more
  seeds (just add more ``seed=`` runs; the merging code does not
  change).

Explore interactively
---------------------

In a Jupyter notebook, the ``Panel`` dashboard does all of the above
interactively: it lists the log files it finds, plots selectable
metrics, and gives you sliders for smoothing, fonts and figure size,
plus one-click export to PDF/SVG/PNG:

.. literalinclude:: ../../test-files/tutorial_visualization.py
   :language: python
   :start-after: # --- begin: panel ---
   :end-before: # --- end: panel ---

Run this cell in a notebook (JupyterLab, Colab, SageMaker, …) and the
dashboard renders inline; see :doc:`viz` for a demo animation and the
full feature list.

Next steps
----------

- :doc:`tutorial-parallel-reproducible` — make these runs faster
  (parallel executor) without giving up reproducibility.
- :doc:`viz` — the reference for every visualization option.
