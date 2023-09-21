"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1].resolve()))

from fl_sim.utils.viz import Panel

logdir = Path(__file__).parents[1].resolve() / "test-files"


def test_panel():
    p = Panel(logdir=logdir)
    # TODO: test the following:
    # select log files from the SelectMultiple widget and plot curves
    # change the part and the metrics Text widgets and plot curves
    # adjust plt rc params via the slider widgets and plot curves
    # and more....
