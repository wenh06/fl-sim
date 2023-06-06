"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.ifca import test_ifca as ifca_test_func


def test_ifca():
    """ """
    ifca_test_func()
