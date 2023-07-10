"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.feddyn import test_feddyn as feddyn_test_func


def test_feddyn():
    """ """
    feddyn_test_func()
