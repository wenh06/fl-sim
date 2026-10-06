""" """

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.fpfc import test_fpfc as fpfc_test_func


def test_fpfc():
    """ """
    fpfc_test_func()
