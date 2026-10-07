""" """

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.fedcr import test_fedcr as fedcr_test_func


def test_fedcr():
    """ """
    fedcr_test_func()
