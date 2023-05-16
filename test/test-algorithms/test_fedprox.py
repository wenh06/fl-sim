"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.fedprox import test_fedprox as fedprox_test_func


def test_fedprox():
    """ """
    fedprox_test_func()
