"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.fedopt import test_fedopt as fedopt_test_func


def test_fedopt():
    """ """
    fedopt_test_func("avg")
