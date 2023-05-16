"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.ditto import test_ditto as ditto_test_func


def test_ditto():
    """ """
    ditto_test_func()
