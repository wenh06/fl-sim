"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.apfl import test_apfl as apfl_test_func


def test_apfl():
    """ """
    apfl_test_func()
