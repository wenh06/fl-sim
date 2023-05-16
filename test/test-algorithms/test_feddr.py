"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.feddr import test_feddr as feddr_test_func


def test_feddr():
    """ """
    feddr_test_func()
