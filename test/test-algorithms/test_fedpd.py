"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.fedpd import test_fedpd as fedpd_test_func


def test_fedpd():
    """ """
    fedpd_test_func()
