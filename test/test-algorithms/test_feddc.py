""" """

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.feddc import test_feddc as feddc_test_func


def test_feddc():
    """ """
    feddc_test_func()
