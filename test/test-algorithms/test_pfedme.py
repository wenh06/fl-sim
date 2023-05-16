"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.pfedme import test_pfedme as pfedme_test_func


def test_pfedme():
    """ """
    pfedme_test_func()
