"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.pfedmac import test_pfedmac as pfedmac_test_func


def test_pfedmac():
    """ """
    pfedmac_test_func()
