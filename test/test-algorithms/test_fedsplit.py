"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.fedsplit import test_fedsplit as fedsplit_test_func


def test_fedsplit():
    """ """
    fedsplit_test_func()
