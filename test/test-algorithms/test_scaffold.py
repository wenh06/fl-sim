"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.scaffold import test_scaffold as scaffold_test_func


def test_scaffold():
    """ """
    scaffold_test_func()
