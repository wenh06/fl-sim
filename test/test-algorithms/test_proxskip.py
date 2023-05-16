"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2].resolve()))

from fl_sim.algorithms.proxskip import test_proxskip as proxskip_test_func


def test_proxskip():
    """ """
    proxskip_test_func()
