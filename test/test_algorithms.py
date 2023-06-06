"""
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1].resolve()))

from fl_sim.algorithms.ditto import test_ditto as ditto_test_func  # noqa: F401
from fl_sim.algorithms.feddr import test_feddr as feddr_test_func  # noqa: F401
from fl_sim.algorithms.fedopt import test_fedopt as fedopt_test_func  # noqa: F401
from fl_sim.algorithms.fedpd import test_fedpd as fedpd_test_func  # noqa: F401
from fl_sim.algorithms.fedprox import test_fedprox as fedprox_test_func
from fl_sim.algorithms.fedsplit import test_fedsplit as fedsplit_test_func  # noqa: F401
from fl_sim.algorithms.pfedmac import test_pfedmac as pfedmac_test_func  # noqa: F401
from fl_sim.algorithms.pfedme import test_pfedme as pfedme_test_func  # noqa: F401
from fl_sim.algorithms.proxskip import test_proxskip as proxskip_test_func  # noqa: F401
from fl_sim.algorithms.scaffold import test_scaffold as scaffold_test_func  # noqa: F401
from fl_sim.algorithms.ifca import test_ifca as ifca_test_func  # noqa: F401


def test_algorithms():
    """ """
    # ditto_test_func()
    # feddr_test_func()
    # fedopt_test_func("avg")
    # fedpd_test_func()
    fedprox_test_func()
    # fedsplit_test_func()
    # pfedmac_test_func()
    # pfedme_test_func()
    # proxskip_test_func()
    # scaffold_test_func()
    # ifca_test_func()
