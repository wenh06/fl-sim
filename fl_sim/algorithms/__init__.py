"""
"""

from . import (  # noqa: F401
    apfl,
    ditto,
    feddr,
    feddyn,
    fedopt,
    fedpd,
    fedprox,
    fedsplit,
    ifca,
    pfedmac,
    pfedme,
    proxskip,
    scaffold,
)
from ._register import get_algorithm, list_algorithms, register_algorithm  # noqa: F401

# check all registered algorithms have required attributes
for name in list_algorithms():
    assert set(["server_config", "client_config", "server"]).issubset(
        set(get_algorithm(name))
    ), f"{name} has missing attributes."
del name
builtin_algorithms = list_algorithms()


__all__ = [
    "list_algorithms",
    "get_algorithm",
    "register_algorithm",
    "builtin_algorithms",
]
