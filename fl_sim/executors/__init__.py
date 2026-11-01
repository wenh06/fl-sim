"""
fl_sim.executors
================

Pluggable client-executor backends for the parallel execution layer.

The executor maps a batch of per-client tasks onto execution resources
while preserving the sampled order of the results; see
:class:`~fl_sim.executors.base.ClientExecutor` for the contract.

"""

from .base import ClientExecutor
from .sequential import SequentialExecutor
from .threads import ThreadsExecutor

__all__ = [
    "ClientExecutor",
    "SequentialExecutor",
    "ThreadsExecutor",
]
