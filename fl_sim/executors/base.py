"""The client-executor abstraction of the parallel execution layer.

The design (``tmp/parallel-execution-design.md`` v5, §5.2) fixes the
executor contract to a single method: the main loop only ever needs
"run this batch of client tasks, hand back the results in the sampled
order, and tell me as each one finishes". Exposing anything richer
(futures, per-task submission) would invite the caller to move server-state
mutations into the concurrent region, breaking the state-ownership
invariant I2 (server mutations happen on the main thread only, in the
sampled client order).

"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, TypeVar

__all__ = [
    "ClientExecutor",
]

T = TypeVar("T")


class ClientExecutor(ABC):
    """Maps a batch of client tasks onto execution resources.

    Contract
    --------
    - :meth:`map` calls ``fn`` once per item and returns the results as a
      list whose order matches ``items`` (never completion order — the
      aggregation consumes the buffer in the sampled order, which is what
      makes the aggregation arithmetic backend-independent).
    - ``on_complete(result, index)`` is called on the main thread as soon
      as each task finishes, driving progress feedback; it must not mutate
      server aggregation state.
    - a task exception is re-raised with the failing client's context
      (``raise ... from exc``); pending not-yet-started tasks are cancelled
      on a best-effort basis (already-running ones inevitably run to
      completion — a thread/process-pool physical fact, documented).
    - instances are context managers: entering saves and applies the
      execution resources (e.g. the intra-op thread budget), exiting
      restores the previous values.
    """

    @abstractmethod
    def map(
        self,
        fn: Callable[[object], T],
        items: Sequence[object],
        on_complete: Optional[Callable[[T, int], None]] = None,
    ) -> List[T]:
        raise NotImplementedError

    def __enter__(self) -> "ClientExecutor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None
