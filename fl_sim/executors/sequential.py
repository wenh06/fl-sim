"""The sequential (default) client executor: exactly the pre-parallel for-loop."""

from typing import Callable, List, Optional, Sequence, TypeVar

from .base import ClientExecutor

__all__ = [
    "SequentialExecutor",
]

T = TypeVar("T")


class SequentialExecutor(ClientExecutor):
    """Runs the client tasks one by one, in the sampled order.

    The default backend: behaviorally identical to the pre-parallel
    implementation, down to the log ordering and the exception timing.
    """

    def map(
        self,
        fn: Callable[[object], T],
        items: Sequence[object],
        on_complete: Optional[Callable[[T, int], None]] = None,
    ) -> List[T]:
        results: List[T] = []
        for index, item in enumerate(items):
            result = fn(item)
            results.append(result)
            if on_complete is not None:
                on_complete(result, index)
        return results
