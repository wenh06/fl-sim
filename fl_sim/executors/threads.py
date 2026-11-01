"""The thread-pool client executor.

Threads share the process: client state needs no serialization, the client
models and datasets stay put, and PyTorch releases the GIL inside its
compute kernels (and CUDA calls), so the local-training workload parallelizes
despite the GIL. Two process-global resources need care and are handled here:

- the intra-op thread budget: PyTorch defaults to one CPU thread per core
  per process; ``num_workers`` concurrent clients times that default would
  oversubscribe the machine, so on entering, the budget is saved and set to
  ``torch_num_threads`` if given, else ``cores // num_workers`` (at least
  1), and restored on exit;
- the automatic worker count: ``num_workers = 0`` resolves to the number of
  usable CPUs via ``os.sched_getaffinity`` (which respects cgroup/affinity
  limits inside containers), falling back to ``os.cpu_count()``.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Sequence, TypeVar

import torch

from .base import ClientExecutor

__all__ = [
    "ThreadsExecutor",
]

T = TypeVar("T")


def num_available_cpus() -> int:
    """The number of CPUs this process may actually use."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux platforms
        return os.cpu_count() or 1


class ThreadsExecutor(ClientExecutor):
    """Runs the client tasks on a thread pool, preserving the sampled order.

    Parameters
    ----------
    num_workers : int
        The maximum number of client tasks running concurrently;
        ``0`` means auto (the usable CPU count).
    torch_num_threads : int, optional
        The intra-op thread budget to set while active; ``None`` means
        auto (``usable cores // num_workers``, at least 1). Passing an
        explicit value also disables the automatic budgeting.
    """

    def __init__(self, num_workers: int = 0, torch_num_threads: Optional[int] = None) -> None:
        self.num_workers = num_workers if num_workers > 0 else num_available_cpus()
        self.torch_num_threads = torch_num_threads
        self._previous_num_threads: Optional[int] = None

    def __enter__(self) -> "ThreadsExecutor":
        self._previous_num_threads = torch.get_num_threads()
        if self.torch_num_threads is not None:
            torch.set_num_threads(max(1, self.torch_num_threads))
        else:
            torch.set_num_threads(max(1, num_available_cpus() // self.num_workers))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._previous_num_threads is not None:
            torch.set_num_threads(self._previous_num_threads)
            self._previous_num_threads = None

    def map(
        self,
        fn: Callable[[object], T],
        items: Sequence[object],
        on_complete: Optional[Callable[[T, int], None]] = None,
    ) -> List[T]:
        pool = ThreadPoolExecutor(max_workers=self.num_workers)
        try:
            futures = {pool.submit(fn, item): index for index, item in enumerate(items)}
            results: List[T] = [None] * len(items)  # type: ignore[list-item]
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    # fail fast, matching the sequential semantics; already-
                    # running tasks run to completion (documented physical
                    # fact), queued ones are cancelled on a best-effort basis.
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(f"client task failed (index {index}) in the threads executor") from exc
                if on_complete is not None:
                    on_complete(results[index], index)
            return results
        finally:
            pool.shutdown(wait=False)  # no-op on the happy path
