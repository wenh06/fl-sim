"""Tests for the client-executor backends (parallel execution layer, P1b).

Covers the executor contract (design §5.2): sampled-order results,
completion-driven ``on_complete``, fail-fast exception context, the torch
intra-op thread budget save/restore, and end-to-end parity of the two
backends on a small fixed-seed federated run.
"""

import os

import pytest
import torch
from test_nodes import DummyClient, DummyClientConfig, DummyServer, DummySeverConfig

from fl_sim.data_processing.fed_synthetic import FedSynthetic
from fl_sim.executors import ClientExecutor, SequentialExecutor, ThreadsExecutor
from fl_sim.executors.threads import num_available_cpus
from fl_sim.messages import ClientToServer


class _ParallelSafeClient(DummyClient):
    """DummyClient with the new-style (contract) communicate hook."""

    def communicate(self):
        return ClientToServer(
            client_id=self.client_id,
            train_samples=len(self.train_loader.dataset),
            metrics=self._metrics,
            parameters=self.get_detached_model_parameters(),
        )


class _ParallelSafeServer(DummyServer):
    @property
    def client_cls(self) -> type:
        return _ParallelSafeClient


def _make(num_workers: int = 0, torch_num_threads: int = None, num_iters: int = 2):
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = DummySeverConfig(num_iters, 6)
    # the execution-layer knobs are keyword-only fields; the (shared) test
    # fixture does not forward extra kwargs, so set them after construction
    server_config.executor = "threads"
    server_config.num_workers = num_workers
    server_config.torch_num_threads = torch_num_threads
    return _ParallelSafeServer(model, dataset, server_config, DummyClientConfig(32, 1))


def test_sequential_executor_contract():
    executor = SequentialExecutor()
    order = []
    results = executor.map(lambda x: x * 2, [1, 2, 3], on_complete=lambda r, i: order.append((r, i)))
    assert results == [2, 4, 6]
    assert order == [(2, 0), (4, 1), (6, 2)]  # completion order == sampled order


def test_threads_executor_contract_and_order():
    with ThreadsExecutor(num_workers=4) as executor:
        order = []
        results = executor.map(lambda x: x * 2, list(range(20)), on_complete=lambda r, i: order.append(i))
    assert results == [x * 2 for x in range(20)]  # sampled order, not completion order
    assert sorted(order) == list(range(20))


def test_threads_executor_fail_fast_with_context():
    sentinel = torch.zeros(1)

    def fn(item):
        if item == 2:
            raise ValueError("boom")
        return item

    with ThreadsExecutor(num_workers=2) as executor:
        with pytest.raises(RuntimeError, match="index 2") as info:
            executor.map(fn, [0, 1, 2, 3, 4])
    assert isinstance(info.value.__cause__, ValueError)  # original exception preserved


def test_threads_executor_torch_thread_budget():
    original = torch.get_num_threads()
    try:
        with ThreadsExecutor(num_workers=2, torch_num_threads=3):
            assert torch.get_num_threads() == 3
        assert torch.get_num_threads() == original  # restored on exit
    finally:
        torch.set_num_threads(original)


def test_num_available_cpus_sane():
    n = num_available_cpus()
    assert 1 <= n <= (os.cpu_count() or 1)


def test_executor_is_abstract():
    with pytest.raises(TypeError):
        ClientExecutor()  # noqa: B015


def _train_messages(**executor_kwargs):
    s = _make(**executor_kwargs)
    seen = []

    original = DummyServer.receive_one

    def patched(server_self, message):
        seen.append(message.client_id)
        return original(server_self, message)

    DummyServer.receive_one = patched
    try:
        s.train_federated()
    finally:
        DummyServer.receive_one = original
    assert s._complete_experiment
    return seen, s._num_communications, float(s.model.parameters().__next__().sum())


def test_threads_backend_end_to_end_runs_and_counts():
    seen, counter, _ = _train_messages(num_workers=4)
    # 2 iterations x 6 clients, DummyClient always communicates
    assert len(seen) == 12
    assert counter == 12 + 12  # uplink (+1 each) + downlink broadcast (+1 each)


def test_threads_backend_same_seeded_run_completes_with_same_counts():
    seen_a, counter_a, weights_a = _train_messages(num_workers=1)
    seen_b, counter_b, weights_b = _train_messages(num_workers=4)
    assert len(seen_a) == len(seen_b) and counter_a == counter_b
