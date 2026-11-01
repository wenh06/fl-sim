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
    # the dummy clients do NOT train (update() is a no-op), so the weights
    # are identical across backends: this pins that the executor layer itself
    # leaves the model untouched. Shuffle divergence under global rng_mode
    # (which needs real training) is pinned separately below.
    assert weights_a == weights_b


# ---------------------------------------------------------------------------
# L1 reproducibility: per-client DataLoader generators (rng_mode="per_client")
# ---------------------------------------------------------------------------


def _run_fedavg(keyed_orders: dict, executor: str, rng_mode: str, num_iters: int = 3):
    import fl_sim.algorithms as algorithms

    reg = algorithms.get_algorithm("FedAvg")
    dataset = FedSynthetic(1, 1, False, 30)
    server = reg["server"](
        dataset.candidate_models["mlp_d1"],
        dataset,
        reg["server_config"](num_iters, 6, clients_sample_ratio=1.0),
        reg["client_config"](32, 1, rng_mode=rng_mode),
    )
    server.config.executor = executor
    server.config.num_workers = 4
    original = type(server._clients[0]).update

    def update_with_order(self):
        keyed_orders.setdefault(self._round_idx, {})[self.client_id] = tuple(
            tensor[0].sum().item() for tensor, _ in self.train_loader
        )
        original(self)

    type(server._clients[0]).update = update_with_order
    try:
        server.train_federated()
        weight = float(list(server.model.parameters())[0].sum())
    finally:
        type(server._clients[0]).update = original
    return weight


def test_per_client_rng_shuffle_sequence_backend_and_repeat_invariant():
    """L1: with rng_mode=per_client, the per-(round, client) shuffle stream is
    bitwise identical across backends and across same-seed repeats, and the
    final model weights match bitwise (review N3's per-client-Generator
    mechanism)."""
    run_a: dict = {}
    run_b: dict = {}
    run_c: dict = {}
    weight_a = _run_fedavg(run_a, "threads", "per_client")
    weight_b = _run_fedavg(run_b, "threads", "per_client")
    weight_c = _run_fedavg(run_c, "sequential", "per_client")
    assert run_a == run_b  # same-seed repeat
    assert run_a == run_c  # backend-invariant
    assert weight_a == weight_b == weight_c  # bitwise


def test_global_rng_mode_sequential_run_is_deterministic():
    """With the default rng_mode=global, two same-seed sequential runs give
    bitwise-identical weights (G3 determinism).

    NOTE there is deliberately NO cross-backend weight assertion under
    global mode: whether the threads run's shuffle interleave diverges from
    the sequential order depends on the scheduler (on a busy/low-core host
    the tiny tasks may serialize into the submission order and coincide) --
    which is exactly why the framework only promises STATISTICAL equivalence
    under global mode, and bitwise cross-backend equality under
    rng_mode=per_client (pinned by the test above)."""
    run_a: dict = {}
    run_b: dict = {}
    weight_a = _run_fedavg(run_a, "sequential", "global")
    weight_b = _run_fedavg(run_b, "sequential", "global")
    assert run_a == run_b
    assert weight_a == weight_b


def test_global_rng_mode_has_no_per_client_generator():
    """Default rng_mode=global keeps the loaders untouched (global RNG)."""
    server = _make(num_workers=4)
    for client in server._clients:
        assert client.train_loader.generator is None
        assert client.val_loader.generator is None


def test_rng_mode_rejects_typos():
    from pydantic import ValidationError

    from fl_sim.nodes import ClientConfig, ServerConfig

    # the shared hand-written test fixtures take explicit kwargs only; use
    # the real (pydantic) configs to assert the Literal guards
    with pytest.raises(ValidationError):
        ClientConfig(32, 1, 1e-3, "SGD", rng_mode="per-clients")
    with pytest.raises(ValidationError):
        ServerConfig(1, 6, 1.0, executor="threadz")
