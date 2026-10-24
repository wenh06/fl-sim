"""Shared harness for the F1 equivalence / golden-regression tests.

This module is the single source of truth for *what* is compared. It runs a
federated scenario and reduces it to a canonical, JSON-serializable digest:

* ``messages``      -- one entry per upload the server accepted, **in the
                       order the server consumed them** (the sampled order).
                       This is the aggregation input, so a digest over it
                       pins the aggregation arithmetic.
* ``decisions``     -- ``[round, client_id, communicates?]`` per client per
                       round, taken from the server's uplink decision (the
                       decision that was hoisted onto the server in P1a-2).
* ``metrics_log``   -- ``[client_id, part, metrics_digest, step]`` for every
                       logged metric, so that "a client that skipped the
                       upload still evaluated and still logged" is observable.
* ``updates``       -- ``[round, client_id, digest_before, digest_after]``
                       around each client's local step, so that "a client
                       that skipped the upload still trained" is observable.
* ``final_weights`` -- digest of the server's final model state.
* ``outcome``       -- ``ok`` or the exception the run raised (so a known
                       algorithm defect can be pinned as a baseline too).

Everything is seeded right before ``train_federated()`` so that a scenario is
fully determined by its specification, independent of what the dataset and
the config constructors consumed from the global RNG.

The harness only uses APIs that exist both **before** and **after** the
parallel-execution refactor, so it can also be pointed at a pre-refactor
checkout to *prove* that the recorded baseline is unchanged by the refactor
(see ``make_golden.py --repo``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

__all__ = [
    "Scenario",
    "SCENARIOS",
    "run_scenario",
    "digest_of_tensors",
]


# ---------------------------------------------------------------------------
# digest helpers
# ---------------------------------------------------------------------------


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:16]


def digest_of_tensors(value: Any) -> str:
    """A stable digest of tensors / sequences / mappings of tensors.

    Floats are digested through ``repr`` (exact round-trip), tensors through
    their raw bytes, so the digest is bitwise sensitive.
    """
    if value is None:
        return "none"
    if isinstance(value, torch.Tensor):
        array = value.detach().to(torch.float64).contiguous().cpu().numpy()
        return "t:" + _digest_bytes(array.tobytes())
    if isinstance(value, (list, tuple)):
        h = hashlib.sha256()
        for item in value:
            h.update(digest_of_tensors(item).encode())
        return "l:" + h.hexdigest()[:16]
    if isinstance(value, dict):
        h = hashlib.sha256()
        for key in sorted(value, key=repr):
            h.update(repr(key).encode())
            h.update(digest_of_tensors(value[key]).encode())
        return "d:" + h.hexdigest()[:16]
    if isinstance(value, float):
        return "f:" + repr(value)
    if isinstance(value, int):
        return "i:" + str(value)
    return "r:" + repr(value)


def _metrics_digest(metrics: Optional[Dict[str, Any]]) -> str:
    if metrics is None:
        return "none"
    return digest_of_tensors({k: metrics[k] for k in sorted(metrics)})


#: The round the server is currently in.  Tracked by the harness itself (not
#: by the client) so that it works on the pre-refactor code too, where the
#: client has no ``_round_idx`` snapshot.
_CURRENT_ROUND = [None]


def _state_digest(state_dict: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict):
        h.update(key.encode())
        h.update(digest_of_tensors(state_dict[key]).encode())
    return h.hexdigest()[:16]


def _message_digest(message: Any) -> Dict[str, Any]:
    """Digest one upload, tolerating both the typed and the legacy shape."""
    if hasattr(message, "to_legacy_kwargs"):
        payload = message.to_legacy_kwargs()
    elif isinstance(message, dict):
        payload = dict(message)
    else:  # pragma: no cover - defensive
        payload = {"raw": repr(message)}

    entry: Dict[str, Any] = {
        "client_id": payload.get("client_id"),
        "train_samples": payload.get("train_samples"),
        "keys": sorted(payload.keys()),
        "metrics": _metrics_digest(payload.get("metrics")),
    }
    for key, value in payload.items():
        if key in ("client_id", "train_samples", "metrics"):
            continue
        entry[key] = digest_of_tensors(value)
    return entry


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """One reproducible federated run reduced to a digest."""

    name: str
    algorithm: str
    num_iters: int = 2
    num_clients: int = 4
    epochs: int = 1
    batch_size: int = 32
    seed: int = 0
    server_kwargs: Dict[str, Any] = field(default_factory=dict)
    client_kwargs: Dict[str, Any] = field(default_factory=dict)
    #: ``True`` when the scenario is known to raise (a pinned algorithm defect)
    expect_raise: bool = False


SCENARIOS: List[Scenario] = [
    # --- plain upload path -------------------------------------------------
    Scenario(name="fedavg", algorithm="FedAvg"),
    Scenario(name="fedopt", algorithm="FedOpt"),
    Scenario(name="fedadagrad", algorithm="FedAdagrad"),
    Scenario(name="fedyogi", algorithm="FedYogi"),
    Scenario(name="fedadam", algorithm="FedAdam"),
    Scenario(name="fedprox", algorithm="FedProx", client_kwargs={"mu": 0.01}),
    Scenario(name="scaffold", algorithm="SCAFFOLD", server_kwargs={"lr": 0.1}),
    Scenario(name="ditto", algorithm="Ditto"),
    Scenario(name="feddyn", algorithm="FedDyn"),
    Scenario(name="feddr", algorithm="FedDR"),
    Scenario(name="fedsplit", algorithm="FedSplit"),
    Scenario(name="feddc", algorithm="FedDC"),
    Scenario(name="fpfc", algorithm="FPFC"),
    Scenario(name="pfedme", algorithm="pFedMe"),
    Scenario(name="pfedmac", algorithm="pFedMac"),
    # --- typed uploads already before the refactor --------------------------
    Scenario(name="ifca", algorithm="IFCA", server_kwargs={"num_clusters": 2}),
    Scenario(name="fedcr", algorithm="FedCR"),
    # --- uplink skipping (the decision hoisted in P1a-2) -------------------
    # ProxSkip's pattern is per ROUND (all clients of a round share it), so
    # the seed is chosen such that the pattern is mixed over the rounds
    # ([1, 0, 1, 1]) -- otherwise the scenario would only ever exercise
    # "everybody uploads" or "nobody uploads".
    Scenario(name="proxskip", algorithm="ProxSkip", num_iters=4, seed=1, server_kwargs={"p": 0.5}),
    Scenario(name="fedpd-periodic", algorithm="FedPD", server_kwargs={"p": 0.5, "stochastic": False}),
    Scenario(name="fedpd-stochastic", algorithm="FedPD", server_kwargs={"p": 0.5, "stochastic": True}),
    # --- conditional downlink ----------------------------------------------
    Scenario(name="apfl-tau1", algorithm="APFL", server_kwargs={"tau": 1}),
    # APFL with tau>1 used to hit a pre-existing defect (server/client
    # sync-counter misalignment -> the server averaged messages without
    # "parameters" and crashed with KeyError). FIXED by aligning the server's
    # aggregation with the clients' period-end uploads ((n_iter + 1) % tau
    # == 0); the scenario now must train to completion -- the baseline pins
    # the fixed behavior, so a regression back to the crash fails loudly.
    Scenario(name="apfl-tau2", algorithm="APFL", server_kwargs={"tau": 2}),
]


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------


def run_scenario(scenario: Scenario) -> Dict[str, Any]:
    """Run one scenario and return its canonical digest."""
    from fl_sim.algorithms import get_algorithm
    from fl_sim.data_processing.fed_synthetic import FedSynthetic
    from fl_sim.nodes import Client, Server

    registry = get_algorithm(scenario.algorithm)
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    # NOTE: the server config seeds the global RNGs in its constructor
    # (``ServerConfig.__init__`` -> ``set_seed(self.seed)``). Everything that
    # consumes randomness afterwards -- the model init, the client setup
    # (ProxSkip draws its communication pattern THERE) and the training --
    # therefore starts from a known state. Do NOT re-seed after this point:
    # a re-seed would land *between* the client setup and the training for
    # the eagerly-setup case and *after* the setup for the lazy case, i.e. the
    # scenario would stop being reproducible.
    #
    # The order below is load-bearing, not cosmetic: ProxSkip's communication
    # pattern is drawn from the GLOBAL numpy RNG inside ``_setup_clients``,
    # and ``set_seed`` is called by BOTH the dataset constructor and the
    # server-config constructor.  Swapping those two (or letting any numpy
    # draw happen in between) changes the pattern table -- and with it the
    # whole scenario digest -- so ALL golden scenarios would mismatch at once.
    # See appendix C of ``tmp/parallel-execution-review.md``.
    server_config = registry["server_config"](
        scenario.num_iters,
        scenario.num_clients,
        clients_sample_ratio=1.0,
        seed=scenario.seed,
        **scenario.server_kwargs,
    )
    client_config = registry["client_config"](
        scenario.batch_size,
        scenario.epochs,
        **scenario.client_kwargs,
    )
    server = registry["server"](model, dataset, server_config, client_config)

    messages: List[Dict[str, Any]] = []
    decisions: List[List[Any]] = []
    uploads_per_round: List[List[Any]] = []
    metrics_log: List[List[Any]] = []
    updates: List[List[Any]] = []

    original_receive = Server.receive_one
    # ``_accept_reply`` only exists after the refactor; on the pre-refactor
    # code the upload decision is not observable from the outside, so the
    # decision list stays empty there.  ``uploads_per_round`` IS observable
    # on both sides (it is derived from the server's round-end buffer), and
    # it is the field the cross-version comparison uses.
    original_accept = getattr(Server, "_accept_reply", None)
    original_server_update = Server._update
    # the per-round hook differs between the two versions: the refactored
    # code broadcasts in ``_broadcast``, the pre-refactor one does the local
    # step in ``_local_step``.  Both are called once per client per round.
    round_hook_name = "_broadcast" if hasattr(Server, "_broadcast") else "_local_step"
    original_round_hook = getattr(Server, round_hook_name)
    original_client_update = Client._update
    logger = server._logger_manager
    original_log = logger.log_metrics

    def receive_one(self, message):
        messages.append(_message_digest(message))
        return original_receive(self, message)

    def accept_reply(self, client, reply):
        # NOTE: the effective per-round uplink decision is read HERE, not from
        # ``_decide_uplink``: the algorithm overrides call ``super()`` first
        # and set the flag afterwards, so patching the hook would observe the
        # previous round's value. A reply of ``None`` *is* the decision.
        decisions.append([_CURRENT_ROUND[0], client.client_id, reply is not None])
        return original_accept(self, client, reply)

    def round_hook(self, *args, **kwargs):
        _CURRENT_ROUND[0] = self.n_iter
        return original_round_hook(self, *args, **kwargs)

    def server_update(self):
        # the server's buffer still holds this round's uploads here
        uploads_per_round.append([self.n_iter, sorted(int(m["client_id"]) for m in self._received_messages)])
        return original_server_update(self)

    def client_update(self):
        before = _state_digest(self.model.state_dict())
        original_client_update(self)
        after = _state_digest(self.model.state_dict())
        updates.append([_CURRENT_ROUND[0], self.client_id, before, after])

    def log_metrics(*args, **kwargs):
        client_id = args[0] if len(args) > 0 else kwargs.get("client_id")
        metrics = args[1] if len(args) > 1 else kwargs.get("metrics")
        step = kwargs.get("step", args[2] if len(args) > 2 else None)
        part = kwargs.get("part", args[4] if len(args) > 4 else "val")
        metrics_log.append([client_id, part, _metrics_digest(metrics), step])
        return original_log(*args, **kwargs)

    Server.receive_one = receive_one
    Server._update = server_update
    setattr(Server, round_hook_name, round_hook)
    if original_accept is not None:
        Server._accept_reply = accept_reply
    Client._update = client_update
    logger.log_metrics = log_metrics

    try:
        server.train_federated()
        outcome: Dict[str, Any] = {"status": "ok"}
    except Exception as exc:  # noqa: BLE001 - a pinned defect must be recordable
        outcome = {
            "status": "raised",
            "exc_type": type(exc).__name__,
            "exc_msg": str(exc),
        }
    finally:
        Server.receive_one = original_receive
        Server._update = original_server_update
        setattr(Server, round_hook_name, original_round_hook)
        if original_accept is not None:
            Server._accept_reply = original_accept
        Client._update = original_client_update
        logger.log_metrics = original_log

    return {
        "scenario": scenario.name,
        "algorithm": scenario.algorithm,
        "expect_raise": scenario.expect_raise,
        "outcome": outcome,
        "messages": messages,
        "decisions": decisions,
        "uploads_per_round": uploads_per_round,
        "metrics_log": metrics_log,
        "updates": updates,
        "num_communications": getattr(server, "_num_communications", None),
        "final_weights": _state_digest(server.model.state_dict()),
    }


def run_all(scenarios: Optional[List[Scenario]] = None) -> Dict[str, Any]:
    """Run every scenario and return the baseline structure."""
    results = {}
    for scenario in scenarios if scenarios is not None else SCENARIOS:
        results[scenario.name] = run_scenario(scenario)
    return results


def run_quiet(scenario: Scenario) -> Dict[str, Any]:
    """Run a scenario with the framework's console logging silenced.

    The framework logs through ``tqdm.write`` (i.e. to the *current*
    ``sys.stdout``), so redirecting the streams around the run is enough to
    keep the pytest output readable.
    """
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        return run_scenario(scenario)
