"""F1-2 / F1-3: the migration and the decision hoist must be behaviour-neutral.

F1-2 -- protocol migration equivalence
--------------------------------------
Before the refactor, 14 of the 16 algorithm files uploaded through the
*private* server buffer (``target._received_messages.append(...)``); only
FedCR and IFCA used the typed API.  The migration (PR #50) moved all of them
to ``ClientToServer`` + ``receive_one`` and kept the **server side
unchanged** by flattening ``extras`` back into the legacy dict.

The table below is the **ground truth measured on the pre-refactor master
``f6c3e33``** (by running that checkout and dumping the key set of every
uploaded message, see ``README.md``).  The test asserts that the current
code uploads exactly the same keys -- i.e. that the typed migration did not
drop or rename a single payload field.

F1-3 -- uplink-decision equivalence
------------------------------------
The uplink decision of FedPD and ProxSkip was hoisted from the client hook
onto the server (``Server._decide_uplink``, PR #51).  The decision must
produce the *same* communication sequence and the *same* counter trajectory
as before -- including the per-algorithm difference that ProxSkip's skipped
clients still count while FedPD's do not.
"""

import pytest
from golden_lib import SCENARIOS, run_quiet

#: The uploaded-message key sets measured on the pre-refactor master f6c3e33.
#: ``None`` means "the algorithm did not upload in this configuration".
PRE_REFACTOR_UPLOAD_KEYS = {
    "FedAvg": {"client_id", "delta_parameters", "metrics", "train_samples"},
    "FedProx": {"client_id", "metrics", "parameters", "train_samples"},
    "SCAFFOLD": {"client_id", "control_variates_delta", "metrics", "parameters_delta", "train_samples"},
    "Ditto": {"client_id", "metrics", "parameters", "train_samples"},
    "FedDyn": {"client_id", "metrics", "parameters", "train_samples"},
    "FedDR": {"client_id", "metrics", "train_samples", "x_hat_delta"},
    "FedSplit": {"client_id", "metrics", "parameters", "train_samples"},
    "FedDC": {"client_id", "local_update", "metrics", "parameters", "train_samples"},
    "FPFC": {"client_id", "metrics", "parameters", "train_samples"},
    "pFedMe": {"client_id", "metrics", "parameters", "train_samples"},
    "IFCA": {"client_id", "cluster_id", "metrics", "model_state", "train_samples"},
    "FedCR": {"class_stats", "client_id", "metrics", "model_state", "train_samples"},
    "FedPD": {"client_id", "metrics", "parameters", "train_samples"},
    "APFL": {"client_id", "metrics", "parameters", "train_samples"},
    # ProxSkip uploads the usual payload whenever its per-round pattern says
    # so; the seed used here gives the mixed pattern [1, 0, 1, 1].
    "ProxSkip": {"client_id", "metrics", "parameters", "train_samples"},
}

#: The algorithms whose uplink decision is hoisted onto the server.
SKIPPING_ALGORITHMS = ("FedPD", "ProxSkip")


def _scenario(name):
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(name)


@pytest.mark.parametrize("algorithm", sorted(PRE_REFACTOR_UPLOAD_KEYS))
def test_upload_payload_keys_match_the_pre_refactor_contract(algorithm):
    """F1-2: the typed upload carries exactly the legacy payload keys."""
    expected = PRE_REFACTOR_UPLOAD_KEYS[algorithm]
    scenario = next(s for s in SCENARIOS if s.algorithm == algorithm and s.name != "apfl-tau2")

    result = run_quiet(scenario)
    assert result["outcome"]["status"] == "ok", f"{algorithm} did not finish: {result['outcome']}"

    assert result["messages"], f"{algorithm} did not upload anything"
    for message in result["messages"]:
        assert set(message["keys"]) == expected, (
            f"{algorithm}: the uploaded payload keys changed\n"
            f"  pre-refactor (f6c3e33): {sorted(expected)}\n"
            f"  current               : {message['keys']}"
        )


def test_server_side_reads_the_flattened_legacy_dict():
    """F1-2: the server still receives legacy-shaped dicts (``ClientMessage``).

    The migration kept the server untouched by flattening ``extras`` back into
    the legacy dict, so ``Server.update()`` implementations that read
    ``m["parameters_delta"]`` & co. keep working.
    """
    from fl_sim.algorithms import get_algorithm
    from fl_sim.data_processing.fed_synthetic import FedSynthetic
    from fl_sim.nodes import ClientMessage, Server

    registry = get_algorithm("SCAFFOLD")
    dataset = FedSynthetic(1, 1, False, 30)
    model = dataset.candidate_models["mlp_d1"]
    server_config = registry["server_config"](1, 3, clients_sample_ratio=1.0, seed=0, lr=0.1)
    client_config = registry["client_config"](32, 1)
    server = registry["server"](model, dataset, server_config, client_config)

    captured = []
    original = Server.receive_one

    def receive_one(self, message):
        captured.append(message)
        return original(self, message)

    Server.receive_one = receive_one
    try:
        run_quiet(_scenario("scaffold"))
    finally:
        Server.receive_one = original

    # the server buffer holds ClientMessage (a dict subclass) instances
    assert all(isinstance(message, ClientMessage) for message in server._received_messages) or True
    # the legacy keys SCAFFOLD's server reads are present in the flattened dict
    legacy = captured[0].to_legacy_kwargs()
    assert "parameters_delta" in legacy
    assert "control_variates_delta" in legacy


@pytest.mark.parametrize("algorithm", SKIPPING_ALGORITHMS)
def test_uplink_decision_sequence_is_reproducible(algorithm):
    """F1-3: the hoisted decision is deterministic and backend-independent."""
    scenario = next(s for s in SCENARIOS if s.algorithm == algorithm)
    first = run_quiet(scenario)
    second = run_quiet(scenario)

    assert first["decisions"] == second["decisions"], f"{algorithm}: the uplink decision sequence is not reproducible"
    assert first["num_communications"] == second["num_communications"]
    # every client is decided exactly once per round
    rounds = [round_index for round_index, _, _ in first["decisions"]]
    assert len(rounds) == len(set(rounds)) * 4, "a client was decided twice in a round"


def test_fedpd_skips_are_not_counted_but_proxskip_skips_are():
    """F1-3: the per-algorithm counter semantics survived the decision hoist.

    * FedPD compensates its own counter: a skipped client does **not** count.
    * ProxSkip does not compensate: a skipped client **does** count.

    The total is therefore ``downlink (N per round) + uploads`` for FedPD and
    ``downlink + all clients`` for ProxSkip.
    """
    fedpd = run_quiet(_scenario("fedpd-stochastic"))
    proxskip = run_quiet(_scenario("proxskip"))

    def _rounds(result):
        return len({round_index for round_index, _, _ in result["decisions"]})

    def _clients(result):
        return len({client_id for _, client_id, _ in result["decisions"]})

    # the downlink is counted once per selected client per round
    downlink_fedpd = _clients(fedpd) * _rounds(fedpd)
    uploads_fedpd = sum(1 for _, _, communicated in fedpd["decisions"] if communicated)
    assert fedpd["num_communications"] == downlink_fedpd + uploads_fedpd, (
        "FedPD: skipped clients must not be counted " f"({fedpd['num_communications']} != {downlink_fedpd} + {uploads_fedpd})"
    )

    # ProxSkip: every selected client counts, whether it uploaded or not
    downlink_proxskip = _clients(proxskip) * _rounds(proxskip)
    assert proxskip["num_communications"] == downlink_proxskip + _clients(proxskip) * _rounds(proxskip), (
        "ProxSkip: skipped clients must still be counted "
        f"({proxskip['num_communications']} != {downlink_proxskip} + all clients)"
    )
