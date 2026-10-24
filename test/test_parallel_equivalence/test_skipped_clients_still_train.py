"""F1-4 (review item M3): a client that skips the upload still trains and logs.

Why this test exists
--------------------
The first version of the task contract had an early ``return`` for clients
that do not communicate:

    def _client_task(self, client, evaluate):
        if not client._communicate_this_round:
            return TaskResult(reply=None, ...)      # <-- skipped training too!
        ...

That would have been a **behaviour change**: in the original sequential loop
``Server._local_step`` evaluated, trained and logged for *every* selected
client, and only the *upload* could be skipped (design §5.3 / N2 of the
review).  With the early return, the personalized models of FedPD and
ProxSkip would have stopped evolving, and the skipped clients' metrics would
have disappeared from the log -- a regression that does not necessarily show
up in the accuracy curve.

The contract was corrected to "always evaluate, always train, only the reply
is conditional", and this test is the direct guard for that correction.
"""

from dataclasses import replace

from golden_lib import SCENARIOS, run_quiet


def _scenario(name):
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(name)


def test_skipped_client_still_trains_and_still_logs_metrics():
    """A client whose upload is skipped still trains and still logs metrics."""
    # three rounds -> rounds 1 and 2 are evaluation rounds (the framework
    # evaluates on the clients only when ``n_iter > 0``)
    scenario = replace(
        _scenario("fedpd-stochastic"),
        name="fedpd-stochastic-m3",
        num_iters=3,
    )
    result = run_quiet(scenario)
    assert result["outcome"]["status"] == "ok", result["outcome"]

    skipped = [(round_index, client_id) for round_index, client_id, communicated in result["decisions"] if not communicated]
    assert skipped, "the scenario did not skip any upload -- it cannot prove anything"

    # 1) every skipped client still ran its local step, and its model changed
    updates = {(round_index, client_id): (before, after) for round_index, client_id, before, after in result["updates"]}
    for round_index, client_id in skipped:
        assert (round_index, client_id) in updates, f"client {client_id} did not run its local step in round {round_index}"
        before, after = updates[(round_index, client_id)]
        assert before != after, (
            f"client {client_id} skipped the upload but its model did NOT change "
            f"in round {round_index} -- the personalized model stopped evolving"
        )

    # 2) on the evaluation rounds, the skipped clients still logged their
    #    train metrics (the metric log entry carries the round as its step)
    logged = {(client_id, step) for client_id, part, _digest, step in result["metrics_log"] if part == "train"}
    evaluation_rounds = {round_index for round_index, _ in skipped if round_index > 0}
    assert evaluation_rounds, "no evaluation round with a skipped upload"
    for round_index, client_id in skipped:
        if round_index in evaluation_rounds:
            assert (client_id, round_index) in logged, (
                f"client {client_id} skipped the upload in round {round_index} " f"but did not log its train metrics"
            )


def test_communicating_client_is_treated_the_same_way():
    """The counterpart: a client that uploads is of course trained and logged."""
    scenario = replace(_scenario("fedpd-stochastic"), name="fedpd-stochastic-m3b", num_iters=3)
    result = run_quiet(scenario)

    communicating = [(round_index, client_id) for round_index, client_id, communicated in result["decisions"] if communicated]
    assert communicating, "no upload in the scenario"

    updates = {(round_index, client_id): (before, after) for round_index, client_id, before, after in result["updates"]}
    logged = {(client_id, step) for client_id, part, _digest, step in result["metrics_log"] if part == "train"}
    for round_index, client_id in communicating:
        before, after = updates[(round_index, client_id)]
        assert before != after
        if round_index > 0:
            assert (client_id, round_index) in logged


def test_skipped_client_does_not_reach_the_server():
    """The other half of the contract: a skipped client uploads nothing."""
    scenario = replace(_scenario("fedpd-stochastic"), name="fedpd-stochastic-m3c", num_iters=3)
    result = run_quiet(scenario)

    uploaded = {message["client_id"] for message in result["messages"]}
    skipped_ids = {client_id for _round, client_id, communicated in result["decisions"] if not communicated}
    # a client may upload in one round and skip in another; the invariant is
    # about the *count*: the number of uploads equals the number of
    # "communicating" decisions
    assert len(result["messages"]) == sum(1 for _, _, communicated in result["decisions"] if communicated)
    assert uploaded <= set(range(4))
    assert skipped_ids <= set(range(4))
