"""F1-1: golden regression over the parallel-execution layer.

What it guards
--------------
The two invariants that the parallel-execution refactor promised but that
were, until now, only *described* in the PR descriptions:

* **G2** -- the algorithms' observable behaviour is unchanged by the
  migration (legacy ``_received_messages.append(...)`` -> typed
  ``ClientToServer`` / ``receive_one``);
* **G3** -- the ``sequential`` backend is bitwise identical to the
  pre-refactor code.

How it works
------------
``golden_lib`` reduces a federated run to a canonical digest (the uploaded
messages in the order the server consumed them, the per-round uplink
decisions, the metric log, the per-client model changes and the final
server weights).  The digest of the current code is compared against the
committed baseline in ``golden/baseline.json``.

A difference means: *something in the observable behaviour changed*.  The
baseline is then updated deliberately (and reviewed), never silently.

Regenerating / proving the baseline
-----------------------------------
    # regenerate the baseline from the current code
    python test/test_parallel_equivalence/make_golden.py --out test/test_parallel_equivalence/golden/baseline.json

    # prove the baseline also holds on the pre-refactor master (f6c3e33)
    git worktree add /tmp/flsim-pre f6c3e33
    python test/test_parallel_equivalence/make_golden.py --repo /tmp/flsim-pre --out /tmp/pre.json
    python tmp/f1-tests/make_golden.py --compare /tmp/pre.json
"""

import json
from pathlib import Path

import pytest
from golden_lib import SCENARIOS, machine_fingerprint, run_quiet

BASELINE_PATH = Path(__file__).resolve().parent / "golden" / "baseline.json"


def _load_baseline():
    if not BASELINE_PATH.exists():
        pytest.fail(
            f"the golden baseline is missing: {BASELINE_PATH}\n"
            "generate it with: python tmp/f1-tests/make_golden.py --out "
            f"{BASELINE_PATH}"
        )
    with open(BASELINE_PATH, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("scenarios", payload)


def _baseline_fingerprint():
    with open(BASELINE_PATH, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("fingerprint")


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_scenario_matches_golden_baseline(scenario):
    """Each scenario reproduces its recorded digest, bit for bit.

    Two tiers, by machine fingerprint (float digests are bitwise-stable only
    on the CPU/build the baseline was generated on):

    - **fingerprint match** (the baseline machine, i.e. local development):
      the FULL digest comparison -- any float change fails;
    - **foreign machine** (e.g. a CI runner): STRUCTURAL comparison only --
      the machine-invariant behaviour (outcome, participation decisions,
      upload structure, counters, log structure) is still guarded, while the
      float digests are not compared.
    """
    baseline = _load_baseline()
    assert scenario.name in baseline, f"{scenario.name} is missing from the baseline"

    actual = run_quiet(scenario)
    expected = baseline[scenario.name]

    if _baseline_fingerprint() != machine_fingerprint():
        _assert_structural_equivalence(scenario.name, actual, expected)
        return

    if actual == expected:
        return

    # report the FIRST differing field rather than dumping two big dicts
    for key in ("outcome", "num_communications", "final_weights"):
        if actual.get(key) != expected.get(key):
            pytest.fail(
                f"{scenario.name}: '{key}' changed\n"
                f"  baseline: {expected.get(key)}\n"
                f"  current : {actual.get(key)}\n"
                f"  (a behaviour change: regenerate the baseline deliberately, "
                f"and check that the change is intended)"
            )
    for key in ("messages", "decisions", "metrics_log", "updates"):
        if actual.get(key) != expected.get(key):
            for index, (left, right) in enumerate(zip(actual.get(key, []), expected.get(key, []))):
                if left != right:
                    pytest.fail(f"{scenario.name}: '{key}[{index}]' changed\n" f"  baseline: {right}\n" f"  current : {left}")
            pytest.fail(
                f"{scenario.name}: '{key}' changed in length " f"({len(actual.get(key, []))} vs {len(expected.get(key, []))})"
            )
    pytest.fail(f"{scenario.name}: the digest changed but no field was identified")


def _assert_structural_equivalence(name: str, actual: dict, expected: dict) -> None:
    """Machine-invariant comparison for a foreign-machine baseline.

    Guards everything that does NOT depend on the CPU's floating-point
    reduction order: the outcome, the communication counters, the per-round
    uplink decisions and upload counts, the structure of every uploaded
    message (who, how many samples, which payload keys), and the structure
    of the metric log (who, which part, which step). Float digests
    (payload contents, model states) are deliberately not compared here.
    """
    for key in ("outcome", "num_communications", "decisions", "uploads_per_round"):
        assert actual.get(key) == expected.get(
            key
        ), f"{name}: '{key}' changed\n  baseline: {expected.get(key)}\n  current : {actual.get(key)}"
    actual_msgs, expected_msgs = actual.get("messages", []), expected.get("messages", [])
    assert len(actual_msgs) == len(
        expected_msgs
    ), f"{name}: the number of uploads changed ({len(actual_msgs)} vs {len(expected_msgs)})"
    for index, (left, right) in enumerate(zip(actual_msgs, expected_msgs)):
        structural = ("client_id", "train_samples", "keys")
        differing = [k for k in structural if left.get(k) != right.get(k)]
        assert not differing, f"{name}: messages[{index}] changed: {differing}"
    # the machine-invariant positions of each list entry (the digests --
    # float-dependent -- sit at metrics_log[2] and updates[2:4])
    invariant_positions = {"metrics_log": (0, 1, 3), "updates": (0, 1)}
    for key, positions in invariant_positions.items():
        actual_entries, expected_entries = actual.get(key, []), expected.get(key, [])
        assert len(actual_entries) == len(
            expected_entries
        ), f"{name}: '{key}' changed in length ({len(actual_entries)} vs {len(expected_entries)})"
        for index, (left, right) in enumerate(zip(actual_entries, expected_entries)):
            left_invariant = [left[i] for i in positions]
            right_invariant = [right[i] for i in positions]
            if left_invariant != right_invariant:
                pytest.fail(
                    f"{name}: '{key}[{index}]' changed\n" f"  baseline: {right_invariant}\n  current : {left_invariant}"
                )


def test_baseline_covers_every_registered_algorithm():
    """The baseline must cover all 20 registered algorithm names.

    The equivalence claim of the migration is "all algorithms", so the
    regression baseline has to cover them all -- otherwise a later change to
    an uncovered algorithm would go unnoticed.
    """
    from fl_sim.algorithms import list_algorithms

    baseline = _load_baseline()
    covered = {entry["algorithm"] for entry in baseline.values()}
    registered = set(list_algorithms())

    missing = registered - covered
    assert not missing, f"the golden baseline does not cover: {sorted(missing)}"


def test_baseline_pins_the_apfl_alignment_fix():
    """APFL ``tau>1`` used to crash (server/client sync-counter misalignment:
    the server aggregated on the broadcast rounds, where no client had
    uploaded yet).  The aggregation now happens at the period-end rounds,
    exactly when the clients upload -- the baseline pins the FIXED behavior,
    so a regression back to the crash fails loudly.

    (The pre-fix baseline recorded ``outcome.status == "raised"`` /
    ``KeyError``; changing that into ``ok`` was the explicit, reviewed
    baseline update accompanying the fix.)
    """
    baseline = _load_baseline()
    entry = baseline.get("apfl-tau2")
    assert entry is not None, "the apfl-tau2 scenario is missing from the baseline"
    assert entry["outcome"]["status"] == "ok"
    assert entry["num_communications"] is not None and entry["num_communications"] > 0
    # uploads must exist somewhere (the period-end rounds carry "parameters")
    assert any(uploads for _, uploads in entry["uploads_per_round"])
