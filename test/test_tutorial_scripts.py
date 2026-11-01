"""Behavioral smoke tests for the tutorial scripts in ``test-files/``.

The scripts are the source of truth for the code blocks shown in the
documentation: ``docs/source/*.rst`` embeds them via Sphinx
``literalinclude`` directives, so a green run here means the code the
documentation shows actually executes and produces what the docs promise.

The runs are pinned to the tutorial budget (``server.num_iters <= 5``,
``server.num_clients <= 30``, ``client.num_epochs == 1``), asserted on
the *runtime* values each run writes to its ``config.json`` — so the
"tutorial-sized runs" property is guarded even if someone edits the
scripts later.

Each run is isolated: fl-sim's log directory is redirected into a
temporary directory, and ``sys.path`` is restored after ``runpy``.
The whole suite exercises real training runs and takes well under a
minute on a laptop.
"""

import json
import re
import runpy
import sys
from pathlib import Path

import pytest

import fl_sim.utils.loggers as loggers_mod

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_SOURCE = REPO_ROOT / "docs" / "source"

TUTORIAL_SCRIPTS = {
    "quickstart": REPO_ROOT / "test-files" / "quickstart.py",
    "tutorial_first_experiment": REPO_ROOT / "test-files" / "tutorial_first_experiment.py",
    "tutorial_visualization": REPO_ROOT / "test-files" / "tutorial_visualization.py",
    "tutorial_parallel_check": REPO_ROOT / "test-files" / "tutorial_parallel_check.py",
}
CLI_MATRIX_DEMO = REPO_ROOT / "example-configs" / "fedsynthetic-demo.yml"


@pytest.fixture()
def isolated_log_dir(tmp_path, monkeypatch):
    """Redirect fl-sim's default log directory into a temporary directory."""
    monkeypatch.setattr(loggers_mod, "DEFAULT_LOG_DIR", tmp_path)
    yield tmp_path


@pytest.fixture()
def in_tmp_cwd(tmp_path, monkeypatch):
    """Run scripts from a temporary working directory, so any file they
    write relative to the CWD (e.g. an exported PDF) stays contained."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


def _run_script(script: Path):
    """Execute a tutorial script in-process, restoring ``sys.path`` afterwards.

    ``run_name="__main__"`` is required: without it the scripts'
    module-level code still runs under ``runpy``, but any
    ``if __name__ == "__main__":`` guard would silently not execute.
    Returns the script's final global namespace for behavioral assertions.
    """
    saved_sys_path = list(sys.path)
    try:
        return runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.path[:] = saved_sys_path


def _check_tutorial_budget(log_dir: Path) -> dict:
    """Assert the run respected the tutorial size budget, via config.json."""
    config = json.loads((log_dir / "config.json").read_text())
    assert config["server"]["num_iters"] <= 5
    assert config["server"]["num_clients"] <= 30
    assert config["client"]["num_epochs"] == 1
    return config


def test_quickstart_script_runs_and_respects_tutorial_budget(isolated_log_dir):
    _run_script(TUTORIAL_SCRIPTS["quickstart"])

    run_dir = isolated_log_dir / "quickstart"
    assert run_dir.is_dir(), "the quickstart run should log into its own subdirectory"

    _check_tutorial_budget(run_dir)

    # the docs promise one metrics json + one txt log per run
    metrics_files = [p for p in run_dir.glob("*.json") if p.name != "config.json"]
    assert len(metrics_files) == 1
    assert metrics_files[0].with_suffix(".txt").exists()

    # and the script's own promise: a readable latest val accuracy
    metrics = json.loads(metrics_files[0].read_text())
    latest = metrics["val"]["Server"][-1]
    assert 0.0 <= latest["acc"] <= 1.0


def test_first_experiment_script_runs_and_produces_the_documented_table(isolated_log_dir):
    script = TUTORIAL_SCRIPTS["tutorial_first_experiment"]
    script_dir = isolated_log_dir  # the script logs into t1_* subdirectories

    namespace = _run_script(script)

    # six comparison runs, each within the tutorial budget
    results = namespace["results"]
    assert len(results) == 6
    assert {row["algorithm"] for row in results} == {"FedAvg", "FedProx", "SCAFFOLD"}
    assert {row["sample_ratio"] for row in results} == {0.5, 1.0}
    for row in results:
        assert 0.0 <= row["final_acc"] <= 1.0
        # 2 x sampled x rounds: 15 sampled clients x 3 rounds x 2 directions
        expected = 90 if row["sample_ratio"] == 0.5 else 180
        assert (
            row["num_communications"] == expected
        ), "the docs teach this identity; if it breaks, the tutorial narrative breaks"
    for sub in script_dir.iterdir():
        if sub.name.startswith("t1_") and sub.is_dir():
            _check_tutorial_budget(sub)

    # the FedPD coda: fewer communications than FedAvg at the same
    # schedule, and at least one real uplink (guards against the degenerate
    # p/comm_freq combination where FedPD never uploads and trivially
    # "wins" the comparison)
    fedavg, fedpd = namespace["fedavg"], namespace["fedpd"]
    sampled = 15
    assert fedavg._num_communications == 2 * sampled * 5 == 150
    assert fedpd._num_communications < fedavg._num_communications
    downlink = sampled * 5
    assert fedpd._num_communications - downlink > 0, "FedPD should have uploaded at least once"


def test_cli_matrix_demo_expands_and_runs(isolated_log_dir):
    from fl_sim.cli import parse_config_file, single_run

    configs, _parallel = parse_config_file(CLI_MATRIX_DEMO)

    # 2 ratios x 2 seeds, within the tutorial matrix budget
    assert 1 < len(configs) <= 4
    ratios = {cfg["algorithm"]["server"]["clients_sample_ratio"] for cfg in configs}
    seeds = {cfg["seed"] for cfg in configs}
    assert len(ratios) * len(seeds) == len(configs), "the matrix should expand into distinct combinations"
    # interpolation resolved everywhere
    for cfg in configs:
        assert "${{ matrix." not in str(cfg)

    for cfg in configs:
        single_run(cfg)

    run_dir = isolated_log_dir / "fedsynthetic-demo"
    # every run must leave its own logs: distinct tags prevent same-minute
    # overwrites, which is exactly the pitfall the tutorial teaches
    jsons = [p for p in run_dir.glob("*.json") if p.name != "config.json"]
    txts = list(run_dir.glob("*.txt"))
    assert len(jsons) == len(configs), "each matrix arm should produce its own json log"
    assert len(txts) == len(configs)
    _check_tutorial_budget(run_dir)


def test_visualization_script_merges_seeds_and_exports(isolated_log_dir, in_tmp_cwd):
    namespace = _run_script(TUTORIAL_SCRIPTS["tutorial_visualization"])

    # two tagged runs, both within the tutorial budget
    run_dir = isolated_log_dir / "t3_seeds"
    jsons = [p for p in run_dir.glob("*.json") if p.name != "config.json"]
    assert len(jsons) == 2, "the two seeds must not overwrite each other's logs"
    _check_tutorial_budget(run_dir)

    # the merge produced one non-empty curve per seed, and the PDF export
    # landed in the (temporary) working directory
    curves = namespace["curves"]
    assert len(curves) == 2
    for curve in curves:
        assert len(curve["x"]) > 0 and len(curve["y"]) > 0
    assert (in_tmp_cwd / "fedavg-multi-seed.pdf").exists(), "the mean curve should be exported to PDF"

    # the dashboard object is constructible outside Jupyter (it prints a
    # note and stays inert there; rendering is covered by manual use)
    assert namespace["panel"] is not None


def test_parallel_check_script_verifies_bitwise_reproducibility(isolated_log_dir):
    namespace = _run_script(TUTORIAL_SCRIPTS["tutorial_parallel_check"])

    # the L1 guarantee, demonstrated end to end: same seed, per-client RNG,
    # two different executor backends -> bitwise identical final weights
    sequential, threads = namespace["sequential"], namespace["threads"]
    assert sequential._num_communications == threads._num_communications
    assert namespace["bitwise_equal"](
        sequential, threads
    ), "sequential and threads with rng_mode='per_client' must agree bitwise"

    # global mode: participation stays backend-independent; local
    # trajectories may interleave by design, so only the counts are asserted
    sequential_g, threads_g = namespace["sequential_g"], namespace["threads_g"]
    assert sequential_g._num_communications == threads_g._num_communications

    for sub in isolated_log_dir.iterdir():
        if sub.name.startswith("t4_") and sub.is_dir():
            _check_tutorial_budget(sub)


def test_tutorial_scripts_keep_literalinclude_markers():
    """The markers delimit the fragments the docs include; dropping one
    silently empties a documentation code block."""
    for name, script in TUTORIAL_SCRIPTS.items():
        src = script.read_text()
        markers = re.findall(r"^# --- (begin|end): ([a-z-]+) ---$", src, re.MULTILINE)
        begins = [tag for mode, tag in markers if mode == "begin"]
        ends = [tag for mode, tag in markers if mode == "end"]
        assert begins, f"{name}: no literalinclude markers"
        assert begins == ends, f"{name}: unbalanced begin/end markers {begins} vs {ends}"


def test_documented_literalinclude_targets_exist():
    """Every ``literalinclude`` in the docs must point at an existing file —
    a moved script would otherwise leave an empty code block on the site."""
    pattern = re.compile(r"^\.\.\s+literalinclude::\s+(\S+)", re.MULTILINE)
    checked = 0
    for rst in DOCS_SOURCE.rglob("*.rst"):
        for match in pattern.finditer(rst.read_text()):
            target = match.group(1)
            if target.startswith(("http://", "https://")):
                continue
            resolved = (rst.parent / target).resolve()
            assert resolved.exists(), f"{rst.relative_to(DOCS_SOURCE)}: literalinclude target missing: {target}"
            checked += 1
    assert checked >= 1, "expected at least one literalinclude in the docs"
