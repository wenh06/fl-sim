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
}


@pytest.fixture()
def isolated_log_dir(tmp_path, monkeypatch):
    """Redirect fl-sim's default log directory into a temporary directory."""
    monkeypatch.setattr(loggers_mod, "DEFAULT_LOG_DIR", tmp_path)
    yield tmp_path


def _run_script(script: Path) -> None:
    """Execute a tutorial script in-process, restoring ``sys.path`` afterwards.

    ``run_name="__main__"`` is required: without it the scripts'
    module-level code still runs under ``runpy``, but any
    ``if __name__ == "__main__":`` guard would silently not execute.
    """
    saved_sys_path = list(sys.path)
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.path[:] = saved_sys_path


def test_quickstart_script_runs_and_respects_tutorial_budget(isolated_log_dir):
    script = TUTORIAL_SCRIPTS["quickstart"]
    _run_script(script)

    run_dir = isolated_log_dir / "quickstart"
    assert run_dir.is_dir(), "the quickstart run should log into its own subdirectory"

    # tutorial budget, asserted on the values the run actually used
    config = json.loads((run_dir / "config.json").read_text())
    assert config["server"]["num_iters"] <= 5
    assert config["server"]["num_clients"] <= 30
    assert config["client"]["num_epochs"] == 1

    # the docs promise one metrics json + one txt log per run
    metrics_files = [p for p in run_dir.glob("*.json") if p.name != "config.json"]
    assert len(metrics_files) == 1
    assert metrics_files[0].with_suffix(".txt").exists()

    # and the script's own promise: a readable latest val accuracy
    metrics = json.loads(metrics_files[0].read_text())
    latest = metrics["val"]["Server"][-1]
    assert 0.0 <= latest["acc"] <= 1.0


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
