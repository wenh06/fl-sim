#!/usr/bin/env python
"""Generate (or verify) the golden baseline for the parallel-execution layer.

Usage
-----
Generate the baseline from the current checkout::

    python make_golden.py --out golden/baseline.json

Verify that the *current* code still reproduces the committed baseline::

    python make_golden.py --out golden/baseline.json --check

Prove that the baseline also holds on a **pre-refactor** checkout (this is the
"migration did not change the observable behaviour" evidence; the refactor
landed in PR #50/#51, the pre-refactor master is ``f6c3e33``)::

    git worktree add /tmp/flsim-pre f6c3e33
    python make_golden.py --repo /tmp/flsim-pre --out /tmp/pre_refactor.json
    python make_golden.py --out golden/baseline.json --compare /tmp/pre_refactor.json

The harness only uses APIs that exist on both sides of the refactor, so the
two JSON files are directly comparable.  The ``--compare`` mode reports the
first differing scenario (and its field) instead of dumping two large files.

Notes
-----
* The baseline records a *failing* scenario (``apfl-tau2``) as an exception
  rather than a crash of the generator: a known algorithm defect is pinned
  too, so that fixing it shows up as an explicit baseline change.
* ``--repo`` must point at a checkout whose ``fl_sim`` package is importable;
  the script prepends it to ``sys.path`` before importing ``fl_sim`` and
  prints the resolved path so you can confirm which copy was used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent


def _prepend_repo(repo: Optional[str]) -> None:
    if repo:
        sys.path.insert(0, str(Path(repo).resolve()))
    # the repo root of the *current* checkout, so that `fl_sim` is importable
    # when the script is run from anywhere
    sys.path.append(str(HERE.parent.parent))


def _flatten(flat: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in flat.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value and all(isinstance(v, (dict, list, tuple)) for v in value.values()):
            out.update(_flatten(value, path + "."))
        else:
            out[path] = value
    return out


#: Fields that only the refactored code can produce (the pre-refactor code
#: has no ``_accept_reply``, so the per-client uplink decision is not
#: observable from the outside).  They are reported, not compared.
VERSION_SPECIFIC_FIELDS = (
    "decisions",  # the per-client uplink decision (no _accept_reply pre-refactor)
    "messages",  # the typed upload path (the pre-refactor code pushes into
    # the server buffer directly, so receive_one is never called)
)


def _first_difference(left: Dict[str, Any], right: Dict[str, Any]) -> Optional[str]:
    """The first differing field (version-specific fields are skipped)."""
    flat_left = _flatten(left)
    flat_right = _flatten(right)
    for key in sorted(set(flat_left) & set(flat_right)):
        if any(key == field or key.startswith(field + ".") or key.startswith(field + "[") for field in VERSION_SPECIFIC_FIELDS):
            continue
        if flat_left[key] != flat_right[key]:
            return f"{key}: {flat_left[key]!r} != {flat_right[key]!r}"
    return None


def _not_comparable(left: Dict[str, Any], right: Dict[str, Any]) -> list:
    """Fields that only one of the two versions has (version-specific)."""
    flat_left = set(_flatten(left))
    flat_right = set(_flatten(right))
    only = sorted(flat_left ^ flat_right)
    return [key for key in only if key.split(".")[0].split("[")[0] in VERSION_SPECIFIC_FIELDS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(HERE / "golden" / "baseline.json"), help="output path")
    parser.add_argument("--repo", default=None, help="run against another checkout (e.g. a pre-refactor worktree)")
    parser.add_argument("--check", action="store_true", help="fail if the current code does not reproduce the baseline")
    parser.add_argument("--compare", default=None, help="compare the current run against another baseline file")
    parser.add_argument("--only", default=None, help="restrict to one scenario by name")
    args = parser.parse_args()

    _prepend_repo(args.repo)

    import fl_sim  # noqa: E402  (import after sys.path surgery)

    print(f"using fl_sim from: {fl_sim.__file__}")

    from golden_lib import SCENARIOS, run_scenario  # noqa: E402

    scenarios = SCENARIOS
    if args.only:
        scenarios = [s for s in SCENARIOS if s.name == args.only]
        if not scenarios:
            print(f"unknown scenario: {args.only}", file=sys.stderr)
            return 2

    results = {scenario.name: run_scenario(scenario) for scenario in scenarios}

    if args.compare:
        with open(args.compare, encoding="utf-8") as handle:
            other = json.load(handle)
        other_scenarios = other.get("scenarios", other)
        differences = []
        incomparable = []
        for name, current in results.items():
            reference = other_scenarios.get(name)
            if reference is None:
                differences.append(f"{name}: absent in {args.compare}")
                continue
            diff = _first_difference(current, reference)
            if diff:
                differences.append(f"{name}: {diff}")
            only = _not_comparable(current, reference)
            if only:
                incomparable.append(f"{name}: {only[:6]}{' ...' if len(only) > 6 else ''}")
        if incomparable:
            print("version-specific fields (not compared):")
            for line in incomparable:
                print(f"  - {line}")
        if differences:
            print(f"DIFFERENCES against {args.compare} ({len(differences)}):")
            for line in differences:
                print(f"  - {line}")
            return 1
        print(f"IDENTICAL to {args.compare} for {len(results)} scenario(s)")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"scenarios": results}, handle, indent=1, sort_keys=True)
    print(f"wrote {out_path} ({len(results)} scenarios)")

    if args.check:
        if out_path.exists():
            with open(out_path, encoding="utf-8") as handle:
                committed = json.load(handle)
            committed_scenarios = committed.get("scenarios", committed)
            differences = []
            for name, current in results.items():
                reference = committed_scenarios.get(name)
                if reference is None:
                    differences.append(f"{name}: absent in the committed baseline")
                    continue
                diff = _first_difference(current, reference)
                if diff:
                    differences.append(f"{name}: {diff}")
            if differences:
                print(f"DIFFERENCES against the committed baseline ({len(differences)}):")
                for line in differences:
                    print(f"  - {line}")
                return 1
            print("the current code reproduces the committed baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
