"""pytest configuration for the F1 equivalence / golden-regression skeleton.

Makes the repository root importable (``fl_sim``) and this directory
importable (``golden_lib``), so the tests can be run from anywhere:

    pytest test/test_parallel_equivalence -q
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

for path in (str(REPO_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)
