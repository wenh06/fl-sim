"""
Test custom dataset and custom algorithm.

For the test to succeed, it should be run in the root directory of the project with
`pytest -s -v test/test_custom.py`,
**NOT** in the test directory with `pytest -s -v test_custom.py`.
"""

import sys

from fl_sim.utils.misc import execute_cmd


def test_custom_dataset_algorithm():
    # invoke the module entry point through the running interpreter instead
    # of the `fl-sim` console script, so the test does not depend on the
    # script directory being on PATH
    cmd = f"'{sys.executable}' -m fl_sim.cli test-files/custom_conf.yml"
    exitcode, _ = execute_cmd(cmd)
    assert exitcode == 0
