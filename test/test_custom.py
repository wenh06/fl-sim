"""
Test custom dataset and custom algorithm.

For the test to succeed, it should be run in the root directory of the project with
`pytest -s -v test/test_custom.py`,
**NOT** in the test directory with `pytest -s -v test_custom.py`.
"""

from fl_sim.utils.misc import execute_cmd


def test_custom_dataset_algorithm():
    cmd = "fl-sim test-files/custom_conf.yml"
    exitcode, _ = execute_cmd(cmd)
    assert exitcode == 0
