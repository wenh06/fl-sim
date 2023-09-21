"""
"""

from pathlib import Path

from fl_sim.utils.misc import clear_logs, execute_cmd

action_test_config_files = [
    Path(__file__).parents[1].resolve() / "example-configs" / filename
    for filename in [
        "action-test.yml",
        "action-test-simple.yml",
    ]
]


def test_cli():
    for file in action_test_config_files:
        cmd = f"fl-sim {str(file)}"
        exitcode, _ = execute_cmd(cmd)
        assert exitcode == 0
    clear_logs()
