"""
"""

import collections
import subprocess
from pathlib import Path
from typing import Union, List, Tuple


action_test_config_file = (
    Path(__file__).parents[1].resolve() / "example-configs" / "action-test.yml"
)


def execute_cmd(
    cmd: Union[str, List[str]], raise_error: bool = True
) -> Tuple[int, List[str]]:
    """Execute shell command using `Popen`.

    Parameters
    ----------
    cmd : str or list of str
        Shell command to be executed,
        or a list of .sh files to be executed.
    raise_error : bool, default True
        If True, error will be raised when occured.

    Returns
    -------
    exitcode : int
        Exit code returned by `Popen`.
    output_msg : list of str
        Outputs from `stdout` of `Popen`.

    """
    shell_arg, executable_arg = True, None
    s = subprocess.Popen(
        cmd,
        shell=shell_arg,
        executable=executable_arg,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    debug_stdout = collections.deque(maxlen=1000)
    print("\n" + "*" * 10 + "  execute_cmd starts  " + "*" * 10 + "\n")
    while 1:
        line = s.stdout.readline().decode("utf-8", errors="replace")
        if line.rstrip():
            debug_stdout.append(line)
            # print(line)
        exitcode = s.poll()
        if exitcode is not None:
            for line in s.stdout:
                debug_stdout.append(line.decode("utf-8", errors="replace"))
            if exitcode is not None and exitcode != 0:
                error_msg = " ".join(cmd) if not isinstance(cmd, str) else cmd
                error_msg += "\n"
                error_msg += "".join(debug_stdout)
                s.communicate()
                s.stdout.close()
                print("\n" + "*" * 10 + "  execute_cmd failed  " + "*" * 10 + "\n")
                if raise_error:
                    raise subprocess.CalledProcessError(exitcode, error_msg)
                else:
                    output_msg = list(debug_stdout)
                    return exitcode, output_msg
            else:
                break
    s.communicate()
    s.stdout.close()
    output_msg = list(debug_stdout)

    print("\n" + "*" * 10 + "  execute_cmd succeeded  " + "*" * 10 + "\n")

    exitcode = 0

    return exitcode, output_msg


def test_cli():
    cmd = f"fl-sim {str(action_test_config_file)}"
    exitcode, _ = execute_cmd(cmd)
    assert exitcode == 0
