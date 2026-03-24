"""
Sandboxed Python subprocess executor.

Runs generated transformation code in a clean subprocess with only
INPUT_PATH, OUTPUT_PATH, and PATH in the environment.

Never call this with database credentials, API keys, or any secrets.
The calling code is responsible for constructing input/output paths.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_sandboxed(
    code: str,
    input_path: str,
    output_path: str,
    timeout: int = 30,
) -> dict:
    """
    Execute a Python script string in a sandboxed subprocess.

    The script receives INPUT_PATH and OUTPUT_PATH as environment variables.
    It must print "ROWS_IN: N" before processing and "ROWS_OUT: N" after.
    These lines are parsed to populate rows_input and rows_output.

    Args:
        code:        Python source code to execute.
        input_path:  Absolute path to the input data file.
        output_path: Absolute path where the script should write its output.
        timeout:     Maximum seconds to allow before killing the process.

    Returns:
        {
            success:           bool,
            stdout:            str,
            stderr:            str,
            rows_input:        int | None,
            rows_output:       int | None,
            execution_time_ms: int,
        }
    """
    tmp_script: str | None = None

    try:
        # Write code to a temp file in /tmp
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="sandbox_",
            dir="/tmp",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(code)
            tmp_script = f.name

        # Clean environment — only what the script is allowed to see
        clean_env = {
            "INPUT_PATH": input_path,
            "OUTPUT_PATH": output_path,
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }

        start = time.monotonic()
        result = subprocess.run(
            [sys.executable, tmp_script],
            capture_output=True,
            text=True,
            cwd="/tmp",
            env=clean_env,
            timeout=timeout,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        success = result.returncode == 0
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        rows_input = _parse_rows(stdout, "ROWS_IN")
        rows_output = _parse_rows(stdout, "ROWS_OUT")

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "rows_input": rows_input,
            "rows_output": rows_output,
            "execution_time_ms": elapsed_ms,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds.",
            "rows_input": None,
            "rows_output": None,
            "execution_time_ms": timeout * 1000,
        }

    except Exception as exc:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(exc),
            "rows_input": None,
            "rows_output": None,
            "execution_time_ms": 0,
        }

    finally:
        if tmp_script and Path(tmp_script).exists():
            try:
                Path(tmp_script).unlink()
            except OSError:
                pass


def _parse_rows(stdout: str, marker: str) -> int | None:
    """
    Parse "ROWS_IN: 500" or "ROWS_OUT: 487" from stdout.
    Returns the integer value, or None if the marker is not found.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(f"{marker}:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None
