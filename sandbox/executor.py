"""
Sandboxed Python subprocess executor.

Runs generated transformation code in a clean subprocess with only
INPUT_PATH, OUTPUT_PATH, and PATH in the environment.

Docker mode (3.1):
  Set DOCKER_SANDBOX_IMAGE to a Docker image that has pandas + pyarrow
  installed (e.g. "data-agent-sandbox:latest"). When set, execution uses
  `docker run --rm --network none` for full network/filesystem isolation.
  Falls back to subprocess automatically if the image is unavailable.

Subprocess fallback:
  Used when DOCKER_SANDBOX_IMAGE is unset or Docker is unavailable.
  Clean env, /tmp cwd — same contract as Docker mode.

Never call this with database credentials, API keys, or any secrets.
The calling code is responsible for constructing input/output paths.
"""

import os
import shutil
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
    docker_image = os.getenv("DOCKER_SANDBOX_IMAGE", "").strip()
    if docker_image and shutil.which("docker"):
        return _run_docker(code, input_path, output_path, timeout, docker_image)
    return _run_subprocess(code, input_path, output_path, timeout)


# ---------------------------------------------------------------------------
# Docker execution path
# ---------------------------------------------------------------------------

def _run_docker(
    code: str,
    input_path: str,
    output_path: str,
    timeout: int,
    image: str,
) -> dict:
    """
    Execute in a Docker container with --network none for full isolation.

    Mount strategy:
      - input file is mounted read-only at /data/input/<filename>
      - output directory is mounted read-write at /data/output/
      - a tmpfs at /tmp for the script itself

    The generated code sees:
      INPUT_PATH  = /data/input/<filename>
      OUTPUT_PATH = /data/output/out.parquet
    """
    input_path = str(Path(input_path).resolve())
    input_filename = Path(input_path).name
    output_dir = str(Path(output_path).parent.resolve())

    tmp_script: str | None = None
    try:
        # Write script to a temp file that we can mount into the container
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="sandbox_", dir="/tmp",
            delete=False, encoding="utf-8",
        ) as f:
            f.write(code)
            tmp_script = f.name

        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", "512m",
            "--cpus", "1",
            # Mount input file read-only
            "--volume", f"{input_path}:/data/input/{input_filename}:ro",
            # Mount output dir read-write
            "--volume", f"{output_dir}:/data/output:rw",
            # Mount the script read-only
            "--volume", f"{tmp_script}:/sandbox/script.py:ro",
            # Env vars the script sees
            "--env", f"INPUT_PATH=/data/input/{input_filename}",
            "--env", f"OUTPUT_PATH=/data/output/{Path(output_path).name}",
            "--env", "PATH=/usr/local/bin:/usr/bin:/bin",
            "--workdir", "/tmp",
            image,
            "python", "/sandbox/script.py",
        ]

        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return _timeout_result(timeout)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        return _build_result(result.returncode, result.stdout, result.stderr, elapsed_ms)

    except Exception as exc:
        # Docker unavailable or misconfigured — fall back to subprocess
        return _run_subprocess(code, input_path, output_path, timeout)

    finally:
        if tmp_script and Path(tmp_script).exists():
            try:
                Path(tmp_script).unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Subprocess execution path (fallback)
# ---------------------------------------------------------------------------

def _run_subprocess(
    code: str,
    input_path: str,
    output_path: str,
    timeout: int,
) -> dict:
    """
    Execute in a clean subprocess. No network restriction, but env is wiped
    to only INPUT_PATH, OUTPUT_PATH, and PATH.
    """
    tmp_script: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="sandbox_", dir="/tmp",
            delete=False, encoding="utf-8",
        ) as f:
            f.write(code)
            tmp_script = f.name

        clean_env = {
            "INPUT_PATH": input_path,
            "OUTPUT_PATH": output_path,
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }

        start = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, tmp_script],
                capture_output=True,
                text=True,
                cwd="/tmp",
                env=clean_env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return _timeout_result(timeout)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return _build_result(result.returncode, result.stdout, result.stderr, elapsed_ms)

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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_result(returncode: int, stdout: str, stderr: str, elapsed_ms: int) -> dict:
    success = returncode == 0
    return {
        "success": success,
        "stdout": stdout or "",
        "stderr": stderr or "",
        "rows_input": _parse_rows(stdout, "ROWS_IN"),
        "rows_output": _parse_rows(stdout, "ROWS_OUT"),
        "execution_time_ms": elapsed_ms,
    }


def _timeout_result(timeout: int) -> dict:
    return {
        "success": False,
        "stdout": "",
        "stderr": f"Execution timed out after {timeout} seconds.",
        "rows_input": None,
        "rows_output": None,
        "execution_time_ms": timeout * 1000,
    }


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
