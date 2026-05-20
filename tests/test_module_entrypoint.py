from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from gpu_holder import __version__


ROOT = Path(__file__).resolve().parents[1]


def test_package_module_entrypoint_prints_version() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-m", "gpu_holder", "--version"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f"gpu-holder {__version__}"
    assert result.stderr == ""


def test_package_module_entrypoint_runs_cli_commands() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-m", "gpu_holder", "plan", "--fake"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "gpu=0 action=hold" in result.stdout
