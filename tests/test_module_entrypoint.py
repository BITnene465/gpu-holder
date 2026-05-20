from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from gpu_holder import __version__


ROOT = Path(__file__).resolve().parents[1]


def run_module(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "gpu_holder", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_package_module_entrypoint_prints_version() -> None:
    result = run_module("--version")

    assert result.returncode == 0
    assert result.stdout.strip() == f"gpu-holder {__version__}"
    assert result.stderr == ""


def test_package_module_entrypoint_exposes_guard_help() -> None:
    result = run_module("guard", "--help")

    assert result.returncode == 0
    assert "--target-util" in result.stdout
    assert "--risk-util" in result.stdout
    assert "--mem" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--once" in result.stdout
    assert "from 0 to 1" in result.stdout
