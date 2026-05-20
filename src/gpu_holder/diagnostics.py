from __future__ import annotations

from dataclasses import dataclass, asdict
import platform
import subprocess
import sys


@dataclass(frozen=True)
class DiagnosticCheck:
    name: str
    ok: bool
    detail: str
    suggestion: str | None = None
    required: bool = True


def run_diagnostics() -> dict[str, object]:
    checks = [
        _check_python(),
        _check_monitor_backend(),
        _check_pynvml(),
        _check_nvml_runtime(),
        _check_nvidia_smi(),
        _check_torch(),
    ]
    return {
        "ok": all(check.ok for check in checks if check.required),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "checks": [asdict(check) for check in checks],
    }


def _check_python() -> DiagnosticCheck:
    version = sys.version_info
    ok = version >= (3, 10)
    return DiagnosticCheck(
        name="python",
        ok=ok,
        detail=f"{version.major}.{version.minor}.{version.micro}",
        suggestion=None if ok else "Use Python 3.10 or newer.",
    )


def _check_pynvml() -> DiagnosticCheck:
    try:
        import pynvml  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return DiagnosticCheck(
            name="pynvml",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            suggestion=(
                "Optional but preferred. gpu-holder can fall back to `nvidia-smi` "
                "for monitoring when NVML bindings are unavailable."
            ),
            required=False,
        )
    return DiagnosticCheck(name="pynvml", ok=True, detail="available")


def _check_nvml_runtime() -> DiagnosticCheck:
    try:
        import pynvml
    except Exception as exc:  # noqa: BLE001
        return DiagnosticCheck(
            name="nvml",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            suggestion="Optional when `nvidia-smi` fallback is available.",
            required=False,
        )

    try:
        pynvml.nvmlInit()
        try:
            count = int(pynvml.nvmlDeviceGetCount())
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001
        return DiagnosticCheck(
            name="nvml",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            suggestion=(
                "Check the NVIDIA driver and NVML runtime. "
                "Verify `nvidia-smi` works in this shell."
            ),
            required=False,
        )
    return DiagnosticCheck(name="nvml", ok=True, detail=f"gpu_count={count}")


def _check_monitor_backend() -> DiagnosticCheck:
    nvml_count = _pynvml_gpu_count()
    if nvml_count is not None:
        return DiagnosticCheck(name="monitor_backend", ok=True, detail=f"nvml gpu_count={nvml_count}")
    smi_count = _nvidia_smi_gpu_count()
    if smi_count is not None:
        return DiagnosticCheck(
            name="monitor_backend",
            ok=True,
            detail=f"nvidia-smi fallback gpu_count={smi_count}",
        )
    return DiagnosticCheck(
        name="monitor_backend",
        ok=False,
        detail="neither pynvml/NVML nor nvidia-smi monitoring is available",
        suggestion="Install nvidia-ml-py or ensure `nvidia-smi` is available in PATH.",
    )


def _check_nvidia_smi() -> DiagnosticCheck:
    count = _nvidia_smi_gpu_count()
    if count is None:
        return DiagnosticCheck(
            name="nvidia_smi",
            ok=False,
            detail="unavailable",
            suggestion="Ensure `nvidia-smi` is available in PATH if you want dependency-light monitoring.",
            required=False,
        )
    return DiagnosticCheck(
        name="nvidia_smi",
        ok=True,
        detail=f"gpu_count={count}",
        required=False,
    )


def _check_torch() -> DiagnosticCheck:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return DiagnosticCheck(
            name="torch",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            suggestion=(
                "Install a CUDA-compatible PyTorch build for this machine. "
                "For package metadata only, use: python -m pip install 'gpu-holder[cuda]'"
            ),
        )
    try:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
    except Exception as exc:  # noqa: BLE001
        return DiagnosticCheck(
            name="torch_cuda",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            suggestion="Check PyTorch CUDA installation and driver compatibility.",
        )
    return DiagnosticCheck(
        name="torch_cuda",
        ok=cuda_available,
        detail=f"torch={torch.__version__} cuda_available={cuda_available} device_count={device_count}",
        suggestion=(
            None
            if cuda_available
            else "PyTorch is installed but CUDA is unavailable. Check CUDA-enabled torch wheels, driver visibility, and CUDA_VISIBLE_DEVICES."
        ),
    )


def _pynvml_gpu_count() -> int | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return int(pynvml.nvmlDeviceGetCount())
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        return None


def _nvidia_smi_gpu_count() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return len(lines)
