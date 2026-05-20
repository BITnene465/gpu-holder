from __future__ import annotations

import gpu_holder.diagnostics as diagnostics
from gpu_holder.diagnostics import DiagnosticCheck, run_diagnostics


def test_diagnostics_accepts_nvidia_smi_monitor_fallback(monkeypatch) -> None:
    monkeypatch.setattr(diagnostics, "_pynvml_gpu_count", lambda: None)
    monkeypatch.setattr(diagnostics, "_nvidia_smi_gpu_count", lambda: 8)
    monkeypatch.setattr(
        diagnostics,
        "_check_pynvml",
        lambda: DiagnosticCheck("pynvml", False, "missing", required=False),
    )
    monkeypatch.setattr(
        diagnostics,
        "_check_nvml_runtime",
        lambda: DiagnosticCheck("nvml", False, "missing", required=False),
    )
    monkeypatch.setattr(
        diagnostics,
        "_check_torch",
        lambda: DiagnosticCheck("torch_cuda", True, "cuda_available=True"),
    )

    payload = run_diagnostics()

    assert payload["ok"] is True
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["monitor_backend"]["ok"] is True
    assert checks["monitor_backend"]["detail"] == "nvidia-smi fallback gpu_count=8"
    assert checks["pynvml"]["required"] is False


def test_diagnostics_fails_without_any_monitor_backend(monkeypatch) -> None:
    monkeypatch.setattr(diagnostics, "_pynvml_gpu_count", lambda: None)
    monkeypatch.setattr(diagnostics, "_nvidia_smi_gpu_count", lambda: None)
    monkeypatch.setattr(
        diagnostics,
        "_check_pynvml",
        lambda: DiagnosticCheck("pynvml", False, "missing", required=False),
    )
    monkeypatch.setattr(
        diagnostics,
        "_check_nvml_runtime",
        lambda: DiagnosticCheck("nvml", False, "missing", required=False),
    )
    monkeypatch.setattr(
        diagnostics,
        "_check_torch",
        lambda: DiagnosticCheck("torch_cuda", True, "cuda_available=True"),
    )

    payload = run_diagnostics()

    assert payload["ok"] is False
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["monitor_backend"]["ok"] is False
    assert checks["monitor_backend"]["required"] is True
