# Development Log

This file records project decisions and regressions that are likely to matter again. It is not a
changelog; release-facing changes belong in `README.md` or `ROADMAP.md`.

## 2026-05-21 - Driver backend compatibility boundary

### Symptom

The project direction favored `ctypes + libcuda.so.1 + embedded PTX` as the lightweight worker
path, but that could be misread as "runs on every machine".

### Root Cause

The Driver API path removes the PyTorch and CUDA Toolkit dependency, but it still depends on an
installed NVIDIA driver, accessible NVIDIA GPU devices, and a driver new enough to JIT the embedded
PTX.

### Impact

Users on hosts without NVIDIA GPUs, broken driver installs, restricted containers, macOS, AMD/Intel
GPUs, or unsupported Windows setups should not expect the driver backend to work.

### Fix

Document the precise boundary as "Linux machines with an NVIDIA driver and accessible CUDA devices",
keep PyTorch as an explicit fallback backend, and expose `gpu-holder doctor --backend driver` as the
first diagnostic entry point.

### Regression Tests

- `gpu-holder doctor --backend driver --json`
- short `WorkerProcess(..., backend="driver")` start/stop smoke
- `gpu-holder guard --gpus 0 --dry-run --backend driver`
- unit tests covering backend strategy documentation and shared worker controls

### Follow-Up Defense

Keep backend-neutral timing and hold-mode semantics in `worker_controls.py`, and keep future
compatibility claims in `docs/backend_strategy.md` tied to concrete doctor and worker smoke tests.
