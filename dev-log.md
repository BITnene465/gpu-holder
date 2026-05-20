# gpu-holder Development Log

This log records implementation decisions, validation results, and operational
lessons that should not live only in chat history.

## 2026-05-20

### Initial Product Scope

- Goal: build a polite GPU utilization guard for shared NVIDIA machines that
  keeps hourly machine utilization above reclaim thresholds while yielding to
  real workloads.
- Default policy:
  - target machine utilization: `75%`
  - low-util emergency threshold: `50%`
  - low-util emergency window: `60s`
  - default memory hold: `20%`
  - busy process threshold: `10GiB`
- Safety invariant: external GPU processes are read-only scheduling signals.
  The tool may stop only worker processes it started itself.

### Architecture Decisions

- Keep scheduling logic pure and testable in `policy`; keep live process
  control inside `controller`.
- Keep CUDA allocation and compute isolated in per-GPU worker processes.
- Use strict TOML config validation so misspelled keys fail before a long tmux
  or daemon run starts.
- Generate command metadata for shell completion and manuals from shared
  command references to avoid documentation drift.
- Add scenario recipes on top of profiles because real users need copyable
  deployment starting points, not only abstract profile defaults.

### GPU Smoke Result

- Released the old manual `hold.py` session from cuda:7 without killing
  unrelated processes by sending `Ctrl-C` in the `hold` tmux session, then
  restarted it on cuda:5/6 only.
- Verified cuda:7 was free before testing:
  - memory: `3 MiB / 81920 MiB`
  - utilization: `0%`
  - processes: none
- Ran a short real `gpu-holder` worker smoke on physical cuda:7:
  - observed approximately `1650 MiB` memory use
  - observed approximately `93%` utilization
  - observed approximately `333W` power draw
  - stopped only the worker started by the smoke test
- Verified cuda:7 returned to idle after the smoke:
  - memory: `3 MiB / 81920 MiB`
  - utilization: `0%`
  - processes: none

### Dependency Lesson

- Finding: the machine had CUDA-capable PyTorch but lacked `pynvml` /
  `nvidia-ml-py`, so the original full `guard`/`doctor` monitoring path was
  too strict for a dependency-light tool.
- Fix: add a `nvidia-smi` monitoring fallback and make `doctor` validate the
  required monitoring backend separately from optional NVML bindings.
- Follow-up: keep PyTorch and NVML bindings optional package extras because
  CUDA/PyTorch versions are often controlled by the host environment.

### Validation

- Focused tests for recipes and fallback monitoring passed.
- Full test suite passed: `339 passed`.
- Static check passed: `python -m ruff check --no-cache src tests`.
- Compile check passed with external pycache:
  `PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache python -m compileall -q src tests`.

### Open Quality Gaps

- Add CI once the GitHub repository is created or pushed.
- Add a short demo GIF or terminal screenshot for the dashboard.
- Run a supervised full `guard` smoke after confirming the intended GPU is
  available and `nvidia-smi` fallback behavior is acceptable for that host.
