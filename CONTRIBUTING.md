# Contributing

`gpu-holder` is a small operations tool for shared NVIDIA training machines. Changes should keep
the project easy to install, easy to audit, and conservative around other users' jobs.

## Development Setup

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[torch]"  # optional, only needed for real CUDA worker tests
```

Run the normal local checks before opening a pull request:

```bash
python -m ruff check --no-cache src tests
python -m pytest -q
PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache python -m compileall -q src tests
python -m build
```

## Safety Rules

These invariants are part of the project contract:

- External CUDA processes are read-only scheduling signals.
- `gpu-holder` must only stop the guard and worker processes it created.
- Do not add behavior that kills, suspends, renices, ptraces, or otherwise controls external jobs.
- Prefer releasing a holder over risking interference with a training or inference process.
- Real CUDA smoke tests must only run on GPUs you own or know are idle.

## Architecture Rules

- Keep per-GPU scheduling decisions in `policy.py`.
- Keep CLI, daemon lifecycle, and status files in `cli.py`.
- Keep read-only `nvidia-smi` collection and GPU snapshot parsing in `telemetry.py`.
- Keep worker process startup, shutdown, and backend dispatch in `worker.py`.
- Keep backend-neutral worker timing and hold-mode controls in `worker_controls.py`.
- Keep PyTorch-specific compute code in `torch_backend.py`.
- Keep NVIDIA Driver API diagnostics and worker code in `driver_backend.py`.
- Keep backend names, validation, and health-check dispatch in `backends.py`.

The policy layer should stay backend-neutral and should not import PyTorch, `ctypes`, or subprocess
worker implementations.

See [ROADMAP.md](ROADMAP.md) before expanding scope. New features should preserve the project
non-goals and safety boundary.

## Testing Guidance

Use unit tests for policy, parsing, status payloads, backend validation, and worker lifecycle.
Do not require real GPUs in the default test suite.

Use manual GPU checks only when a change affects real CUDA behavior:

```bash
gpu-holder doctor --json
gpu-holder doctor --backend driver --json
gpu-holder guard --gpus 0 --risk-util 0.6 --target-util 0.9 --mem 0 --backend driver --once
gpu-holder guard --gpus 0 --risk-util 0.6 --target-util 0.9 --mem 0.05 --backend torch --once
```

Include the commands you ran in the pull request.
