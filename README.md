# gpu-holder

[![CI](https://github.com/BITnene465/gpu-holder/actions/workflows/ci.yml/badge.svg)](https://github.com/BITnene465/gpu-holder/actions/workflows/ci.yml)

`gpu-holder` is a small NVIDIA GPU guard for shared training machines. It starts lightweight CUDA workers when selected GPUs fall below a per-GPU utilization floor, and it yields to real training or inference jobs without killing, suspending, or modifying external processes.

The default worker backend uses optional PyTorch because it is widely available on training hosts.
An experimental NVIDIA Driver API worker is also available for machines where installing PyTorch is
unwanted. The scheduling policy is independent from worker backends, so either backend follows the
same per-GPU safety rules.

## Why

Shared GPU machines are often reclaimed or flagged as idle when utilization drops during data loading, startup, checkpointing, or stalled training. `gpu-holder` keeps utilization above a configured floor while preserving a narrow safety boundary:

- It only starts and stops its own holder workers.
- External CUDA processes are read-only scheduling signals.
- It never kills, suspends, renices, or otherwise controls user training jobs.
- Decisions are per GPU, not based on whole-machine average utilization.
- Large training processes get a startup grace window before any assist holder is allowed.

## Quick Start

Install the package in editable mode:

```bash
git clone https://github.com/BITnene465/gpu-holder.git
cd gpu-holder
python -m pip install -e .
```

If your environment already has a CUDA-enabled PyTorch, the worker can use it directly. Otherwise install the optional worker dependency:

```bash
python -m pip install -e ".[torch]"
```

Run a foreground guard on 8 GPUs:

```bash
gpu-holder guard --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05 --backend torch
```

Stop a foreground guard with `Ctrl+C`.

## Common Commands

```bash
gpu-holder doctor
gpu-holder doctor --backend torch
gpu-holder doctor --backend driver
gpu-holder guard --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05 --backend torch
gpu-holder guard --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05 --backend driver
gpu-holder start --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05 --backend torch
gpu-holder status
gpu-holder dashboard
gpu-holder stop
```

Dry-run a single scheduling decision without starting workers or writing state:

```bash
gpu-holder guard --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05 --dry-run
```

Run one real guard iteration and exit:

```bash
gpu-holder guard --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05 --once
```

## Policy

All utilization and memory ratios use `0..1` floats. Legacy percent forms like `60`, `60%`, and `0.6` are accepted and normalized to `0.6`.

Important defaults:

- `--risk-util 0.5`: start a holder when a GPU is below 50% utilization.
- `--target-util 0.75`: holder compute duty target for each GPU.
- `--mem 0.2`: reserve up to 20% of each GPU's memory for the holder.
- `--reserve 2GiB`: keep this much GPU memory free.
- `--busy-process-mem-threshold 10GiB`: classify large external CUDA processes as training/inference jobs.
- `--process-grace-window 120`: let large external processes initialize for 120 seconds before assist is allowed.
- `--sample-interval 2`: re-evaluate every 2 seconds.

Per-GPU scheduling rules:

1. No external CUDA process and utilization is below `risk-util`: start or keep a normal holder.
2. Normal holder is already running: keep it running until an external CUDA process appears.
3. Large external CUDA process appears: stop the holder on that GPU and enter `busy_process_grace`.
4. Grace window is active: keep yielding, even if the external process has low utilization.
5. Grace window has expired and utilization is still below `risk-util`: start an assist holder with `--assist-mem`.
6. External process utilization rises above `risk-util`: release the assist holder again.
7. External process exits and the GPU drops below `risk-util`: reclaim the GPU on the next sample.

## Safety Model

`gpu-holder` deliberately does not manage external jobs. The only process-control operations are:

- stop its own foreground/background guard
- stop worker child processes created by that guard

It refuses to stop a pidfile that does not point to a `gpu-holder` guard process.

## Architecture

```text
src/gpu_holder/
  cli.py       # CLI, daemon lifecycle, status files, and guard orchestration
  backends.py  # backend selection and backend health checks
  driver_backend.py # NVIDIA Driver API diagnostics and worker implementation
  models.py    # shared dataclasses for GPU snapshots, processes, and decisions
  policy.py    # per-GPU scheduling policy and memory sizing
  telemetry.py # read-only nvidia-smi collection and snapshot parsing
  worker.py    # worker process lifecycle and backend dispatch
  torch_backend.py # PyTorch CUDA worker implementation
  __main__.py
```

The policy layer has no subprocess or PyTorch dependency. GPU telemetry is isolated from the CLI
and scheduler, and the worker lifecycle is separate from backend implementations. This keeps the
handoff rules testable and makes lighter backends easier to add.

## Dependency Strategy

Base install has no third-party runtime dependency. It uses system `nvidia-smi` for read-only GPU status.

The default compute worker needs CUDA-enabled PyTorch:

```bash
python -m pip install -e ".[torch]"
```

The backend selector exposes `--backend torch` and `--backend driver`. The `torch` backend remains
the default. The `driver` backend uses `ctypes`, `libcuda.so.1`, and embedded PTX, so it does not
require PyTorch, CUDA Toolkit, `nvcc`, CuPy, Numba, or local compilation.
`gpu-holder doctor --backend driver` can already check whether `libcuda.so.1` and CUDA devices are
visible through the NVIDIA Driver API, then JIT and launch a tiny embedded PTX kernel.

Planned backend direction:

- default: NVIDIA Driver API worker through `libcuda.so.1`, no PyTorch or CUDA Toolkit requirement
- fallback: PyTorch worker for environments that already have PyTorch installed
- monitor-only: status and policy checks without starting CUDA workers

This should be described as "works on machines with NVIDIA driver and accessible CUDA devices", not "works on every machine".

See [docs/backend_strategy.md](docs/backend_strategy.md) for the compatibility boundary and
acceptance criteria before switching the default backend.

See [ROADMAP.md](ROADMAP.md) for current scope, near-term priorities, and explicit non-goals.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture rules, safety invariants, and pull request
expectations.

```bash
python -m pip install -e ".[dev]"
python -m ruff check --no-cache src tests
python -m pytest -q
PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache python -m compileall -q src tests
python -m build
```

Real CUDA smoke tests should only run on GPUs you own or know are idle.
