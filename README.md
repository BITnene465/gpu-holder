# gpu-holder

[![CI](https://github.com/BITnene465/gpu-holder/actions/workflows/ci.yml/badge.svg)](https://github.com/BITnene465/gpu-holder/actions/workflows/ci.yml)

`gpu-holder` is a small NVIDIA GPU guard for shared training machines. It starts lightweight CUDA workers when selected GPUs fall below a per-GPU utilization floor, and it yields to real training or inference jobs without killing, suspending, or modifying external processes.

The current worker backend uses optional PyTorch because it is widely available on training hosts. The project is structured so the scheduling policy is independent from the worker backend; a lighter NVIDIA Driver API backend can be added without rewriting the guard logic.

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
gpu-holder guard --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05
```

Stop a foreground guard with `Ctrl+C`.

## Common Commands

```bash
gpu-holder doctor
gpu-holder guard --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05
gpu-holder start --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05
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
  cli.py       # CLI, daemon lifecycle, nvidia-smi snapshots, status output
  models.py    # shared dataclasses for GPU snapshots, processes, and decisions
  policy.py    # per-GPU scheduling policy and memory sizing
  worker.py    # CUDA worker implementation
  __main__.py
```

The policy layer has no subprocess or PyTorch dependency. This keeps the training handoff rules testable and makes it possible to add a lighter worker backend later.

## Dependency Strategy

Base install has no third-party runtime dependency. It uses system `nvidia-smi` for read-only GPU status.

The current compute worker needs CUDA-enabled PyTorch:

```bash
python -m pip install -e ".[torch]"
```

Planned backend direction:

- default: NVIDIA Driver API worker through `libcuda.so.1`, no PyTorch or CUDA Toolkit requirement
- fallback: PyTorch worker for environments that already have PyTorch installed
- monitor-only: status and policy checks without starting CUDA workers

This should be described as "works on machines with NVIDIA driver and accessible CUDA devices", not "works on every machine".

See [docs/backend_strategy.md](docs/backend_strategy.md) for the compatibility boundary and
acceptance criteria before switching the default backend.

## Development

```bash
python -m pip install -e ".[dev]"
python -m ruff check --no-cache src tests
python -m pytest -q
PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache python -m compileall -q src tests
python -m build
```

Real CUDA smoke tests should only run on GPUs you own or know are idle.
