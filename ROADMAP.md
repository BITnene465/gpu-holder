# Roadmap

`gpu-holder` targets engineers and researchers who use shared NVIDIA training machines and need a
small, auditable guard for idle GPU utilization.

## Current Scope

- Foreground and background guard CLI.
- Per-GPU scheduling policy based on utilization, memory, and external CUDA processes.
- Startup grace window for large external training or inference jobs.
- PyTorch worker backend for real CUDA utilization.
- NVIDIA Driver API diagnostics with embedded PTX smoke through
  `gpu-holder doctor --backend driver`.
- Read-only `nvidia-smi` telemetry.
- No third-party runtime dependency for status, policy, and diagnostics.

## Near-Term Priorities

1. Implement a Driver API worker backend using `ctypes + libcuda.so.1 + embedded conservative PTX`.
2. Preserve `--backend torch` as a fallback for environments that already depend on PyTorch.
3. Improve `CUDA_VISIBLE_DEVICES`, physical GPU index, and GPU UUID handling before making the
   Driver API backend the default.
4. Keep default tests GPU-free while documenting explicit manual CUDA smoke checks.

## Compatibility Reality

The Driver API direction is meant to make `gpu-holder` easier to run on shared NVIDIA training
machines. It is not a promise that every machine can run it.

Expected baseline:

- Linux.
- NVIDIA driver installed and healthy.
- `libcuda.so.1` visible to the process.
- Target CUDA devices exposed through `/dev/nvidia*`.
- Driver new enough to JIT the embedded PTX.

Still not covered:

- machines without NVIDIA GPUs
- broken driver installs
- containers without driver libraries or GPU device mounts
- non-NVIDIA GPUs
- macOS
- Windows until a separate `nvcuda.dll` path exists

## Non-Goals

- Managing, killing, suspending, renicing, or otherwise controlling external GPU jobs.
- Replacing cluster schedulers, job queues, or quota systems.
- Guaranteeing compatibility with non-NVIDIA GPUs.
- Claiming support for every machine or every operating system.
- Adding heavyweight monitoring stacks or dashboards to the core package.

## Default Backend Graduation Criteria

The Driver API backend can become the default only after:

- `gpu-holder doctor --backend driver` validates driver load, device count, PTX JIT, and a tiny
  kernel launch.
- `gpu-holder guard --backend driver --once` starts and stops one worker cleanly.
- Foreground shutdown leaves no lingering holder workers.
- `CUDA_VISIBLE_DEVICES` remapping is verified by tests or documented manual checks.
- The PyTorch backend remains available and documented as a fallback.
