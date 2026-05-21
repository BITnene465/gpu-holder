# Backend Strategy

`gpu-holder` should be easy to run on shared NVIDIA training machines. The long-term default
backend should avoid heavyweight Python CUDA stacks, but the compatibility claim must stay precise.

## Current Backend

The current worker backend uses CUDA-enabled PyTorch when holder workers are started. This is
simple and stable on development machines that already use PyTorch for training, but it is too
heavy as a default dependency for a small guard tool.

The base package has no third-party runtime dependency. PyTorch is optional and is only needed by
the current compute worker:

```bash
python -m pip install -e ".[torch]"
```

The public backend selector currently supports:

```bash
gpu-holder guard --backend torch
gpu-holder doctor --backend torch
```

Keeping this selector explicit makes the future `driver` backend an additive implementation change
instead of a CLI redesign.

## Preferred Default Backend

The preferred default backend is:

```text
ctypes + libcuda.so.1 + embedded conservative PTX
```

This means:

- use the NVIDIA Driver API directly through `ctypes`
- load `libcuda.so.1`, which is provided by the installed NVIDIA driver
- embed a small PTX kernel in the package
- JIT the PTX through the driver at runtime
- avoid PyTorch, CUDA Toolkit, `nvcc`, CuPy, Numba, and local compilation

This is the best fit for a small CLI because it depends on the same driver stack that is already
required to run CUDA workloads.

## Compatibility Boundary

The Driver API backend should be documented as:

> Works on Linux machines with an NVIDIA driver and accessible CUDA devices.

It should not be documented as "works on every machine".

Expected to work:

- Linux training hosts with NVIDIA GPUs
- containers with `/dev/nvidia*` devices and driver libraries mounted correctly
- environments without PyTorch or CUDA Toolkit installed
- machines where `nvidia-smi` and `libcuda.so.1` can see the target devices

Not expected to work:

- machines without NVIDIA GPUs
- machines with broken or missing NVIDIA drivers
- containers that do not expose GPU devices or driver libraries
- macOS
- AMD, Intel, or other non-NVIDIA GPUs
- Windows without a separate `nvcuda.dll` implementation path

Driver age also matters. Embedded PTX must use a conservative PTX version and simple kernels so old
but still common data-center drivers can JIT it.

## Implementation Rules

A Driver API backend should follow these rules:

1. Load `libcuda.so.1` first. Do not depend on the development symlink `libcuda.so`.
2. Treat `CUDA_VISIBLE_DEVICES`, physical indexes, and GPU UUIDs carefully. Do not assume logical
   index `0` always means physical GPU `0`.
3. Run a startup smoke test before claiming the backend is usable:
   - initialize the driver
   - enumerate devices
   - create a context on one selected GPU
   - load the embedded PTX
   - launch one tiny kernel
4. Keep the policy layer backend-neutral. Backend code should only implement worker execution.
5. Keep the PyTorch backend as a fallback while the Driver API backend matures.
6. Surface clear diagnostics in `gpu-holder doctor` instead of failing later inside a worker.

## Acceptance Criteria

Before making the Driver API backend the default, verify:

- `gpu-holder doctor` reports driver library, GPU count, and PTX smoke-test status
- `gpu-holder guard --once` can start and stop one worker cleanly
- foreground shutdown exits promptly with no lingering worker processes
- `CUDA_VISIBLE_DEVICES` remapping is covered by tests or a documented manual check
- the PyTorch fallback remains available for users who already rely on it
