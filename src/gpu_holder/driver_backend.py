from __future__ import annotations

import ctypes
import random
import time
from typing import Any
from typing import Callable

from .backends import BackendCheck


DEFAULT_CUDA_LIBRARY = "libcuda.so.1"

NOOP_PTX = b"""
.version 3.0
.target sm_30
.address_size 64

.visible .entry gpu_holder_noop()
{
    ret;
}
"""

BURN_PTX = b"""
.version 3.0
.target sm_30
.address_size 64

.visible .entry gpu_holder_spin(.param .u64 cycles)
{
    .reg .pred %keep_spinning;
    .reg .u64 %start;
    .reg .u64 %now;
    .reg .u64 %elapsed;
    .reg .u64 %cycles_value;

    ld.param.u64 %cycles_value, [cycles];
    mov.u64 %start, %clock64;

spin:
    mov.u64 %now, %clock64;
    sub.u64 %elapsed, %now, %start;
    setp.lt.u64 %keep_spinning, %elapsed, %cycles_value;
    @%keep_spinning bra spin;
    ret;
}
"""

MEMORY_CHUNK_BYTES = 256 * 1024 * 1024
SPIN_CYCLES_PER_LAUNCH = 5_000_000


def check_driver_cuda(
    *,
    load_library: Callable[[str], Any] | None = None,
    library_name: str | None = None,
) -> BackendCheck:
    load_library = ctypes.CDLL if load_library is None else load_library
    library_name = library_name or DEFAULT_CUDA_LIBRARY
    try:
        libcuda = load_library(library_name)
    except OSError as exc:
        return BackendCheck(name="driver_cuda", ok=False, detail=f"load_failed library={library_name}: {exc}")

    try:
        cu_init = libcuda.cuInit
        cu_driver_get_version = libcuda.cuDriverGetVersion
        cu_device_get_count = libcuda.cuDeviceGetCount
        cu_device_get = libcuda.cuDeviceGet
        cu_ctx_create = libcuda.cuCtxCreate_v2
        cu_ctx_destroy = libcuda.cuCtxDestroy_v2
        cu_module_load_data = libcuda.cuModuleLoadData
        cu_module_unload = libcuda.cuModuleUnload
        cu_module_get_function = libcuda.cuModuleGetFunction
        cu_launch_kernel = libcuda.cuLaunchKernel
        cu_ctx_synchronize = libcuda.cuCtxSynchronize
    except AttributeError as exc:
        return BackendCheck(name="driver_cuda", ok=False, detail=f"missing_symbol: {exc}")

    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int
    cu_driver_get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cu_driver_get_version.restype = ctypes.c_int
    cu_device_get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cu_device_get_count.restype = ctypes.c_int
    cu_device_get.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cu_device_get.restype = ctypes.c_int
    cu_ctx_create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    cu_ctx_create.restype = ctypes.c_int
    cu_ctx_destroy.argtypes = [ctypes.c_void_p]
    cu_ctx_destroy.restype = ctypes.c_int
    cu_module_load_data.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    cu_module_load_data.restype = ctypes.c_int
    cu_module_unload.argtypes = [ctypes.c_void_p]
    cu_module_unload.restype = ctypes.c_int
    cu_module_get_function.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    cu_module_get_function.restype = ctypes.c_int
    cu_launch_kernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    cu_launch_kernel.restype = ctypes.c_int
    cu_ctx_synchronize.argtypes = []
    cu_ctx_synchronize.restype = ctypes.c_int

    result = int(cu_init(0))
    if result != 0:
        return BackendCheck(name="driver_cuda", ok=False, detail=f"cuInit_failed code={result}")

    driver_version = ctypes.c_int()
    result = int(cu_driver_get_version(ctypes.byref(driver_version)))
    if result != 0:
        return BackendCheck(
            name="driver_cuda",
            ok=False,
            detail=f"cuDriverGetVersion_failed code={result}",
        )

    device_count = ctypes.c_int()
    result = int(cu_device_get_count(ctypes.byref(device_count)))
    if result != 0:
        return BackendCheck(name="driver_cuda", ok=False, detail=f"cuDeviceGetCount_failed code={result}")

    detail_prefix = (
        f"library={library_name} driver_version={driver_version.value} "
        f"device_count={device_count.value}"
    )
    if device_count.value <= 0:
        return BackendCheck(name="driver_cuda", ok=False, detail=detail_prefix)

    smoke_result = _run_ptx_smoke(
        cu_device_get=cu_device_get,
        cu_ctx_create=cu_ctx_create,
        cu_ctx_destroy=cu_ctx_destroy,
        cu_module_load_data=cu_module_load_data,
        cu_module_unload=cu_module_unload,
        cu_module_get_function=cu_module_get_function,
        cu_launch_kernel=cu_launch_kernel,
        cu_ctx_synchronize=cu_ctx_synchronize,
    )
    detail = f"{detail_prefix} {smoke_result}"
    return BackendCheck(name="driver_cuda", ok=smoke_result == "ptx_smoke=ok", detail=detail)


def _run_ptx_smoke(
    *,
    cu_device_get: Any,
    cu_ctx_create: Any,
    cu_ctx_destroy: Any,
    cu_module_load_data: Any,
    cu_module_unload: Any,
    cu_module_get_function: Any,
    cu_launch_kernel: Any,
    cu_ctx_synchronize: Any,
) -> str:
    device = ctypes.c_int()
    result = int(cu_device_get(ctypes.byref(device), 0))
    if result != 0:
        return f"ptx_smoke=cuDeviceGet_failed code={result}"

    context = ctypes.c_void_p()
    result = int(cu_ctx_create(ctypes.byref(context), 0, device.value))
    if result != 0:
        return f"ptx_smoke=cuCtxCreate_failed code={result}"

    module = ctypes.c_void_p()
    try:
        result = int(cu_module_load_data(ctypes.byref(module), NOOP_PTX))
        if result != 0:
            return f"ptx_smoke=cuModuleLoadData_failed code={result}"

        function = ctypes.c_void_p()
        result = int(cu_module_get_function(ctypes.byref(function), module, b"gpu_holder_noop"))
        if result != 0:
            return f"ptx_smoke=cuModuleGetFunction_failed code={result}"

        result = int(cu_launch_kernel(function, 1, 1, 1, 1, 1, 1, 0, None, None, None))
        if result != 0:
            return f"ptx_smoke=cuLaunchKernel_failed code={result}"

        result = int(cu_ctx_synchronize())
        if result != 0:
            return f"ptx_smoke=cuCtxSynchronize_failed code={result}"
        return "ptx_smoke=ok"
    finally:
        if module.value:
            cu_module_unload(module)
        cu_ctx_destroy(context)


def run_driver_worker(
    *,
    gpu_index: int,
    memory_bytes: int,
    duty_cycle: float,
    program: str,
    hold_mode: str,
    burst_seconds: float,
    burst_jitter: float,
    ready_queue: Any,
) -> None:
    del program
    libcuda = ctypes.CDLL(DEFAULT_CUDA_LIBRARY)
    functions = _configure_worker_driver_functions(libcuda)
    _driver_call(functions["cuInit"](0), "cuInit")
    device = ctypes.c_int()
    _driver_call(functions["cuDeviceGet"](ctypes.byref(device), int(gpu_index)), "cuDeviceGet")
    context = ctypes.c_void_p()
    _driver_call(functions["cuCtxCreate"](ctypes.byref(context), 0, device.value), "cuCtxCreate")
    module = ctypes.c_void_p()
    allocations: list[ctypes.c_ulonglong] = []
    try:
        _driver_call(functions["cuModuleLoadData"](ctypes.byref(module), BURN_PTX), "cuModuleLoadData")
        kernel = ctypes.c_void_p()
        _driver_call(
            functions["cuModuleGetFunction"](ctypes.byref(kernel), module, b"gpu_holder_spin"),
            "cuModuleGetFunction",
        )
        mode = _normalize_hold_mode(hold_mode)
        if mode != "compute-only":
            allocations = _allocate_driver_memory(
                cu_mem_alloc=functions["cuMemAlloc"],
                memory_bytes=max(0, int(memory_bytes)),
            )
        _put_ready_message(ready_queue, {"status": "ready"})
        _run_driver_loop(
            cu_launch_kernel=functions["cuLaunchKernel"],
            cu_ctx_synchronize=functions["cuCtxSynchronize"],
            kernel=kernel,
            duty_cycle=duty_cycle,
            hold_mode=mode,
            burst_seconds=burst_seconds,
            burst_jitter=burst_jitter,
        )
    finally:
        for allocation in allocations:
            functions["cuMemFree"](allocation.value)
        if module.value:
            functions["cuModuleUnload"](module)
        functions["cuCtxDestroy"](context)


def _configure_worker_driver_functions(libcuda: Any) -> dict[str, Any]:
    functions = {
        "cuInit": libcuda.cuInit,
        "cuDeviceGet": libcuda.cuDeviceGet,
        "cuCtxCreate": libcuda.cuCtxCreate_v2,
        "cuCtxDestroy": libcuda.cuCtxDestroy_v2,
        "cuModuleLoadData": libcuda.cuModuleLoadData,
        "cuModuleUnload": libcuda.cuModuleUnload,
        "cuModuleGetFunction": libcuda.cuModuleGetFunction,
        "cuLaunchKernel": libcuda.cuLaunchKernel,
        "cuCtxSynchronize": libcuda.cuCtxSynchronize,
        "cuMemAlloc": libcuda.cuMemAlloc_v2,
        "cuMemFree": libcuda.cuMemFree_v2,
    }
    functions["cuInit"].argtypes = [ctypes.c_uint]
    functions["cuInit"].restype = ctypes.c_int
    functions["cuDeviceGet"].argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    functions["cuDeviceGet"].restype = ctypes.c_int
    functions["cuCtxCreate"].argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    functions["cuCtxCreate"].restype = ctypes.c_int
    functions["cuCtxDestroy"].argtypes = [ctypes.c_void_p]
    functions["cuCtxDestroy"].restype = ctypes.c_int
    functions["cuModuleLoadData"].argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    functions["cuModuleLoadData"].restype = ctypes.c_int
    functions["cuModuleUnload"].argtypes = [ctypes.c_void_p]
    functions["cuModuleUnload"].restype = ctypes.c_int
    functions["cuModuleGetFunction"].argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    functions["cuModuleGetFunction"].restype = ctypes.c_int
    functions["cuLaunchKernel"].argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    functions["cuLaunchKernel"].restype = ctypes.c_int
    functions["cuCtxSynchronize"].argtypes = []
    functions["cuCtxSynchronize"].restype = ctypes.c_int
    functions["cuMemAlloc"].argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t]
    functions["cuMemAlloc"].restype = ctypes.c_int
    functions["cuMemFree"].argtypes = [ctypes.c_ulonglong]
    functions["cuMemFree"].restype = ctypes.c_int
    return functions


def _allocate_driver_memory(*, cu_mem_alloc: Any, memory_bytes: int) -> list[ctypes.c_ulonglong]:
    allocations: list[ctypes.c_ulonglong] = []
    remaining = max(0, int(memory_bytes))
    while remaining > 0:
        current = min(MEMORY_CHUNK_BYTES, remaining)
        pointer = ctypes.c_ulonglong()
        _driver_call(cu_mem_alloc(ctypes.byref(pointer), current), "cuMemAlloc")
        allocations.append(pointer)
        remaining -= current
    return allocations


def _run_driver_loop(
    *,
    cu_launch_kernel: Any,
    cu_ctx_synchronize: Any,
    kernel: ctypes.c_void_p,
    duty_cycle: float,
    hold_mode: str,
    burst_seconds: float,
    burst_jitter: float,
) -> None:
    rng = random.SystemRandom()
    base_burst_seconds = max(0.001, float(burst_seconds))
    jitter = max(0.0, min(1.0, float(burst_jitter)))
    duty = max(0.0, min(1.0, float(duty_cycle)))
    while True:
        current_burst_seconds = _jittered_burst_seconds(base_burst_seconds, jitter, rng=rng)
        compute_seconds = current_burst_seconds
        if hold_mode != "memory-only":
            started = time.monotonic()
            while time.monotonic() - started < current_burst_seconds:
                _launch_spin_kernel(cu_launch_kernel=cu_launch_kernel, kernel=kernel)
                _driver_call(cu_ctx_synchronize(), "cuCtxSynchronize")
            compute_seconds = time.monotonic() - started
        sleep_seconds = _sleep_seconds_for_duty(burst_seconds=compute_seconds, duty=duty)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def _launch_spin_kernel(*, cu_launch_kernel: Any, kernel: ctypes.c_void_p) -> None:
    cycles = ctypes.c_ulonglong(SPIN_CYCLES_PER_LAUNCH)
    kernel_params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(cycles), ctypes.c_void_p))
    _driver_call(
        cu_launch_kernel(
            kernel,
            128,
            1,
            1,
            256,
            1,
            1,
            0,
            None,
            ctypes.cast(kernel_params, ctypes.c_void_p),
            None,
        ),
        "cuLaunchKernel",
    )


def _driver_call(result: int, name: str) -> None:
    code = int(result)
    if code != 0:
        raise RuntimeError(f"{name} failed with CUDA driver error code {code}")


def _jittered_burst_seconds(base_seconds: float, jitter: float, *, rng: object | None = None) -> float:
    base = max(0.001, float(base_seconds))
    amount = max(0.0, min(1.0, float(jitter)))
    if amount <= 0:
        return base
    chooser = rng if rng is not None else random
    factor = 1.0 + float(chooser.uniform(-amount, amount))
    return max(0.001, base * factor)


def _sleep_seconds_for_duty(*, burst_seconds: float, duty: float) -> float:
    duty = max(0.0, min(1.0, float(duty)))
    if duty <= 0:
        return 1.0
    return max(0.0, float(burst_seconds) * (1.0 - duty) / duty)


def _normalize_hold_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"balanced", "memory-only", "compute-only", "assist"}:
        return normalized
    raise ValueError(f"unknown hold mode: {mode!r}")


def _put_ready_message(queue: Any, message: dict[str, Any]) -> None:
    try:
        queue.put_nowait(message)
    except Exception:
        pass
