from __future__ import annotations

import ctypes
from ctypes.util import find_library
from typing import Any
from typing import Callable

from .backends import BackendCheck


NOOP_PTX = b"""
.version 3.0
.target sm_30
.address_size 64

.visible .entry gpu_holder_noop()
{
    ret;
}
"""


def check_driver_cuda(
    *,
    load_library: Callable[[str], Any] | None = None,
    library_name: str | None = None,
) -> BackendCheck:
    load_library = ctypes.CDLL if load_library is None else load_library
    library_name = library_name or find_library("cuda") or "libcuda.so.1"
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
