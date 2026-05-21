from __future__ import annotations

import ctypes
from ctypes.util import find_library
from typing import Any
from typing import Callable

from .backends import BackendCheck


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
    except AttributeError as exc:
        return BackendCheck(name="driver_cuda", ok=False, detail=f"missing_symbol: {exc}")

    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int
    cu_driver_get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cu_driver_get_version.restype = ctypes.c_int
    cu_device_get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cu_device_get_count.restype = ctypes.c_int

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

    detail = (
        f"library={library_name} driver_version={driver_version.value} "
        f"device_count={device_count.value}"
    )
    return BackendCheck(name="driver_cuda", ok=device_count.value > 0, detail=detail)

