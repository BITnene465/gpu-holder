from __future__ import annotations

import ctypes
from ctypes.util import find_library
from dataclasses import dataclass
from typing import Any
from typing import Callable


DEFAULT_BACKEND = "torch"
DRIVER_BACKEND = "driver"
SUPPORTED_WORKER_BACKENDS = (DEFAULT_BACKEND,)
SUPPORTED_DIAGNOSTIC_BACKENDS = (DEFAULT_BACKEND, DRIVER_BACKEND)


@dataclass(frozen=True)
class BackendCheck:
    name: str
    ok: bool
    detail: str

    def as_payload(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def normalize_backend(raw: str) -> str:
    backend = str(raw).strip().lower()
    if backend not in SUPPORTED_WORKER_BACKENDS:
        supported = ", ".join(SUPPORTED_WORKER_BACKENDS)
        raise ValueError(f"unsupported worker backend: {raw!r}; supported backends: {supported}")
    return backend


def check_backend(backend: str = DEFAULT_BACKEND) -> BackendCheck:
    normalized = normalize_diagnostic_backend(backend)
    if normalized == "torch":
        return check_torch_cuda()
    if normalized == "driver":
        return check_driver_cuda()
    raise AssertionError(f"unhandled backend: {normalized}")


def normalize_diagnostic_backend(raw: str) -> str:
    backend = str(raw).strip().lower()
    if backend not in SUPPORTED_DIAGNOSTIC_BACKENDS:
        supported = ", ".join(SUPPORTED_DIAGNOSTIC_BACKENDS)
        raise ValueError(f"unsupported diagnostic backend: {raw!r}; supported backends: {supported}")
    return backend


def check_torch_cuda() -> BackendCheck:
    try:
        import torch
    except Exception as exc:
        return BackendCheck(name="torch_cuda", ok=False, detail=f"{type(exc).__name__}: {exc}")
    ok = bool(torch.cuda.is_available())
    detail = (
        f"torch={torch.__version__} cuda_available={ok} "
        f"device_count={torch.cuda.device_count() if ok else 0}"
    )
    return BackendCheck(name="torch_cuda", ok=ok, detail=detail)


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


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "The torch backend requires a CUDA-enabled PyTorch build. "
            "Install one for this machine or use: pip install 'gpu-holder[torch]'"
        ) from exc
    return torch
