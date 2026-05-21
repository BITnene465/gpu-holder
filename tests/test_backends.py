from __future__ import annotations

from typing import Any

from gpu_holder.backends import normalize_backend
from gpu_holder.backends import normalize_diagnostic_backend
from gpu_holder.driver_backend import check_driver_cuda


class FakeCudaFunction:
    def __init__(self, result: int = 0, value: int | None = None) -> None:
        self.result = result
        self.value = value
        self.argtypes: list[object] = []
        self.restype: object | None = None

    def __call__(self, *args: Any) -> int:
        if self.value is not None and args:
            args[0]._obj.value = self.value
        return self.result


class FakeCudaLibrary:
    def __init__(
        self,
        *,
        init_result: int = 0,
        driver_version_result: int = 0,
        device_count_result: int = 0,
        device_get_result: int = 0,
        ctx_create_result: int = 0,
        module_load_result: int = 0,
        module_get_function_result: int = 0,
        launch_kernel_result: int = 0,
        ctx_synchronize_result: int = 0,
        driver_version: int = 12040,
        device_count: int = 8,
    ) -> None:
        self.cuInit = FakeCudaFunction(init_result)
        self.cuDriverGetVersion = FakeCudaFunction(driver_version_result, driver_version)
        self.cuDeviceGetCount = FakeCudaFunction(device_count_result, device_count)
        self.cuDeviceGet = FakeCudaFunction(device_get_result, 0)
        self.cuCtxCreate_v2 = FakeCudaFunction(ctx_create_result, 1)
        self.cuCtxDestroy_v2 = FakeCudaFunction()
        self.cuModuleLoadData = FakeCudaFunction(module_load_result, 2)
        self.cuModuleUnload = FakeCudaFunction()
        self.cuModuleGetFunction = FakeCudaFunction(module_get_function_result, 3)
        self.cuLaunchKernel = FakeCudaFunction(launch_kernel_result)
        self.cuCtxSynchronize = FakeCudaFunction(ctx_synchronize_result)


def test_normalize_backend_only_accepts_worker_backends() -> None:
    assert normalize_backend(" torch ") == "torch"
    assert normalize_backend(" DRIVER ") == "driver"


def test_normalize_diagnostic_backend_accepts_driver() -> None:
    assert normalize_diagnostic_backend("driver") == "driver"


def test_driver_cuda_check_reports_driver_version_and_device_count() -> None:
    check = check_driver_cuda(
        load_library=lambda name: FakeCudaLibrary(driver_version=12040, device_count=8),
        library_name="libcuda.so.1",
    )

    assert check.name == "driver_cuda"
    assert check.ok is True
    assert "library=libcuda.so.1" in check.detail
    assert "driver_version=12040" in check.detail
    assert "device_count=8" in check.detail
    assert "ptx_smoke=ok" in check.detail


def test_driver_cuda_check_fails_when_library_cannot_load() -> None:
    def fail_load(name: str) -> object:
        raise OSError("not found")

    check = check_driver_cuda(load_library=fail_load, library_name="libcuda.so.1")

    assert check.ok is False
    assert "load_failed" in check.detail


def test_driver_cuda_check_fails_when_no_devices_are_visible() -> None:
    check = check_driver_cuda(
        load_library=lambda name: FakeCudaLibrary(device_count=0),
        library_name="libcuda.so.1",
    )

    assert check.ok is False
    assert "device_count=0" in check.detail


def test_driver_cuda_check_reports_driver_api_error_codes() -> None:
    check = check_driver_cuda(
        load_library=lambda name: FakeCudaLibrary(init_result=100),
        library_name="libcuda.so.1",
    )

    assert check.ok is False
    assert check.detail == "cuInit_failed code=100"


def test_driver_cuda_check_reports_ptx_smoke_failures() -> None:
    check = check_driver_cuda(
        load_library=lambda name: FakeCudaLibrary(module_load_result=209),
        library_name="libcuda.so.1",
    )

    assert check.ok is False
    assert "device_count=8" in check.detail
    assert "ptx_smoke=cuModuleLoadData_failed code=209" in check.detail
