from __future__ import annotations

import pytest

import gpu_holder.torch_backend as torch_backend_mod
import gpu_holder.worker as worker_mod
from gpu_holder.torch_backend import (
    MATMUL_SIZE,
    _PROGRAM_CACHE,
    _jittered_burst_seconds,
    _next_program,
    _program_sequence,
    _run_program,
    _sleep_seconds_for_duty,
)
from gpu_holder.worker import (
    WorkerProcess,
    WorkerStartError,
    _worker_main,
)


def test_program_sequence_supports_random_mode() -> None:
    assert _program_sequence("random") == ("matmul", "conv", "fft", "elementwise")


def test_next_program_rotates_mixed_mode() -> None:
    programs = _program_sequence("mixed")

    assert [_next_program(programs, cursor=index, randomize=False) for index in range(5)] == [
        "matmul",
        "conv",
        "fft",
        "elementwise",
        "matmul",
    ]


def test_program_sequence_supports_custom_rotation() -> None:
    assert _program_sequence("matmul, conv,fft") == ("matmul", "conv", "fft")


def test_next_program_random_mode_uses_rng_choice() -> None:
    class FakeRng:
        def choice(self, values):
            assert values == ("matmul", "conv", "fft", "elementwise")
            return "fft"

    assert _next_program(_program_sequence("random"), cursor=0, randomize=True, rng=FakeRng()) == "fft"


def test_program_sequence_rejects_unknown_program() -> None:
    with pytest.raises(ValueError, match="unknown worker program"):
        _program_sequence("unknown")


def test_program_sequence_rejects_empty_custom_rotation() -> None:
    with pytest.raises(ValueError, match="program sequence must not be empty"):
        _program_sequence(" , ")


def test_jittered_burst_seconds_uses_symmetric_jitter() -> None:
    class FakeRng:
        def uniform(self, low: float, high: float) -> float:
            assert low == -0.25
            assert high == 0.25
            return 0.20

    assert _jittered_burst_seconds(0.5, 0.25, rng=FakeRng()) == 0.6
    assert _jittered_burst_seconds(0.5, 0.0, rng=FakeRng()) == 0.5


def test_sleep_seconds_for_duty_preserves_average_duty() -> None:
    assert _sleep_seconds_for_duty(burst_seconds=0.2, duty=0.5) == 0.2
    assert _sleep_seconds_for_duty(burst_seconds=0.2, duty=1.0) == 0.0
    assert _sleep_seconds_for_duty(burst_seconds=0.2, duty=0.0) == 1.0


def test_matmul_program_reuses_large_cached_tensors() -> None:
    class FakeCuda:
        def __init__(self) -> None:
            self.sync_devices: list[str] = []

        def synchronize(self, device: str) -> None:
            self.sync_devices.append(device)

    class FakeTorch:
        float16 = "float16"

        def __init__(self) -> None:
            self.cuda = FakeCuda()
            self.randn_calls: list[tuple[tuple[int, int], str, str]] = []
            self.empty_calls: list[tuple[tuple[int, int], str, str]] = []
            self.mm_calls = 0

        def randn(self, shape, *, dtype: str, device: str):
            self.randn_calls.append((shape, dtype, device))
            return {"shape": shape, "dtype": dtype, "device": device, "kind": "randn"}

        def empty(self, shape, *, dtype: str, device: str):
            self.empty_calls.append((shape, dtype, device))
            return {"shape": shape, "dtype": dtype, "device": device, "kind": "empty"}

        def mm(self, a, b, *, out) -> None:
            assert a["shape"] == (MATMUL_SIZE, MATMUL_SIZE)
            assert b["shape"] == (MATMUL_SIZE, MATMUL_SIZE)
            assert out["shape"] == (MATMUL_SIZE, MATMUL_SIZE)
            self.mm_calls += 1

    fake_torch = FakeTorch()
    _PROGRAM_CACHE.clear()

    _run_program(torch=fake_torch, name="matmul", gpu_index=0)
    _run_program(torch=fake_torch, name="matmul", gpu_index=0)

    assert len(fake_torch.randn_calls) == 2
    assert len(fake_torch.empty_calls) == 1
    assert fake_torch.mm_calls == 2
    assert fake_torch.cuda.sync_devices == ["cuda:0", "cuda:0"]


def test_worker_start_timeout_stops_unready_live_process(monkeypatch) -> None:
    class FakeQueue:
        def __init__(self, maxsize: int) -> None:
            self.maxsize = maxsize

        def empty(self) -> bool:
            return True

    class FakeProcess:
        pid = 12345

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.started = False
            self.terminated = False
            self.exitcode = None

        def start(self) -> None:
            self.started = True

        def is_alive(self) -> bool:
            return self.started and not self.terminated

        def terminate(self) -> None:
            self.terminated = True
            self.exitcode = -15

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

    monkeypatch.setattr(worker_mod.mp, "Queue", FakeQueue)
    monkeypatch.setattr(worker_mod.mp, "Process", FakeProcess)
    monkeypatch.setattr(worker_mod.time, "sleep", lambda seconds: None)
    clock = iter([0.0, 0.01, 0.05, 0.09, 0.11])
    monkeypatch.setattr(worker_mod.time, "monotonic", lambda: next(clock))

    worker = WorkerProcess(
        gpu_index=0,
        memory_bytes=0,
        duty_cycle=0.1,
        program="mixed",
        hold_mode="compute-only",
        burst_seconds=0.3,
        burst_jitter=0.2,
    )

    with pytest.raises(WorkerStartError, match="timed out"):
        worker.start(timeout=0.01)

    assert worker.process is None


def test_worker_stop_kills_process_after_short_terminate_timeout() -> None:
    class SlowProcess:
        pid = 12345

        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.join_timeouts: list[float | None] = []

        def is_alive(self) -> bool:
            return not self.killed

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def join(self, timeout: float | None = None) -> None:
            self.join_timeouts.append(timeout)

    process = SlowProcess()
    worker = WorkerProcess(
        gpu_index=0,
        memory_bytes=0,
        duty_cycle=1.0,
        program="matmul",
        hold_mode="compute-only",
    )
    worker.process = process

    worker.stop(timeout=0.2)

    assert process.terminated is True
    assert process.killed is True
    assert process.join_timeouts == [0.2, 0.5]
    assert worker.process is None


def test_worker_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported worker backend"):
        WorkerProcess(
            gpu_index=0,
            memory_bytes=0,
            duty_cycle=1.0,
            program="matmul",
            hold_mode="compute-only",
            backend="unknown",
        )


def test_worker_main_dispatches_to_torch_backend(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    ready_queue = object()

    def fake_torch_worker_main(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(worker_mod, "run_torch_worker", fake_torch_worker_main)

    _worker_main(
        gpu_index=0,
        memory_bytes=0,
        duty_cycle=1.0,
        program="matmul",
        hold_mode="compute-only",
        backend="torch",
        burst_seconds=0.2,
        burst_jitter=0.0,
        ready_queue=ready_queue,
    )

    assert calls == [
        {
            "gpu_index": 0,
            "memory_bytes": 0,
            "duty_cycle": 1.0,
            "program": "matmul",
            "hold_mode": "compute-only",
            "burst_seconds": 0.2,
            "burst_jitter": 0.0,
            "ready_queue": ready_queue,
        }
    ]


def test_torch_backend_owns_program_implementation() -> None:
    assert torch_backend_mod.BASE_PROGRAMS == ("matmul", "conv", "fft", "elementwise")
