from __future__ import annotations

import pytest

import gpu_holder.worker as worker_mod
from gpu_holder.worker import (
    WorkerProcess,
    WorkerStartError,
    _jittered_burst_seconds,
    _next_program,
    _program_sequence,
    _sleep_seconds_for_duty,
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
