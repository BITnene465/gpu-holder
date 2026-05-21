from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from typing import Any

from .backends import DEFAULT_BACKEND
from .backends import normalize_backend
from .torch_backend import run_torch_worker


class WorkerStartError(RuntimeError):
    pass


class WorkerProcess:
    def __init__(
        self,
        *,
        gpu_index: int,
        memory_bytes: int,
        duty_cycle: float,
        program: str,
        hold_mode: str,
        backend: str = DEFAULT_BACKEND,
        burst_seconds: float = 0.20,
        burst_jitter: float = 0.0,
    ) -> None:
        self.gpu_index = int(gpu_index)
        self.memory_bytes = int(memory_bytes)
        self.duty_cycle = float(duty_cycle)
        self.program = str(program)
        self.hold_mode = str(hold_mode)
        self.backend = normalize_backend(backend)
        self.burst_seconds = float(burst_seconds)
        self.burst_jitter = float(burst_jitter)
        self.process: mp.Process | None = None

    @property
    def pid(self) -> int | None:
        if self.process is None:
            return None
        return self.process.pid

    @property
    def exitcode(self) -> int | None:
        if self.process is None:
            return None
        return self.process.exitcode

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def start(self, timeout: float = 10.0) -> None:
        if self.is_alive():
            return
        ready_queue: mp.Queue[dict[str, Any]] = mp.Queue(maxsize=1)
        self.process = mp.Process(
            target=_worker_entry,
            kwargs={
                "gpu_index": self.gpu_index,
                "memory_bytes": self.memory_bytes,
                "duty_cycle": self.duty_cycle,
                "program": self.program,
                "hold_mode": self.hold_mode,
                "backend": self.backend,
                "burst_seconds": self.burst_seconds,
                "burst_jitter": self.burst_jitter,
                "ready_queue": ready_queue,
            },
            daemon=False,
        )
        self.process.start()
        deadline = time.monotonic() + max(0.1, float(timeout))
        while time.monotonic() < deadline:
            if not ready_queue.empty():
                message = ready_queue.get()
                if message.get("status") == "ready":
                    return
                error = message.get("error", "unknown worker startup error")
                self.stop(timeout=1.0)
                raise WorkerStartError(str(error))
            if not self.is_alive():
                self.stop(timeout=1.0)
                raise WorkerStartError("worker exited before reporting readiness")
            time.sleep(0.05)
        if self.is_alive():
            self.stop(timeout=1.0)
            raise WorkerStartError("worker startup timed out before reporting readiness")
        self.stop(timeout=1.0)
        raise WorkerStartError("worker exited before reporting readiness")

    def stop(self, timeout: float = 0.2) -> None:
        if self.process is None:
            return
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=timeout)
        if self.process.is_alive():
            if hasattr(self.process, "kill"):
                self.process.kill()
            elif self.process.pid is not None:
                os.kill(self.process.pid, signal.SIGKILL)
            self.process.join(timeout=0.5)
        self.process = None


def _worker_entry(
    *,
    gpu_index: int,
    memory_bytes: int,
    duty_cycle: float,
    program: str,
    hold_mode: str,
    backend: str,
    burst_seconds: float,
    burst_jitter: float,
    ready_queue: mp.Queue[dict[str, Any]],
) -> None:
    try:
        _worker_main(
            gpu_index=gpu_index,
            memory_bytes=memory_bytes,
            duty_cycle=duty_cycle,
            program=program,
            hold_mode=hold_mode,
            backend=backend,
            burst_seconds=burst_seconds,
            burst_jitter=burst_jitter,
            ready_queue=ready_queue,
        )
    except Exception as exc:
        _put_ready_message(ready_queue, {"status": "error", "error": str(exc)})
        raise


def _worker_main(
    *,
    gpu_index: int,
    memory_bytes: int,
    duty_cycle: float,
    program: str,
    hold_mode: str,
    backend: str,
    burst_seconds: float,
    burst_jitter: float,
    ready_queue: mp.Queue[dict[str, Any]],
) -> None:
    normalized_backend = normalize_backend(backend)
    if normalized_backend != "torch":
        raise RuntimeError(f"unhandled worker backend: {normalized_backend}")
    run_torch_worker(
        gpu_index=gpu_index,
        memory_bytes=memory_bytes,
        duty_cycle=duty_cycle,
        program=program,
        hold_mode=hold_mode,
        burst_seconds=burst_seconds,
        burst_jitter=burst_jitter,
        ready_queue=ready_queue,
    )


def _put_ready_message(queue: mp.Queue[dict[str, Any]], message: dict[str, Any]) -> None:
    try:
        queue.put_nowait(message)
    except Exception:
        pass
