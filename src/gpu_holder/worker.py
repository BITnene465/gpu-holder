from __future__ import annotations

import multiprocessing as mp
import os
import random
import signal
import time
from typing import Any


BASE_PROGRAMS = ("matmul", "conv", "fft", "elementwise")


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
        burst_seconds: float = 0.20,
        burst_jitter: float = 0.0,
    ) -> None:
        self.gpu_index = int(gpu_index)
        self.memory_bytes = int(memory_bytes)
        self.duty_cycle = float(duty_cycle)
        self.program = str(program)
        self.hold_mode = str(hold_mode)
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

    def stop(self, timeout: float = 5.0) -> None:
        if self.process is None:
            return
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=timeout)
        if self.process.is_alive() and self.process.pid is not None:
            os.kill(self.process.pid, signal.SIGKILL)
            self.process.join(timeout=1.0)
        self.process = None


def _worker_entry(
    *,
    gpu_index: int,
    memory_bytes: int,
    duty_cycle: float,
    program: str,
    hold_mode: str,
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
    burst_seconds: float,
    burst_jitter: float,
    ready_queue: mp.Queue[dict[str, Any]],
) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "GPU workers require PyTorch. Install with: pip install 'gpu-holder[cuda]'"
        ) from exc

    torch.cuda.set_device(int(gpu_index))
    mode = _normalize_hold_mode(hold_mode)
    allocate_bytes = 0 if mode == "compute-only" else max(0, int(memory_bytes))
    holder = _allocate_memory(torch=torch, memory_bytes=allocate_bytes, gpu_index=gpu_index)
    programs = _program_sequence(program)
    randomize_program = _normalize_program(program) == "random"
    rng = random.SystemRandom()
    cursor = 0
    base_burst_seconds = max(0.001, float(burst_seconds))
    jitter = max(0.0, min(1.0, float(burst_jitter)))
    duty = max(0.0, min(1.0, float(duty_cycle)))
    _put_ready_message(ready_queue, {"status": "ready"})

    while True:
        current_burst_seconds = _jittered_burst_seconds(base_burst_seconds, jitter, rng=rng)
        sleep_seconds = _sleep_seconds_for_duty(
            burst_seconds=current_burst_seconds,
            duty=duty,
        )
        if mode != "memory-only":
            selected = _next_program(
                programs,
                cursor=cursor,
                randomize=randomize_program,
                rng=rng,
            )
            cursor += 1
            started = time.monotonic()
            while time.monotonic() - started < current_burst_seconds:
                _run_program(torch=torch, name=selected, gpu_index=gpu_index)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if holder is not None and len(holder) == -1:
            break


def _allocate_memory(*, torch: object, memory_bytes: int, gpu_index: int) -> list[object]:
    if memory_bytes <= 0:
        return []
    chunk_bytes = 256 * 1024 * 1024
    chunks: list[object] = []
    remaining = int(memory_bytes)
    while remaining > 0:
        current = min(chunk_bytes, remaining)
        elements = max(1, current // 2)
        chunks.append(torch.empty(elements, dtype=torch.float16, device=f"cuda:{gpu_index}"))
        remaining -= current
    return chunks


def _program_sequence(program: str) -> tuple[str, ...]:
    normalized = _normalize_program(program)
    if normalized in {"mixed", "random"}:
        return BASE_PROGRAMS
    if normalized in BASE_PROGRAMS:
        return (normalized,)
    sequence = tuple(part.strip().lower() for part in normalized.split(",") if part.strip())
    if not sequence:
        raise ValueError("program sequence must not be empty")
    for item in sequence:
        if item not in BASE_PROGRAMS:
            raise ValueError(f"unknown worker program: {item!r}")
    return sequence


def _next_program(
    programs: tuple[str, ...],
    *,
    cursor: int,
    randomize: bool,
    rng: object | None = None,
) -> str:
    if not programs:
        raise ValueError("program sequence must not be empty")
    if randomize:
        chooser = rng if rng is not None else random
        return str(chooser.choice(programs))
    return programs[int(cursor) % len(programs)]


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


def _normalize_program(program: str) -> str:
    return str(program).strip().lower()


def _normalize_hold_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"balanced", "memory-only", "compute-only", "assist"}:
        return normalized
    raise ValueError(f"unknown hold mode: {mode!r}")


def _run_program(*, torch: object, name: str, gpu_index: int) -> None:
    device = f"cuda:{gpu_index}"
    if name == "matmul":
        a = torch.randn((2048, 2048), dtype=torch.float16, device=device)
        b = torch.randn((2048, 2048), dtype=torch.float16, device=device)
        torch.mm(a, b)
    elif name == "conv":
        x = torch.randn((16, 32, 128, 128), dtype=torch.float16, device=device)
        weight = torch.randn((32, 32, 3, 3), dtype=torch.float16, device=device)
        torch.nn.functional.conv2d(x, weight, padding=1)
    elif name == "fft":
        x = torch.randn((1024, 1024), dtype=torch.float32, device=device)
        torch.fft.fft2(x)
    elif name == "elementwise":
        x = torch.randn((16 * 1024 * 1024,), dtype=torch.float16, device=device)
        y = torch.sin(x) * torch.cos(x)
        y.sum()
    else:
        raise ValueError(f"unknown program: {name!r}")
    torch.cuda.synchronize(device)


def _put_ready_message(queue: mp.Queue[dict[str, Any]], message: dict[str, Any]) -> None:
    try:
        queue.put_nowait(message)
    except Exception:
        pass
