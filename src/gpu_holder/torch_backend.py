from __future__ import annotations

import random
import time
from typing import Any
from typing import Callable

from .backends import require_torch


BASE_PROGRAMS = ("matmul", "conv", "fft", "elementwise")
MATMUL_SIZE = 8192
_PROGRAM_CACHE: dict[tuple[str, str], tuple[object, ...]] = {}


def run_torch_worker(
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
    torch = require_torch()
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
        compute_seconds = current_burst_seconds
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
            compute_seconds = time.monotonic() - started
        sleep_seconds = _sleep_seconds_for_duty(
            burst_seconds=compute_seconds,
            duty=duty,
        )
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
        a, b, out = _cached_program_tensors(
            torch=torch,
            key=(device, name),
            factory=lambda: (
                torch.randn((MATMUL_SIZE, MATMUL_SIZE), dtype=torch.float16, device=device),
                torch.randn((MATMUL_SIZE, MATMUL_SIZE), dtype=torch.float16, device=device),
                torch.empty((MATMUL_SIZE, MATMUL_SIZE), dtype=torch.float16, device=device),
            ),
        )
        torch.mm(a, b, out=out)
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


def _cached_program_tensors(
    *,
    torch: object,
    key: tuple[str, str],
    factory: Callable[[], tuple[object, ...]],
) -> tuple[object, ...]:
    del torch
    cached = _PROGRAM_CACHE.get(key)
    if cached is None:
        cached = factory()
        _PROGRAM_CACHE[key] = cached
    return cached


def _put_ready_message(queue: Any, message: dict[str, Any]) -> None:
    try:
        queue.put_nowait(message)
    except Exception:
        pass

