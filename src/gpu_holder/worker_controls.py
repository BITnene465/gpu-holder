from __future__ import annotations

import random


def jittered_burst_seconds(base_seconds: float, jitter: float, *, rng: object | None = None) -> float:
    base = max(0.001, float(base_seconds))
    amount = max(0.0, min(1.0, float(jitter)))
    if amount <= 0:
        return base
    chooser = rng if rng is not None else random
    factor = 1.0 + float(chooser.uniform(-amount, amount))
    return max(0.001, base * factor)


def sleep_seconds_for_duty(*, burst_seconds: float, duty: float) -> float:
    duty = max(0.0, min(1.0, float(duty)))
    if duty <= 0:
        return 1.0
    return max(0.0, float(burst_seconds) * (1.0 - duty) / duty)


def normalize_hold_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"balanced", "memory-only", "compute-only", "assist"}:
        return normalized
    raise ValueError(f"unknown hold mode: {mode!r}")
