from __future__ import annotations

from dataclasses import dataclass
import re


_BINARY_UNITS = {
    "b": 1,
    "kib": 1024,
    "kb": 1024,
    "mib": 1024**2,
    "mb": 1024**2,
    "gib": 1024**3,
    "gb": 1024**3,
    "tib": 1024**4,
    "tb": 1024**4,
}


@dataclass(frozen=True)
class MemorySpec:
    raw: str
    percent: float | None = None
    bytes_value: int | None = None

    def resolve(self, total_bytes: int, *, reserve_bytes: int = 0) -> int:
        if total_bytes <= 0:
            raise ValueError("total_bytes must be positive")
        available = max(0, int(total_bytes) - int(reserve_bytes))
        if self.percent is not None:
            return max(0, min(available, int(total_bytes * self.percent / 100.0)))
        if self.bytes_value is not None:
            return max(0, min(available, int(self.bytes_value)))
        raise ValueError(f"invalid memory spec: {self.raw!r}")


def parse_memory_spec(raw: str) -> MemorySpec:
    text = str(raw).strip()
    if not text:
        raise ValueError("memory spec cannot be empty")
    if text.endswith("%"):
        value = float(text[:-1].strip())
        if value < 0 or value > 100:
            raise ValueError("memory percent must be between 0 and 100")
        return MemorySpec(raw=text, percent=value)

    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)", text)
    if not match:
        raise ValueError(f"invalid memory spec: {raw!r}")
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit not in _BINARY_UNITS:
        raise ValueError(f"unknown memory unit: {unit!r}")
    return MemorySpec(raw=text, bytes_value=int(value * _BINARY_UNITS[unit]))


def format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(amount)}B"
            return f"{amount:.1f}{unit}"
        amount /= 1024.0
