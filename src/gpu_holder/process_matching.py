from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Iterable

from .models import GpuProcess


def normalize_process_patterns(patterns: object) -> tuple[str, ...]:
    if not isinstance(patterns, (list, tuple)):
        return ()
    normalized: list[str] = []
    for pattern in patterns:
        text = str(pattern).strip().lower()
        if text:
            normalized.append(text)
    return tuple(normalized)


def matched_protected_process_pattern(
    process: GpuProcess,
    patterns: object,
) -> str | None:
    protected_patterns = normalize_process_patterns(patterns)
    if process.is_holder or not protected_patterns:
        return None
    name = process.name.strip().lower()
    if not name:
        return None
    for pattern in protected_patterns:
        if fnmatchcase(name, pattern):
            return pattern
    return None


def has_protected_process(
    processes: Iterable[GpuProcess],
    patterns: object,
) -> bool:
    return any(
        matched_protected_process_pattern(process, patterns) is not None
        for process in processes
    )
