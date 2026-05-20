from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import time
from typing import Any


def write_event(
    path: Path,
    event_type: str,
    *,
    max_bytes: int | None = None,
    backup_count: int = 3,
    **payload: object,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": time.time(),
        "type": event_type,
        **payload,
    }
    line = json.dumps(event, sort_keys=True) + "\n"
    if max_bytes is not None:
        _rotate_if_needed(
            path,
            incoming_bytes=len(line.encode("utf-8")),
            max_bytes=max_bytes,
            backup_count=backup_count,
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def read_events(
    path: Path,
    *,
    limit: int | None = None,
    event_types: set[str] | None = None,
    gpu_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    if limit is not None and limit < 1:
        return []
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        return []
    with handle:
        if event_types or gpu_indices:
            events = _parse_event_lines(handle)
            return _limit_events(
                filter_events(events, event_types=event_types, gpu_indices=gpu_indices),
                limit=limit,
            )
        lines = deque(handle, maxlen=limit)
    return _parse_event_lines(lines)


def read_events_since(path: Path, *, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    try:
        current_size = path.stat().st_size
    except FileNotFoundError:
        return [], 0
    if current_size < offset:
        offset = 0
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(max(0, int(offset)))
        lines = handle.readlines()
        next_offset = handle.tell()
    return _parse_event_lines(lines), next_offset


def filter_events(
    events: list[dict[str, Any]],
    *,
    event_types: set[str] | None = None,
    gpu_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if _event_matches(event, event_types=event_types, gpu_indices=gpu_indices)
    ]


def _event_matches(
    event: dict[str, Any],
    *,
    event_types: set[str] | None,
    gpu_indices: set[int] | None,
) -> bool:
    if event_types and str(event.get("type")) not in event_types:
        return False
    if gpu_indices is None:
        return True
    try:
        gpu_index = int(event["gpu_index"])
    except (KeyError, TypeError, ValueError):
        return False
    return gpu_index in gpu_indices


def _limit_events(events: list[dict[str, Any]], *, limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return events
    return events[-int(limit) :]


def _parse_event_lines(lines: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in lines:
        if not str(line).strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _rotate_if_needed(path: Path, *, incoming_bytes: int, max_bytes: int, backup_count: int) -> None:
    if max_bytes <= 0 or not path.exists():
        return
    try:
        current_size = path.stat().st_size
    except OSError:
        return
    if current_size + incoming_bytes <= max_bytes:
        return
    if backup_count <= 0:
        path.unlink(missing_ok=True)
        return
    oldest = path.with_name(f"{path.name}.{backup_count}")
    oldest.unlink(missing_ok=True)
    for index in range(backup_count - 1, 0, -1):
        source = path.with_name(f"{path.name}.{index}")
        if source.exists():
            source.replace(path.with_name(f"{path.name}.{index + 1}"))
    path.replace(path.with_name(f"{path.name}.1"))
