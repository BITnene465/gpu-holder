from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time


@dataclass(frozen=True)
class PauseState:
    active: bool
    path: Path
    pause_until: float | None = None


@dataclass(frozen=True)
class DisabledGpuState:
    active: set[int]
    path: Path
    disabled_until_by_gpu: dict[int, float]


def parse_duration_seconds(raw: str) -> float:
    value = raw.strip().lower()
    if not value:
        raise ValueError("duration cannot be empty")
    suffix = value[-1]
    multipliers = {
        "s": 1.0,
        "m": 60.0,
        "h": 60.0 * 60.0,
        "d": 24.0 * 60.0 * 60.0,
    }
    if suffix in multipliers:
        number = value[:-1]
        multiplier = multipliers[suffix]
    else:
        number = value
        multiplier = 1.0
    try:
        seconds = float(number) * multiplier
    except ValueError as exc:
        raise ValueError(f"invalid duration: {raw}") from exc
    if seconds <= 0:
        raise ValueError("duration must be greater than zero")
    return seconds


def pause_file_for(state_dir: str | Path) -> Path:
    return Path(state_dir).expanduser() / "pause"


def pause_guard(
    state_dir: str | Path,
    *,
    duration_seconds: float | None = None,
    now: float | None = None,
) -> Path:
    pause_file = pause_file_for(state_dir)
    if duration_seconds is None:
        pause_file.parent.mkdir(parents=True, exist_ok=True)
        pause_file.write_text("paused\n", encoding="utf-8")
        return pause_file
    timestamp = time.time() if now is None else float(now)
    payload = {
        "paused": True,
        "created_at": timestamp,
        "pause_until": timestamp + float(duration_seconds),
    }
    write_text_atomically(pause_file, json.dumps(payload, indent=2) + "\n")
    return pause_file


def resume_guard(state_dir: str | Path) -> Path:
    pause_file = pause_file_for(state_dir)
    try:
        pause_file.unlink()
    except FileNotFoundError:
        pass
    return pause_file


def read_pause_state(state_dir: str | Path, *, now: float | None = None) -> PauseState:
    return read_pause_state_file(pause_file_for(state_dir), now=now)


def read_pause_state_file(path: str | Path, *, now: float | None = None) -> PauseState:
    pause_file = Path(path).expanduser()
    timestamp = time.time() if now is None else float(now)
    try:
        raw = pause_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return PauseState(active=False, path=pause_file)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return PauseState(active=True, path=pause_file)
    if not isinstance(payload, dict):
        return PauseState(active=True, path=pause_file)
    pause_until_raw = payload.get("pause_until")
    try:
        pause_until = None if pause_until_raw is None else float(pause_until_raw)
    except (TypeError, ValueError):
        return PauseState(active=True, path=pause_file)
    if pause_until is not None and pause_until <= timestamp:
        try:
            pause_file.unlink()
        except FileNotFoundError:
            pass
        return PauseState(active=False, path=pause_file)
    return PauseState(active=True, path=pause_file, pause_until=pause_until)


def disabled_gpus_file_for(state_dir: str | Path) -> Path:
    return Path(state_dir).expanduser() / "disabled_gpus.json"


def read_disabled_gpus(state_dir: str | Path) -> set[int]:
    return read_disabled_gpu_state(state_dir).active


def read_disabled_gpu_state(
    state_dir: str | Path,
    *,
    now: float | None = None,
) -> DisabledGpuState:
    path = disabled_gpus_file_for(state_dir)
    timestamp = time.time() if now is None else float(now)
    entries = _read_disabled_gpu_entries(path)
    active_entries = {
        gpu_index: disabled_until
        for gpu_index, disabled_until in entries.items()
        if disabled_until is None or disabled_until > timestamp
    }
    if entries and active_entries != entries:
        _write_disabled_gpu_entries(path, active_entries)
    return DisabledGpuState(
        active=set(active_entries),
        path=path,
        disabled_until_by_gpu={
            gpu_index: disabled_until
            for gpu_index, disabled_until in active_entries.items()
            if disabled_until is not None
        },
    )


def _read_disabled_gpu_entries(path: Path) -> dict[int, float | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, list):
        return _disabled_entries_from_sequence(payload)
    if isinstance(payload, dict):
        raw_entries = payload.get("disabled_gpus", payload.get("gpus", []))
        if isinstance(raw_entries, dict):
            return _disabled_entries_from_mapping(raw_entries)
        if isinstance(raw_entries, list):
            return _disabled_entries_from_sequence(raw_entries)
    return {}


def _disabled_entries_from_sequence(items: list[object]) -> dict[int, float | None]:
    result: dict[int, float | None] = {}
    for item in items:
        parsed = _parse_disabled_gpu_entry(item)
        if parsed is None:
            continue
        gpu_index, disabled_until = parsed
        result[gpu_index] = disabled_until
    return result


def _disabled_entries_from_mapping(items: dict[object, object]) -> dict[int, float | None]:
    result: dict[int, float | None] = {}
    for raw_gpu_index, raw_entry in items.items():
        try:
            gpu_index = int(raw_gpu_index)
        except (TypeError, ValueError):
            continue
        if gpu_index < 0:
            continue
        disabled_until = _parse_disabled_until(raw_entry)
        result[gpu_index] = disabled_until
    return result


def _parse_disabled_gpu_entry(item: object) -> tuple[int, float | None] | None:
    if isinstance(item, dict):
        raw_index = item.get("index", item.get("gpu_index"))
        disabled_until = _parse_disabled_until(item.get("disabled_until"))
    else:
        raw_index = item
        disabled_until = None
    try:
        gpu_index = int(raw_index)
    except (TypeError, ValueError):
        return None
    if gpu_index < 0:
        return None
    return gpu_index, disabled_until


def _parse_disabled_until(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def write_disabled_gpus(state_dir: str | Path, gpus: set[int]) -> Path:
    path = disabled_gpus_file_for(state_dir)
    entries = {int(gpu_index): None for gpu_index in gpus if int(gpu_index) >= 0}
    return _write_disabled_gpu_entries(path, entries)


def _write_disabled_gpu_entries(path: Path, entries: dict[int, float | None]) -> Path:
    if any(disabled_until is not None for disabled_until in entries.values()):
        payload: object = {
            "version": 1,
            "disabled_gpus": [
                {"index": gpu_index, "disabled_until": disabled_until}
                for gpu_index, disabled_until in sorted(entries.items())
            ],
        }
    else:
        payload = sorted(entries)
    write_text_atomically(path, json.dumps(payload, indent=2) + "\n")
    return path


def _active_disabled_gpu_entries(
    state_dir: str | Path,
    *,
    now: float | None = None,
) -> dict[int, float | None]:
    state = read_disabled_gpu_state(state_dir, now=now)
    return {
        gpu_index: state.disabled_until_by_gpu.get(gpu_index)
        for gpu_index in state.active
    }


def write_text_atomically(path: str | Path, payload: str) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_file.write(payload)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_path = Path(tmp_file.name)
        tmp_path.replace(target)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        raise
    return target


def disable_gpu(
    state_dir: str | Path,
    gpu_index: int,
    *,
    duration_seconds: float | None = None,
    now: float | None = None,
) -> Path:
    timestamp = time.time() if now is None else float(now)
    entries = _active_disabled_gpu_entries(state_dir, now=timestamp)
    disabled_until = None
    if duration_seconds is not None:
        disabled_until = timestamp + float(duration_seconds)
    entries[int(gpu_index)] = disabled_until
    return _write_disabled_gpu_entries(disabled_gpus_file_for(state_dir), entries)


def enable_gpu(state_dir: str | Path, gpu_index: int) -> Path:
    entries = _active_disabled_gpu_entries(state_dir)
    entries.pop(int(gpu_index), None)
    return _write_disabled_gpu_entries(disabled_gpus_file_for(state_dir), entries)
