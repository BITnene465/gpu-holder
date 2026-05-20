from __future__ import annotations

import pytest

from gpu_holder.units import format_bytes, parse_memory_spec


def test_percent_memory_resolves_against_total_with_reserve() -> None:
    total = 80 * 1024**3
    reserve = 2 * 1024**3

    assert parse_memory_spec("20%").resolve(total, reserve_bytes=reserve) == 16 * 1024**3


def test_fixed_memory_is_capped_by_reserve() -> None:
    total = 10 * 1024**3
    reserve = 2 * 1024**3

    assert parse_memory_spec("20GiB").resolve(total, reserve_bytes=reserve) == 8 * 1024**3


def test_parse_memory_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        parse_memory_spec("120%")
    with pytest.raises(ValueError):
        parse_memory_spec("ten gb")


def test_format_bytes() -> None:
    assert format_bytes(1024**3) == "1.0GiB"
