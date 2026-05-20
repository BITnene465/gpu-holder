from __future__ import annotations

from gpu_holder.cli import (
    Config,
    GpuProcess,
    GpuSnapshot,
    child_args,
    decide,
    format_gpus,
    parse_gpus,
    parse_ratio,
    resolve_memory_ratio,
)


def test_ratio_options_accept_new_float_and_legacy_percent() -> None:
    assert parse_ratio("0.2") == 0.2
    assert parse_ratio("20%") == 0.2
    assert parse_ratio("75") == 0.75
    assert parse_ratio("75%") == 0.75


def test_child_args_emit_normalized_ratio_values() -> None:
    config = Config(gpus=parse_gpus("0,1,2,3,4,5,6,7"), target_util=parse_ratio("75%"), mem=parse_ratio("20%"))
    args = child_args(config)

    assert args[args.index("--gpus") + 1] == "0-7"
    assert args[args.index("--target-util") + 1] == "0.75"
    assert args[args.index("--mem") + 1] == "0.2"


def test_gpu_option_accepts_ranges_and_mixed_lists() -> None:
    assert parse_gpus("all") == "all"
    assert parse_gpus("0-7") == tuple(range(8))
    assert parse_gpus("0-3,6,7") == (0, 1, 2, 3, 6, 7)
    assert parse_gpus("0,1,1,2") == (0, 1, 2)
    assert format_gpus((0, 1, 2, 3, 6, 7)) == "0-3,6-7"


def test_memory_ratio_resolves_against_total_and_free_memory() -> None:
    total = 80 * 1024**3
    free = 10 * 1024**3
    reserve = 2 * 1024**3

    assert resolve_memory_ratio(0.2, total=total, free=free, reserve=reserve) == 8 * 1024**3


def test_running_holder_uses_release_margin_above_target() -> None:
    config = Config(target_util=0.75)
    gpu = GpuSnapshot(
        index=0,
        uuid="gpu-0",
        name="GPU 0",
        utilization=76,
        memory_total=80 * 1024**3,
        memory_used=10 * 1024**3,
        memory_free=70 * 1024**3,
        temperature=None,
        processes=[GpuProcess(pid=123, used_memory=4 * 1024**3, name="gpu-holder", is_holder=True)],
    )

    assert decide([gpu], config, {})[0].reason == "target_margin"

    gpu.utilization = 79
    assert decide([gpu], config, {})[0].reason == "target_met"
