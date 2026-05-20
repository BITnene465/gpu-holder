from __future__ import annotations

from gpu_holder.models import GpuProcess
from gpu_holder.process_matching import (
    has_protected_process,
    matched_protected_process_pattern,
    normalize_process_patterns,
)


def test_normalize_process_patterns_accepts_lists_and_tuples() -> None:
    assert normalize_process_patterns([" Python*Train.py ", "", "VLLM*"]) == (
        "python*train.py",
        "vllm*",
    )
    assert normalize_process_patterns(("GPU-HOLDER",)) == ("gpu-holder",)
    assert normalize_process_patterns("gpu-holder") == ()
    assert normalize_process_patterns(None) == ()


def test_matched_protected_process_pattern_is_case_insensitive_glob() -> None:
    process = GpuProcess(pid=1234, used_memory_bytes=1024, name="Python train.py")

    assert (
        matched_protected_process_pattern(process, ["python*train.py", "vllm*"])
        == "python*train.py"
    )


def test_matched_protected_process_pattern_ignores_holder_and_empty_names() -> None:
    patterns = ["gpu-holder", "python*"]

    assert (
        matched_protected_process_pattern(
            GpuProcess(pid=1, used_memory_bytes=1024, name="gpu-holder", is_holder=True),
            patterns,
        )
        is None
    )
    assert (
        matched_protected_process_pattern(
            GpuProcess(pid=2, used_memory_bytes=1024, name=" "),
            patterns,
        )
        is None
    )


def test_has_protected_process_checks_any_non_holder_process() -> None:
    processes = [
        GpuProcess(pid=1, used_memory_bytes=1024, name="gpu-holder", is_holder=True),
        GpuProcess(pid=2, used_memory_bytes=1024, name="python train.py"),
    ]

    assert has_protected_process(processes, ["python*train.py"]) is True
    assert has_protected_process(processes, ["vllm*"]) is False
