from __future__ import annotations

from gpu_holder.reasons import reason_reference


EXPECTED_REASONS = {
    "paused",
    "gpu_disabled",
    "thermal_limit",
    "process_grace",
    "protected_process",
    "low_util_emergency_with_busy_process",
    "low_util_emergency",
    "busy_process",
    "insufficient_free_memory",
    "below_target",
    "machine_target_met",
    "target_met",
    "max_held_gpus_exceeded",
}


def test_reason_reference_documents_all_policy_reasons() -> None:
    reasons = {item["reason"]: item for item in reason_reference()}

    assert set(reasons) == EXPECTED_REASONS
    assert reasons["thermal_limit"]["actions"] == ["release", "standby"]
    assert "cool" in str(reasons["thermal_limit"]["operator_guidance"]).lower()
    assert reasons["protected_process"]["actions"] == ["release", "standby"]
    assert "protected_process_patterns" in str(reasons["protected_process"]["meaning"])
    assert reasons["low_util_emergency_with_busy_process"]["actions"] == ["assist"]
    assert "assist memory" in str(
        reasons["low_util_emergency_with_busy_process"]["operator_guidance"]
    )


def test_reason_reference_priorities_are_unique_and_ordered() -> None:
    priorities = [int(item["priority"]) for item in reason_reference()]

    assert priorities == sorted(priorities)
    assert len(priorities) == len(set(priorities))
