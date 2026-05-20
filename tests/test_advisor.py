from __future__ import annotations

from dataclasses import replace

from gpu_holder.advisor import config_from_status_config, profile_comparison, tuning_advice
from gpu_holder.config import CONFIG_PROFILES, GuardConfig
from gpu_holder.units import parse_memory_spec


def test_tuning_advice_scores_default_config_as_pass() -> None:
    payload = tuning_advice(GuardConfig())

    assert payload["ok"] is True
    assert payload["grade"] == "pass"
    assert payload["score"] >= 90
    assert payload["summary"]["scenario_count"] == 5
    assert "timeline" not in payload["simulation"]["scenarios"][0]
    assert {check["name"] for check in payload["checks"]} >= {
        "target_util_floor",
        "idle_emergency_intervention",
        "busy_process_yield",
        "thermal_yield",
        "memory_hold_size",
        "program_diversity",
        "sample_cadence",
    }


def test_tuning_advice_warns_for_low_target_and_single_program() -> None:
    config = replace(GuardConfig(), target_util=60, program="matmul", sample_interval=20)

    payload = tuning_advice(config)

    assert payload["ok"] is True
    assert payload["grade"] == "warn"
    assert payload["score"] < 90
    warnings = {check["name"] for check in payload["checks"] if check["status"] == "warn"}
    assert {"target_util_floor", "program_diversity", "sample_cadence"} <= warnings
    assert any("target-util" in item for item in payload["recommendations"])


def test_tuning_advice_fails_unsafe_memory_hold() -> None:
    config = replace(GuardConfig(), mem=parse_memory_spec("90%"))

    payload = tuning_advice(config)

    assert payload["ok"] is False
    assert payload["grade"] == "fail"
    failures = {check["name"] for check in payload["checks"] if check["status"] == "fail"}
    assert "memory_hold_size" in failures


def test_profile_comparison_scores_all_builtin_profiles() -> None:
    payload = profile_comparison()

    assert payload["recommended_profile"] in CONFIG_PROFILES
    assert {item["profile"] for item in payload["profiles"]} == set(CONFIG_PROFILES)
    assert all("score" in item for item in payload["profiles"])
    assert all("grade" in item for item in payload["profiles"])
    assert all("description" in item for item in payload["profiles"])
    assert payload["profiles"][0]["profile"] == payload["recommended_profile"]


def test_config_from_status_config_round_trips_payload() -> None:
    source = replace(
        GuardConfig(),
        gpus=(0, 2),
        target_util=80,
        mem=parse_memory_spec("10%"),
        program="random",
    )

    config = config_from_status_config(
        {
            "gpus": [0, 2],
            "target_util": 80,
            "mem": "10%",
            "program": "random",
            "state_dir": "/tmp/gpu-holder",
        }
    )

    assert config.gpus == source.gpus
    assert config.target_util == source.target_util
    assert config.mem.raw == source.mem.raw
    assert config.program == source.program
    assert str(config.state_dir) == "/tmp/gpu-holder"


def test_config_from_status_config_rejects_missing_config() -> None:
    try:
        config_from_status_config(None)
    except ValueError as exc:
        assert "status config is unavailable" in str(exc)
    else:
        raise AssertionError("missing config must raise ValueError")
