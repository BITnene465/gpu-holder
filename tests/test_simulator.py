from __future__ import annotations

import json

import pytest

from gpu_holder.config import GuardConfig
from gpu_holder.simulator import built_in_trace, load_trace, scenario_names, simulate_policy, simulate_scenarios


def test_built_in_simulation_exercises_time_window_and_assist() -> None:
    payload = simulate_policy(built_in_trace(), GuardConfig())

    summary = payload["summary"]
    timeline = payload["timeline"]
    actions = summary["action_counts"]

    assert summary["steps"] == 4
    assert actions["hold"] >= 1
    assert actions["assist"] >= 1
    assert timeline[2]["decisions"][0]["reason"] == "low_util_emergency"
    assert timeline[2]["decisions"][1]["reason"] == "low_util_emergency_with_busy_process"
    assert timeline[2]["machine_policy_average"] > 0
    assert "machine_policy_average" in timeline[2]["decisions"][0]


def test_built_in_scenarios_cover_common_operator_cases() -> None:
    names = scenario_names()

    assert {"mixed", "idle", "busy", "thermal", "quota-risk"} <= set(names)
    assert simulate_policy(built_in_trace("thermal"), GuardConfig())["timeline"][0]["decisions"][0][
        "reason"
    ] == "thermal_limit"
    assert simulate_policy(built_in_trace("busy"), GuardConfig())["timeline"][0]["decisions"][0][
        "reason"
    ] == "busy_process"


def test_simulate_scenarios_runs_every_built_in_scenario() -> None:
    payload = simulate_scenarios(GuardConfig())

    summary = payload["summary"]
    scenarios = payload["scenarios"]

    assert summary["scenario_count"] == len(scenario_names())
    assert set(summary["scenarios"]) == set(scenario_names())
    assert summary["steps"] == sum(item["summary"]["steps"] for item in scenarios)
    assert summary["gpu_decisions"] == sum(
        item["summary"]["steps"] * item["summary"]["gpu_count"] for item in scenarios
    )
    assert {item["scenario"] for item in scenarios} == set(scenario_names())
    assert summary["action_counts"]["hold"] >= 1
    assert summary["action_counts"]["standby"] >= 1


def test_built_in_trace_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="unknown built-in scenario"):
        built_in_trace("unknown")


def test_load_trace_supports_memory_strings(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "timestamp": 0,
                        "gpus": [
                            {
                                "index": 0,
                                "utilization": 10,
                                "memory_total": "80GiB",
                                "memory_used": "2GiB",
                                "processes": [
                                    {
                                        "pid": 123,
                                        "used_memory": "12GiB",
                                        "name": "train.py",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    frames = load_trace(trace_path)

    assert frames[0][0].memory_total_bytes == 80 * 1024**3
    assert frames[0][0].processes[0].used_memory_bytes == 12 * 1024**3


def test_load_trace_reports_missing_required_gpu_field_with_path(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            [
                {
                    "gpus": [
                        {
                            "index": 0,
                        }
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"frames\[0\]\.gpus\[0\]\.utilization is required"):
        load_trace(trace_path)
