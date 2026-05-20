from __future__ import annotations

from gpu_holder.explain import explain_runtime, format_explanation


def test_explain_runtime_reports_missing_status() -> None:
    payload = explain_runtime(status=None, status_error="no status file", events=[])

    assert payload["ok"] is False
    assert payload["findings"][0]["code"] == "status_missing"
    assert payload["history"]["event_count"] == 0


def test_explain_runtime_reports_target_and_gpu_blockers() -> None:
    payload = explain_runtime(
        status={
            "timestamp": 100.0,
            "paused": True,
            "config": {"target_util": 75},
            "machine": {
                "policy_average_utilization": 60,
                "policy_window_seconds": 3600.0,
                "policy_window_sample_count": 2,
                "policy_window_coverage_seconds": 60.0,
                "policy_window_coverage_ratio": 60.0 / 3600.0,
                "worker_backoff_count": 1,
            },
            "disabled_gpus": [2],
            "gpus": [
                {
                    "index": 0,
                    "decision": {"action": "release", "reason": "thermal_limit"},
                    "worker_restart_backoff": {"active": True},
                    "processes": [{"pid": 123, "is_holder": False}],
                },
                {
                    "index": 1,
                    "decision": {"action": "release", "reason": "protected_process"},
                    "processes": [],
                },
                {
                    "index": 2,
                    "decision": {"action": "release", "reason": "busy_process"},
                    "processes": [],
                },
            ],
        },
        status_error=None,
        events=[
            {"timestamp": 101, "type": "worker_start_failed", "gpu_index": 0},
            {"timestamp": 102, "type": "worker_start_backoff", "gpu_index": 0},
        ],
        stale_after=0,
        now=105,
    )

    codes = [finding["code"] for finding in payload["findings"]]
    assert payload["ok"] is True
    assert "paused" in codes
    assert "policy_window_warming_up" in codes
    assert "below_target" in codes
    assert "thermal_limit" in codes
    assert "protected_process" in codes
    assert "busy_process" in codes
    assert "worker_start_backoff" in codes
    assert "worker_start_failures" in codes
    assert "external_processes_present" in codes


def test_format_explanation_renders_guidance_lines() -> None:
    payload = explain_runtime(status=None, status_error="no status file", events=[])

    rendered = "\n".join(format_explanation(payload))

    assert "gpu-holder explain ok=False" in rendered
    assert "finding[critical]: status_missing" in rendered
    assert "guidance:" in rendered


def test_format_explanation_includes_policy_window_health() -> None:
    payload = explain_runtime(
        status={
            "timestamp": 100.0,
            "paused": False,
            "config": {"target_util": 75},
            "machine": {
                "policy_average_utilization": 70.0,
                "policy_window_seconds": 3600.0,
                "policy_window_sample_count": 3,
                "policy_window_coverage_seconds": 120.0,
                "policy_window_coverage_ratio": 120.0 / 3600.0,
            },
            "gpus": [],
        },
        status_error=None,
        events=[],
        stale_after=0,
        now=130.0,
    )

    rendered = "\n".join(format_explanation(payload))

    assert "window=warming_up coverage=3%" in rendered
    assert "quota=recoverable required=75.2%" in rendered
    assert "finding[info]: policy_window_warming_up" in rendered
    assert "avoid treating this as a full-window quota miss yet" in rendered


def test_explain_runtime_reports_quota_forecast_at_risk() -> None:
    payload = explain_runtime(
        status={
            "timestamp": 100.0,
            "paused": False,
            "config": {"target_util": 75},
            "machine": {
                "policy_average_utilization": 40.0,
                "policy_window_seconds": 3600.0,
                "policy_window_sample_count": 50,
                "policy_window_coverage_seconds": 3000.0,
                "policy_window_coverage_ratio": 3000.0 / 3600.0,
            },
            "gpus": [],
        },
        status_error=None,
        events=[],
        stale_after=0,
        now=130.0,
    )

    codes = [finding["code"] for finding in payload["findings"]]

    assert "quota_forecast_at_risk" in codes
    assert payload["status"]["utilization_target_forecast"]["status"] == "at_risk"
