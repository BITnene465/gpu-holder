from __future__ import annotations

from gpu_holder.metrics import prometheus_metrics


def test_prometheus_metrics_reports_missing_status() -> None:
    rendered = prometheus_metrics(None, status_error="no status file")

    assert "gpu_holder_status_available 0" in rendered
    assert 'gpu_holder_status_error{error="no status file"} 1' in rendered


def test_prometheus_metrics_exports_machine_and_gpu_status() -> None:
    rendered = prometheus_metrics(
        {
            "timestamp": 100.0,
            "paused": False,
            "config": {"target_util": 75, "profile": "quota"},
            "machine": {
                "gpu_count": 1,
                "average_utilization": 42.0,
                "policy_average_utilization": 60.0,
                "policy_window_seconds": 3600.0,
                "policy_window_sample_count": 120,
                "policy_window_coverage_seconds": 3500.0,
                "policy_window_coverage_ratio": 3500.0 / 3600.0,
                "action_counts": {"hold": 1},
                "owned_worker_count": 1,
                "worker_backoff_count": 0,
                "process_grace_count": 0,
                "thermal_limit_count": 0,
            },
            "gpus": [
                {
                    "index": 0,
                    "name": 'A"100',
                    "utilization": 42,
                    "temperature_celsius": 70,
                    "memory_used": 2,
                    "memory_total": 10,
                    "gpu_disabled": True,
                    "gpu_disabled_until": 170.0,
                    "thermal_blocked": False,
                    "processes": [{"pid": 1, "is_holder": False, "protected_process": True}],
                    "worker": {"pid": 2, "alive": True},
                    "decision": {
                        "action": "hold",
                        "reason": "below_target",
                        "hold_mode": "balanced",
                        "program": "mixed",
                        "duty_cycle": 0.5,
                        "memory_bytes": 4,
                    },
                }
            ],
        },
        now=110.0,
    )

    assert "gpu_holder_status_available 1" in rendered
    assert "gpu_holder_status_age_seconds 10" in rendered
    assert "gpu_holder_target_gap_percent -15" in rendered
    assert "gpu_holder_target_required_remaining_average_percent 600" in rendered
    assert "gpu_holder_target_remaining_window_seconds 100" in rendered
    assert "gpu_holder_target_forecast_ok 0" in rendered
    assert "gpu_holder_target_forecast_at_risk 1" in rendered
    assert "gpu_holder_machine_policy_window_seconds 3600" in rendered
    assert "gpu_holder_machine_policy_window_sample_count 120" in rendered
    assert "gpu_holder_machine_policy_window_coverage_seconds 3500" in rendered
    assert "gpu_holder_machine_policy_window_coverage_ratio 0.972222" in rendered
    assert "gpu_holder_machine_policy_window_ready 1" in rendered
    assert 'gpu_holder_machine_action_count{action="hold"} 1' in rendered
    assert 'gpu_holder_gpu_utilization_percent{gpu="0",name="A\\"100",profile="quota"} 42' in rendered
    assert 'gpu_holder_gpu_disabled{gpu="0",name="A\\"100",profile="quota"} 1' in rendered
    assert 'gpu_holder_gpu_disabled_until_seconds{gpu="0",name="A\\"100",profile="quota"} 170' in rendered
    assert 'gpu_holder_gpu_disabled_remaining_seconds{gpu="0",name="A\\"100",profile="quota"} 60' in rendered
    assert 'gpu_holder_gpu_external_process_count{gpu="0",name="A\\"100",profile="quota"} 1' in rendered
    assert 'gpu_holder_gpu_protected_process_count{gpu="0",name="A\\"100",profile="quota"} 1' in rendered
    assert 'gpu_holder_gpu_decision{action="hold",gpu="0",mode="balanced",name="A\\"100",profile="quota",program="mixed",reason="below_target"} 1' in rendered
