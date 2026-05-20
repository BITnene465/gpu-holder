from __future__ import annotations

from gpu_holder.history import summarize_events


def test_summarize_events_counts_actions_reasons_and_gpu_runtime_events() -> None:
    summary = summarize_events(
        [
            {"timestamp": 1, "type": "controller_start"},
            {
                "timestamp": 2,
                "type": "decision",
                "gpu_index": 0,
                "action": "hold",
                "reason": "below_target",
            },
            {"timestamp": 3, "type": "worker_start", "gpu_index": 0, "pid": 123},
            {
                "timestamp": 4,
                "type": "decision",
                "gpu_index": 0,
                "action": "release",
                "reason": "busy_process",
            },
            {"timestamp": 5, "type": "thermal_limit_start", "gpu_index": 1},
        ]
    )

    assert summary["ok"] is True
    assert summary["event_count"] == 5
    assert summary["duration_seconds"] == 4
    assert summary["event_type_counts"]["decision"] == 2
    assert summary["action_counts"] == {"hold": 1, "release": 1}
    assert summary["reason_counts"] == {"below_target": 1, "busy_process": 1}
    assert summary["gpu_indices"] == [0, 1]
    assert summary["gpus"]["0"]["worker_starts"] == 1
    assert summary["gpus"]["0"]["last_action"] == "release"
    assert summary["gpus"]["0"]["last_reason"] == "busy_process"
    assert summary["gpus"]["1"]["thermal_limit_starts"] == 1


def test_summarize_events_filters_by_since_and_gpu() -> None:
    summary = summarize_events(
        [
            {"timestamp": 1, "type": "decision", "gpu_index": 0, "action": "hold"},
            {"timestamp": 8, "type": "decision", "gpu_index": 1, "action": "release"},
            {"timestamp": 10, "type": "worker_stop", "gpu_index": 0},
        ],
        since_seconds=5,
        now=10,
        gpu_indices={0},
    )

    assert summary["event_count"] == 1
    assert summary["event_type_counts"] == {"worker_stop": 1}
    assert summary["gpu_indices"] == [0]
