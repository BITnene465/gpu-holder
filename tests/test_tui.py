from __future__ import annotations

import json
from pathlib import Path

import gpu_holder.tui as tui
import pytest

from gpu_holder.state import (
    disable_gpu,
    disabled_gpus_file_for,
    enable_gpu,
    pause_file_for,
    pause_guard,
    parse_duration_seconds,
    read_disabled_gpu_state,
    read_disabled_gpus,
    read_pause_state,
    resume_guard,
    write_disabled_gpus,
)
from gpu_holder.tui import (
    DASHBOARD_HELP_LINES,
    DashboardState,
    dashboard_help_lines,
    filter_dashboard_events,
    format_config_lines,
    format_advice_lines,
    format_dashboard_snapshot,
    format_event_line,
    format_explain_lines,
    format_gpu_detail_lines,
    format_history_lines,
    format_reason_help_lines,
    format_status_age,
    format_worker_label,
    handle_dashboard_key,
    move_selection,
    selected_gpu_payload,
)


def test_pause_and_resume_helpers_only_touch_pause_file(tmp_path) -> None:
    pause_file = pause_guard(tmp_path)

    assert pause_file == pause_file_for(tmp_path)
    assert pause_file.read_text(encoding="utf-8") == "paused\n"

    resumed = resume_guard(tmp_path)

    assert resumed == pause_file
    assert not pause_file.exists()


def test_temporary_pause_expires_and_removes_pause_file(tmp_path) -> None:
    pause_file = pause_guard(tmp_path, duration_seconds=60, now=100)

    state = read_pause_state(tmp_path, now=120)

    assert state.active is True
    assert state.path == pause_file
    assert state.pause_until == 160

    expired = read_pause_state(tmp_path, now=161)

    assert expired.active is False
    assert expired.pause_until is None
    assert not pause_file.exists()


def test_legacy_pause_file_remains_indefinite(tmp_path) -> None:
    pause_file = pause_file_for(tmp_path)
    pause_file.write_text("paused\n", encoding="utf-8")

    state = read_pause_state(tmp_path, now=999)

    assert state.active is True
    assert state.pause_until is None
    assert pause_file.exists()


def test_parse_duration_seconds_accepts_common_suffixes() -> None:
    assert parse_duration_seconds("30") == 30
    assert parse_duration_seconds("30s") == 30
    assert parse_duration_seconds("10m") == 600
    assert parse_duration_seconds("2h") == 7200
    assert parse_duration_seconds("1d") == 86400


def test_disable_and_enable_gpu_helpers_share_state_file(tmp_path) -> None:
    path = disabled_gpus_file_for(tmp_path)

    disable_gpu(tmp_path, 2)
    disable_gpu(tmp_path, 0)
    disable_gpu(tmp_path, 2)

    assert path.exists()
    assert read_disabled_gpus(tmp_path) == {0, 2}

    enable_gpu(tmp_path, 2)

    assert read_disabled_gpus(tmp_path) == {0}


def test_temporary_disable_gpu_expires_and_removes_entry(tmp_path) -> None:
    path = disable_gpu(tmp_path, 2, duration_seconds=60, now=100)

    state = read_disabled_gpu_state(tmp_path, now=120)

    assert state.path == path
    assert state.active == {2}
    assert state.disabled_until_by_gpu == {2: 160}

    expired = read_disabled_gpu_state(tmp_path, now=161)

    assert expired.active == set()
    assert expired.disabled_until_by_gpu == {}
    assert json.loads(path.read_text(encoding="utf-8")) == []


def test_temporary_disable_gpu_keeps_legacy_indefinite_entries(tmp_path) -> None:
    path = disabled_gpus_file_for(tmp_path)
    path.write_text("[1]\n", encoding="utf-8")

    disable_gpu(tmp_path, 2, duration_seconds=60, now=100)

    state = read_disabled_gpu_state(tmp_path, now=120)
    assert state.active == {1, 2}
    assert state.disabled_until_by_gpu == {2: 160}

    enable_gpu(tmp_path, 2)

    assert read_disabled_gpus(tmp_path) == {1}


def test_write_disabled_gpus_keeps_previous_state_when_replace_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = disabled_gpus_file_for(tmp_path)
    path.write_text("[1]\n", encoding="utf-8")
    original_replace = Path.replace
    replace_calls: list[Path] = []

    def fail_disabled_gpu_state_replace(self: Path, target: str | Path) -> Path:
        if Path(target) == path and self.name.startswith(f".{path.name}."):
            replace_calls.append(self)
            raise RuntimeError("replace failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_disabled_gpu_state_replace)

    with pytest.raises(RuntimeError, match="replace failed"):
        write_disabled_gpus(tmp_path, {2})

    assert replace_calls
    assert json.loads(path.read_text(encoding="utf-8")) == [1]
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_handle_dashboard_key_toggles_view_state(tmp_path) -> None:
    state = DashboardState()
    payload = {"gpus": [{"index": 0}, {"index": 2}]}

    assert handle_dashboard_key(ord("h"), payload, state, tmp_path)
    assert state.show_help is True
    assert state.message == "help shown"

    assert handle_dashboard_key(ord("c"), payload, state, tmp_path)
    assert state.show_config is True
    assert state.message == "config shown"

    assert handle_dashboard_key(ord("y"), payload, state, tmp_path)
    assert state.show_history is True
    assert state.message == "history shown"

    assert handle_dashboard_key(ord("x"), payload, state, tmp_path)
    assert state.show_explain is True
    assert state.message == "explain shown"

    assert handle_dashboard_key(ord("a"), payload, state, tmp_path)
    assert state.show_advice is True
    assert state.message == "advice shown"

    assert handle_dashboard_key(ord("i"), payload, state, tmp_path)
    assert state.selected_gpu_index == 0
    assert state.show_detail is True
    assert state.message == "details shown"


def test_handle_dashboard_key_filters_events_to_selected_gpu(tmp_path) -> None:
    state = DashboardState()
    payload = {"gpus": [{"index": 0}, {"index": 2}]}

    assert handle_dashboard_key(ord("f"), payload, state, tmp_path)

    assert state.selected_gpu_index == 0
    assert state.show_events is True
    assert state.filter_events_to_selected_gpu is True
    assert state.event_filter_gpu_index == 0
    assert state.message == "events/history filtered to selected GPU"

    assert handle_dashboard_key(ord("j"), payload, state, tmp_path)
    assert state.selected_gpu_index == 2
    assert state.event_filter_gpu_index == 2

    assert handle_dashboard_key(ord("f"), payload, state, tmp_path)
    assert state.filter_events_to_selected_gpu is False
    assert state.event_filter_gpu_index is None
    assert state.message == "event/history filter cleared"


def test_handle_dashboard_key_updates_pause_and_disable_state(tmp_path) -> None:
    state = DashboardState(selected_gpu_index=2)
    payload = {"gpus": [{"index": 2}]}

    assert handle_dashboard_key(ord("p"), payload, state, tmp_path)
    assert (tmp_path / "pause").exists()
    assert "paused via" in state.message

    assert handle_dashboard_key(ord("r"), payload, state, tmp_path)
    assert not (tmp_path / "pause").exists()
    assert "resumed via" in state.message

    assert handle_dashboard_key(ord("d"), payload, state, tmp_path)
    assert read_disabled_gpus(tmp_path) == {2}
    assert "disabled gpu=2" in state.message

    assert handle_dashboard_key(ord("d"), payload, state, tmp_path)
    assert read_disabled_gpus(tmp_path) == set()
    assert "enabled gpu=2" in state.message


def test_handle_dashboard_key_temporarily_pauses_all_workers(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = DashboardState(temporary_pause_seconds=120)
    payload = {"gpus": [{"index": 2}]}
    monkeypatch.setattr(tui.time, "time", lambda: 100.0)

    assert handle_dashboard_key(ord("P"), payload, state, tmp_path)

    pause_state = read_pause_state(tmp_path, now=110)
    assert pause_state.active is True
    assert pause_state.pause_until == 220
    assert "paused for 2m" in state.message
    assert "until=" in state.message


def test_handle_dashboard_key_temporary_disables_selected_gpu(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = DashboardState(selected_gpu_index=2, temporary_disable_seconds=600)
    payload = {"gpus": [{"index": 2}]}
    monkeypatch.setattr(tui.time, "time", lambda: 100.0)

    assert handle_dashboard_key(ord("D"), payload, state, tmp_path)

    disabled = read_disabled_gpu_state(tmp_path, now=120)
    assert disabled.active == {2}
    assert disabled.disabled_until_by_gpu == {2: 700.0}
    assert "disabled gpu=2 for 10m" in state.message
    assert "until=" in state.message


def test_handle_dashboard_key_quits_on_q(tmp_path) -> None:
    assert not handle_dashboard_key(ord("q"), None, DashboardState(), tmp_path)


def test_format_event_line_keeps_dashboard_compact() -> None:
    line = format_event_line(
        {
            "timestamp": 1,
            "type": "decision",
            "gpu_index": 2,
            "action": "assist",
            "reason": "low_util_emergency_with_busy_process",
            "pid": 1234,
            "program": "mixed",
        }
    )

    assert "type=decision" in line
    assert "gpu_index=2" in line
    assert "action=assist" in line
    assert "reason=low_util_emergency_with_busy_process" in line
    assert "timestamp" not in line


def test_format_dashboard_snapshot_renders_status_and_filtered_events() -> None:
    payload = {
        "timestamp": 100.0,
        "paused": False,
        "config": {"target_util": 75},
        "machine": {
            "average_utilization": 42.0,
            "policy_average_utilization": 50.0,
            "policy_window_seconds": 3600.0,
            "policy_window_sample_count": 120,
            "policy_window_coverage_seconds": 3500.0,
            "policy_window_coverage_ratio": 3500.0 / 3600.0,
            "action_counts": {"hold": 1},
        },
        "gpus": [
            {
                "index": 0,
                "name": "gpu-0",
                "utilization": 42,
                "temperature_celsius": 70,
                "memory_used_human": "2.00GiB",
                "memory_total_human": "80.00GiB",
                "decision": {
                    "action": "hold",
                    "reason": "below_target",
                    "history_average": 40.0,
                    "duty_cycle": 0.5,
                    "hold_mode": "balanced",
                },
                "worker": {
                    "pid": 1234,
                    "hold_mode": "balanced",
                    "duty_cycle": 0.5,
                },
                "processes": [],
            }
        ],
    }
    events = [
        {"type": "decision", "gpu_index": 0, "action": "hold", "reason": "below_target"},
        {"type": "decision", "gpu_index": 1, "action": "standby", "reason": "target_met"},
    ]

    lines = format_dashboard_snapshot(
        payload,
        events=events,
        show_events=True,
        show_detail=True,
        selected_gpu_index=0,
        event_filter_gpu_index=0,
        now=112.0,
    )
    rendered = "\n".join(lines)

    assert "gpu-holder dashboard" in rendered
    assert "age=12s" in rendered
    assert "window=ready coverage=97%" in rendered
    assert "quota=at_risk required=950.0%" in rendered
    assert "target=below_target gap=-25.0" in rendered
    assert "gpu=0 util=42%" in rendered
    assert "worker=pid=1234 balanced 50%" in rendered
    assert "Selected GPU 0: gpu-0" in rendered
    assert "Recent events for gpu=0" in rendered
    assert "reason=below_target" in rendered
    assert "target_met" not in rendered


def test_format_dashboard_snapshot_can_show_history() -> None:
    payload = {
        "timestamp": 100.0,
        "paused": False,
        "machine": {"average_utilization": 42.0, "policy_average_utilization": 50.0},
        "gpus": [
            {
                "index": 0,
                "name": "gpu-0",
                "utilization": 42,
                "temperature_celsius": 70,
                "memory_used_human": "2.00GiB",
                "memory_total_human": "80.00GiB",
                "decision": {"action": "hold", "reason": "below_target"},
                "processes": [],
            }
        ],
    }
    events = [
        {"timestamp": 1, "type": "decision", "gpu_index": 0, "action": "hold", "reason": "below_target"},
        {"timestamp": 2, "type": "worker_start", "gpu_index": 0, "pid": 123},
    ]

    lines = format_dashboard_snapshot(payload, events=events, show_history=True)
    rendered = "\n".join(lines)

    assert "Recent history" in rendered
    assert "events=2 decisions=1 actions=hold:1" in rendered
    assert "gpu=0 last=hold/below_target workers=start:1" in rendered


def test_format_dashboard_snapshot_can_show_explain() -> None:
    payload = {
        "timestamp": 100.0,
        "paused": False,
        "config": {"target_util": 75},
        "machine": {"average_utilization": 42.0, "policy_average_utilization": 50.0},
        "gpus": [
            {
                "index": 0,
                "name": "gpu-0",
                "utilization": 42,
                "temperature_celsius": 70,
                "memory_used_human": "2.00GiB",
                "memory_total_human": "80.00GiB",
                "decision": {"action": "release", "reason": "busy_process"},
                "processes": [{"pid": 123, "is_holder": False}],
            }
        ],
    }

    lines = format_dashboard_snapshot(payload, events=[], show_explain=True, now=112.0)
    rendered = "\n".join(lines)

    assert "Current explanation" in rendered
    assert "gpu-holder explain ok=True" in rendered
    assert "finding[warning]: below_target" in rendered
    assert "finding[info]: busy_process" in rendered


def test_format_dashboard_snapshot_can_show_advice() -> None:
    payload = {
        "timestamp": 100.0,
        "paused": False,
        "config": {
            "target_util": 60,
            "program": "matmul",
            "mem": "20%",
        },
        "machine": {"average_utilization": 42.0, "policy_average_utilization": 50.0},
        "gpus": [],
    }

    lines = format_dashboard_snapshot(payload, show_advice=True)
    rendered = "\n".join(lines)

    assert "Tuning advice" in rendered
    assert "grade=warn" in rendered
    assert "check=target_util_floor status=warn" in rendered
    assert "recommendation: Set --target-util" in rendered


def test_format_advice_lines_reports_missing_config() -> None:
    lines = format_advice_lines(None)

    assert "Tuning advice" in lines[0]
    assert "unavailable: status config is unavailable" in "\n".join(lines)


def test_format_explain_lines_renders_guidance() -> None:
    lines = format_explain_lines(
        {
            "timestamp": 100.0,
            "paused": True,
            "config": {"target_util": 75},
            "machine": {"policy_average_utilization": 80},
            "gpus": [],
        },
        [],
        now=100.0,
    )
    rendered = "\n".join(lines)

    assert "Current explanation" in rendered
    assert "finding[warning]: paused" in rendered
    assert "guidance:" in rendered


def test_format_history_lines_filters_to_gpu() -> None:
    lines = format_history_lines(
        [
            {"timestamp": 1, "type": "decision", "gpu_index": 0, "action": "hold", "reason": "below_target"},
            {"timestamp": 2, "type": "decision", "gpu_index": 1, "action": "release", "reason": "busy_process"},
        ],
        gpu_index=1,
    )
    rendered = "\n".join(lines)

    assert "Recent history for gpu=1" in rendered
    assert "actions=release:1" in rendered
    assert "gpu=1 last=release/busy_process" in rendered
    assert "below_target" not in rendered


def test_format_dashboard_snapshot_can_show_config() -> None:
    lines = format_dashboard_snapshot(
        {"config": {"target_util": 75, "mem": "20%"}},
        show_config=True,
    )

    rendered = "\n".join(lines)
    assert "Resolved config" in rendered
    assert "target_util=75" in rendered


def test_filter_dashboard_events_limits_to_selected_gpu() -> None:
    events = [
        {"type": "decision", "gpu_index": 0},
        {"type": "worker_stop", "gpu_index": 1},
        {"type": "controller_start"},
        {"type": "decision", "gpu_index": "1"},
    ]

    assert filter_dashboard_events(events, gpu_index=None) == events
    assert filter_dashboard_events(events, gpu_index=1) == [
        {"type": "worker_stop", "gpu_index": 1},
        {"type": "decision", "gpu_index": "1"},
    ]


def test_format_worker_label_prefers_owned_worker_over_backoff() -> None:
    label = format_worker_label(
        {
            "worker": {
                "pid": 4321,
                "hold_mode": "balanced",
                "duty_cycle": 0.5,
            },
            "worker_restart_backoff": {
                "remaining_seconds": 30.0,
            },
        }
    )

    assert label == "pid=4321 balanced 50%"


def test_format_worker_label_shows_backoff_when_no_worker_runs() -> None:
    label = format_worker_label(
        {
            "worker": None,
            "worker_restart_backoff": {
                "remaining_seconds": 29.7,
            },
        }
    )

    assert label == "backoff 30s"


def test_format_worker_label_shows_process_grace() -> None:
    label = format_worker_label(
        {
            "worker": None,
            "worker_restart_backoff": None,
            "process_grace": {
                "remaining_seconds": 14.7,
            },
        }
    )

    assert label == "grace 15s"


def test_format_worker_label_shows_disabled_gpu() -> None:
    label = format_worker_label(
        {
            "gpu_disabled": True,
            "worker": None,
        }
    )

    assert label == "disabled"


def test_format_worker_label_shows_temporary_disabled_gpu_expiration() -> None:
    label = format_worker_label(
        {
            "gpu_disabled": True,
            "gpu_disabled_until": 3600.0,
            "worker": None,
        }
    )

    assert label.startswith("disabled until ")
    assert "1970" in label


def test_format_worker_label_shows_thermal_block() -> None:
    label = format_worker_label(
        {
            "thermal_blocked": True,
            "worker": None,
        }
    )

    assert label == "thermal"


def test_dashboard_help_documents_runtime_controls() -> None:
    rendered = "\n".join(DASHBOARD_HELP_LINES)

    assert "p            pause" in rendered
    assert "P            temporarily pause" in rendered
    assert "c            show" in rendered
    assert "y            show" in rendered
    assert "x            show" in rendered
    assert "f            filter" in rendered
    assert "D            temporarily disable" in rendered
    assert "i / Enter" in rendered
    assert "d            disable" in rendered
    assert "h / ?" in rendered


def test_dashboard_help_lines_include_configured_temporary_durations() -> None:
    rendered = "\n".join(
        dashboard_help_lines(
            temporary_pause_seconds=120,
            temporary_disable_seconds=7200,
        )
    )

    assert "P            temporarily pause all holder workers for 2m" in rendered
    assert "D            temporarily disable selected GPU for 2.0h" in rendered


def test_selected_gpu_payload_finds_selected_row() -> None:
    gpus = [{"index": 0, "name": "gpu-0"}, {"index": 2, "name": "gpu-2"}]

    assert selected_gpu_payload(gpus, 2) == {"index": 2, "name": "gpu-2"}
    assert selected_gpu_payload(gpus, 1) is None
    assert selected_gpu_payload(gpus, None) is None


def test_format_config_lines_groups_known_config_fields() -> None:
    lines = format_config_lines(
        {
            "target_util": 75,
            "mem": "20%",
            "program": "mixed",
            "status_file": "/tmp/status.json",
        }
    )

    rendered = "\n".join(lines)
    assert "Resolved config" in rendered
    assert "[policy]" in rendered
    assert "  target_util=75" in rendered
    assert "[memory]" in rendered
    assert "  mem=20%" in rendered
    assert "[worker]" in rendered
    assert "  program=mixed" in rendered
    assert "[derived]" in rendered
    assert "  status_file=/tmp/status.json" in rendered


def test_format_config_lines_handles_missing_config_payload() -> None:
    assert "No config payload" in "\n".join(format_config_lines(None))


def test_format_gpu_detail_lines_includes_decision_worker_and_processes() -> None:
    lines = format_gpu_detail_lines(
        {
            "index": 0,
            "name": "test-gpu",
            "utilization": 42,
            "temperature_celsius": 70,
            "thermal_blocked": False,
            "memory_used_human": "2.00GiB",
            "memory_total_human": "80.00GiB",
            "decision": {
                "action": "hold",
                "reason": "below_target",
                "duty_cycle": 0.25,
                "hold_mode": "balanced",
            },
            "worker": {
                "pid": 4321,
                "hold_mode": "balanced",
                "duty_cycle": 0.25,
            },
            "processes": [
                {
                    "pid": 4321,
                    "name": "gpu-holder",
                    "used_memory_bytes": 1024**3,
                    "is_holder": True,
                    "protected_process": True,
                    "protected_process_pattern": "gpu-holder",
                }
            ],
        }
    )

    rendered = "\n".join(lines)
    assert "Selected GPU 0: test-gpu" in rendered
    assert "decision action=hold reason=below_target" in rendered
    assert "reason meaning=The machine policy average is below target_util" in rendered
    assert "reason guidance=This is normal holder behavior" in rendered
    assert "worker=pid=4321 balanced 25%" in rendered
    assert "pid=4321 name=gpu-holder mem=1.0GiB holder=True protected=True pattern=gpu-holder" in rendered


def test_format_gpu_detail_lines_includes_temporary_disable_expiration() -> None:
    lines = format_gpu_detail_lines(
        {
            "index": 0,
            "name": "test-gpu",
            "utilization": 0,
            "temperature_celsius": None,
            "thermal_blocked": False,
            "gpu_disabled": True,
            "gpu_disabled_until": 3600.0,
            "memory_used_human": "0.00B",
            "memory_total_human": "80.00GiB",
            "decision": {
                "action": "release",
                "reason": "gpu_disabled",
            },
            "worker": None,
            "processes": [],
        }
    )

    rendered = "\n".join(lines)
    assert "gpu_disabled=True until=" in rendered
    assert "1970" in rendered
    assert "worker=disabled until" in rendered


def test_format_reason_help_lines_handles_unknown_reason() -> None:
    assert format_reason_help_lines("unknown_reason") == [
        "  reason meaning=No reference entry for unknown_reason.",
        "  reason guidance=Run `gpu-holder reason-reference` to inspect known reasons.",
    ]


def test_format_gpu_detail_lines_handles_no_selection() -> None:
    assert "No GPU selected" in "\n".join(format_gpu_detail_lines(None))


def test_tui_reuses_status_age_formatting() -> None:
    assert format_status_age(61.0) == "1.0m"


def test_move_selection_uses_visible_gpu_indices() -> None:
    payload = {
        "gpus": [
            {"index": 0},
            {"index": 2},
            {"index": 4},
        ]
    }

    assert move_selection(payload, current_gpu_index=None, delta=1) == 0
    assert move_selection(payload, current_gpu_index=0, delta=1) == 2
    assert move_selection(payload, current_gpu_index=4, delta=1) == 4
    assert move_selection(payload, current_gpu_index=2, delta=-1) == 0
