from __future__ import annotations

import json
import os
from pathlib import Path

import gpu_holder.cli as cli
from gpu_holder.cli import main


def test_plan_fake_prints_decisions(capsys) -> None:
    result = main(["plan", "--fake"])

    captured = capsys.readouterr()
    assert result == 0
    assert "gpu=0 action=hold" in captured.out
    assert "gpu=1 action=assist" in captured.out
    assert "gpu=2 action=standby" in captured.out
    assert "mode=balanced" in captured.out


def test_plan_fake_accepts_compute_only_mode(capsys) -> None:
    result = main(["plan", "--fake", "--hold-mode", "compute-only"])

    captured = capsys.readouterr()
    assert result == 0
    assert "mode=compute-only" in captured.out


def test_plan_fake_accepts_random_program(capsys) -> None:
    result = main(["plan", "--fake", "--program", "random"])

    captured = capsys.readouterr()
    assert result == 0
    assert "program=random" in captured.out


def test_plan_fake_accepts_custom_program_rotation(capsys) -> None:
    result = main(["plan", "--fake", "--program", "matmul,conv"])

    captured = capsys.readouterr()
    assert result == 0
    assert "program=matmul,conv" in captured.out


def test_config_json_rejects_invalid_custom_program_rotation(capsys) -> None:
    result = main(["config", "--program", "matmul,unknown", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert "unsupported program: unknown worker program: 'unknown'" in payload["errors"]


def test_tune_json_scores_default_config(capsys) -> None:
    result = main(["tune", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["ok"] is True
    assert payload["grade"] == "pass"
    assert payload["summary"]["scenario_count"] == 5
    assert payload["simulation"]["summary"]["scenario_count"] == 5


def test_tune_exits_nonzero_for_failed_advice(capsys) -> None:
    result = main(["tune", "--mem", "90%", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is False
    assert payload["grade"] == "fail"


def test_tune_text_prints_recommendations(capsys) -> None:
    result = main(["tune", "--target-util", "60", "--program", "matmul"])

    captured = capsys.readouterr()
    assert result == 0
    assert "gpu-holder tune grade=warn" in captured.out
    assert "check=target_util_floor status=warn" in captured.out
    assert "recommendation:" in captured.out


def test_tune_strict_exits_nonzero_for_warnings(capsys) -> None:
    result = main(["tune", "--target-util", "60", "--program", "matmul", "--strict", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is True
    assert payload["strict"] is True
    assert payload["strict_ok"] is False
    assert payload["grade"] == "warn"


def test_tune_strict_passes_clean_config(capsys) -> None:
    result = main(["tune", "--strict", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["ok"] is True
    assert payload["strict"] is True
    assert payload["strict_ok"] is True
    assert payload["grade"] == "pass"


def test_tune_json_can_compare_builtin_profiles(capsys) -> None:
    result = main(["tune", "--compare-profiles", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    comparison = payload["profile_comparison"]
    assert comparison["recommended_profile"]
    assert {item["profile"] for item in comparison["profiles"]} == {
        "balanced",
        "conservative",
        "quota",
        "compute-only",
    }


def test_tune_text_prints_profile_comparison(capsys) -> None:
    result = main(["tune", "--compare-profiles"])

    captured = capsys.readouterr()
    assert result == 0
    assert "Profile comparison" in captured.out
    assert "profile=balanced" in captured.out
    assert "recommended_profile=" in captured.out


def test_recipes_lists_builtin_recipes(capsys) -> None:
    result = main(["recipes"])

    captured = capsys.readouterr()
    assert result == 0
    assert "gpu-holder recipes" in captured.out
    assert "first-run" in captured.out
    assert "strict-quota" in captured.out


def test_recipes_json_is_machine_readable(capsys) -> None:
    result = main(["recipes", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["schema_version"] == 1
    assert {recipe["name"] for recipe in payload["recipes"]} >= {"first-run", "strict-quota"}


def test_recipes_name_prints_toml(capsys) -> None:
    result = main(["recipes", "--name", "strict-quota"])

    captured = capsys.readouterr()
    assert result == 0
    assert '[guard]' in captured.out
    assert 'profile = "quota"' in captured.out
    assert 'program = "random"' in captured.out


def test_recipes_name_json_includes_toml(capsys) -> None:
    result = main(["recipes", "--name", "compute-only", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["recipe"]["name"] == "compute-only"
    assert 'hold_mode = "compute-only"' in payload["toml"]


def test_plan_fake_json_is_machine_readable(capsys) -> None:
    result = main(["plan", "--fake", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["decisions"][0]["action"] == "hold"
    assert payload["snapshots"][0]["index"] == 0


def test_preflight_fake_json_summarizes_startup_risk(tmp_path: Path, capsys) -> None:
    result = main(
        ["preflight", "--fake", "--no-diagnostics", "--state-dir", str(tmp_path), "--json"]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["ok"] is True
    assert payload["diagnostics"] is None
    assert payload["summary"]["visible_gpu_count"] == 3
    assert payload["summary"]["selected_gpu_count"] == 3
    assert payload["summary"]["action_counts"]["hold"] == 1
    assert payload["summary"]["action_counts"]["assist"] == 1
    assert payload["summary"]["reason_counts"]["low_util_emergency"] == 1
    assert payload["decisions"][0]["action"] == "hold"
    assert payload["risks"][0]["code"] == "holders_would_start"


def test_preflight_reports_missing_selected_gpu(tmp_path: Path, capsys) -> None:
    result = main(
        [
            "preflight",
            "--fake",
            "--gpus",
            "0,9",
            "--no-diagnostics",
            "--state-dir",
            str(tmp_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is False
    assert payload["summary"]["selected_gpu_count"] == 1
    assert payload["missing_selected_gpus"] == [9]
    assert "selected GPU indices are not visible: 9" in payload["errors"]


def test_preflight_reports_no_visible_gpus(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "_fake_snapshots", lambda: [])

    result = main(
        ["preflight", "--fake", "--no-diagnostics", "--state-dir", str(tmp_path), "--json"]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is False
    assert "no visible NVIDIA GPUs were found" in payload["errors"]
    assert payload["risks"][0]["code"] == "no_visible_gpus"


def test_preflight_reports_state_dir_that_is_not_a_directory(
    tmp_path: Path,
    capsys,
) -> None:
    state_path = tmp_path / "gpu-holder-state"
    state_path.write_text("not a directory\n", encoding="utf-8")

    result = main(
        [
            "preflight",
            "--fake",
            "--no-diagnostics",
            "--state-dir",
            str(state_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is False
    assert payload["state_dir"]["exists"] is True
    assert payload["state_dir"]["is_dir"] is False
    assert f"state_dir exists but is not a directory: {state_path}" in payload["errors"]
    assert payload["risks"][0]["code"] == "state_dir_not_directory"


def test_preflight_reports_uncreatable_state_dir_parent(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    state_path = tmp_path / "missing" / "gpu-holder-state"

    def fake_access(path: object, mode: int) -> bool:
        if Path(path) == tmp_path and mode == os.W_OK | os.X_OK:
            return False
        return True

    monkeypatch.setattr(cli.os, "access", fake_access)

    result = main(
        [
            "preflight",
            "--fake",
            "--no-diagnostics",
            "--state-dir",
            str(state_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is False
    assert payload["state_dir"]["exists"] is False
    assert payload["state_dir"]["nearest_existing_parent"] == str(tmp_path)
    assert f"state_dir parent is not writable: {tmp_path}" in payload["errors"]
    assert payload["risks"][0]["code"] == "state_dir_parent_not_writable"


def test_preflight_reports_state_dir_blocked_by_file_parent(
    tmp_path: Path,
    capsys,
) -> None:
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory\n", encoding="utf-8")
    state_path = blocked_parent / "gpu-holder-state"

    result = main(
        [
            "preflight",
            "--fake",
            "--no-diagnostics",
            "--state-dir",
            str(state_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is False
    assert payload["state_dir"]["blocked_by_file"] == str(blocked_parent)
    assert (
        f"state_dir path is blocked by a non-directory parent: {blocked_parent}"
        in payload["errors"]
    )
    assert payload["risks"][0]["code"] == "state_dir_parent_not_directory"


def test_preflight_text_is_human_readable(tmp_path: Path, capsys) -> None:
    result = main(["preflight", "--fake", "--no-diagnostics", "--state-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert "gpu-holder preflight ok=True" in captured.out
    assert "selected_gpus=3 visible_gpus=3" in captured.out
    assert "reasons={" in captured.out
    assert "gpu=0 action=hold reason=low_util_emergency" in captured.out


def test_preflight_reports_protected_process_matches(tmp_path: Path, capsys) -> None:
    result = main(
        [
            "preflight",
            "--fake",
            "--no-diagnostics",
            "--protected-process",
            "train.py",
            "--state-dir",
            str(tmp_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["summary"]["reason_counts"]["protected_process"] == 2
    assert any(risk["code"] == "protected_process_matches" for risk in payload["risks"])
    assert [decision["reason"] for decision in payload["decisions"]].count("protected_process") == 2


def test_preflight_reports_live_non_holder_pid_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    pid_file = tmp_path / "gpu-holder.pid"
    pid_file.write_text("12345\n", encoding="utf-8")

    monkeypatch.setattr(cli, "_is_gpu_holder_guard_process", lambda pid: False)
    monkeypatch.setattr(cli, "_process_exists", lambda pid: True)

    result = main(
        [
            "preflight",
            "--fake",
            "--no-diagnostics",
            "--state-dir",
            str(tmp_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is False
    assert payload["pid_file"]["pid"] == 12345
    assert "pid file points to a live non-holder process: 12345" in payload["errors"]
    assert payload["risks"][0]["code"] == "pid_file_live_non_holder"
    assert pid_file.read_text(encoding="utf-8") == "12345\n"


def test_preflight_reports_existing_gpu_holder_pid_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    pid_file = tmp_path / "gpu-holder.pid"
    pid_file.write_text("12345\n", encoding="utf-8")

    monkeypatch.setattr(cli, "_is_gpu_holder_guard_process", lambda pid: True)
    monkeypatch.setattr(cli, "_process_exists", lambda pid: True)

    result = main(
        [
            "preflight",
            "--fake",
            "--no-diagnostics",
            "--state-dir",
            str(tmp_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is False
    assert "gpu-holder already appears to be running: 12345" in payload["errors"]
    assert payload["risks"][0]["code"] == "pid_file_existing_controller"


def test_preflight_warns_about_stale_pid_file_without_removing_it(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    pid_file = tmp_path / "gpu-holder.pid"
    pid_file.write_text("12345\n", encoding="utf-8")

    monkeypatch.setattr(cli, "_is_gpu_holder_guard_process", lambda pid: False)
    monkeypatch.setattr(cli, "_process_exists", lambda pid: False)

    result = main(
        [
            "preflight",
            "--fake",
            "--no-diagnostics",
            "--state-dir",
            str(tmp_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["ok"] is True
    assert "stale pid file will be removed by guard/start: 12345" in payload["warnings"]
    assert payload["risks"][0]["code"] == "pid_file_stale"
    assert pid_file.read_text(encoding="utf-8") == "12345\n"


def test_plan_uses_config_file_and_cli_overrides(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "gpu-holder.toml"
    config_path.write_text(
        """
[guard]
gpus = [0, 1]
target_util = 60
mem = "10%"
hold_mode = "memory-only"
""".strip(),
        encoding="utf-8",
    )

    result = main(["plan", "--fake", "--json", "--config", str(config_path), "--hold-mode", "compute-only"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["decisions"][0]["hold_mode"] == "compute-only"
    assert payload["decisions"][0]["memory_bytes"] == 0
    assert [item["index"] for item in payload["snapshots"]] == [0, 1]


def test_config_json_prints_resolved_config(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "gpu-holder.toml"
    config_path.write_text(
        """
[guard]
gpus = [0, 2]
target_util = 80
machine_window = 1800
mem = "10%"
state_dir = "~/.gpu-holder-test"
event_log_max_size = "1MiB"
event_log_backup_count = 5
worker_restart_backoff = 45
worker_start_timeout = 6
process_grace_window = 20
protected_process_patterns = ["python*train.py", "vllm*"]
max_gpu_temp = 82
thermal_resume_temp = 77
max_held_gpus = 2
min_duty_cycle = 0.2
max_duty_cycle = 0.8
current_gap_weight = 0.5
history_gap_weight = 0.2
machine_gap_weight = 0.3
""".strip(),
        encoding="utf-8",
    )

    result = main(["config", "--config", str(config_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["ok"] is True
    assert payload["config"]["gpus"] == [0, 2]
    assert payload["config"]["target_util"] == 80
    assert payload["config"]["machine_window"] == 1800.0
    assert payload["config"]["mem"] == "10%"
    assert payload["config"]["state_dir"].endswith(".gpu-holder-test")
    assert payload["config"]["event_log_max_bytes"] == 1024**2
    assert payload["config"]["event_log_backup_count"] == 5
    assert payload["config"]["worker_restart_backoff"] == 45.0
    assert payload["config"]["worker_start_timeout"] == 6.0
    assert payload["config"]["process_grace_window"] == 20.0
    assert payload["config"]["protected_process_patterns"] == ["python*train.py", "vllm*"]
    assert payload["config"]["max_gpu_temp"] == 82
    assert payload["config"]["thermal_resume_temp"] == 77
    assert payload["config"]["max_held_gpus"] == 2
    assert payload["config"]["min_duty_cycle"] == 0.2
    assert payload["config"]["max_duty_cycle"] == 0.8
    assert payload["config"]["current_gap_weight"] == 0.5
    assert payload["config"]["history_gap_weight"] == 0.2
    assert payload["config"]["machine_gap_weight"] == 0.3


def test_config_accepts_repeated_protected_process_cli_flags(capsys) -> None:
    result = main(
        [
            "config",
            "--protected-process",
            "python*train.py,vllm*",
            "--protected-process",
            "*serve*",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["config"]["protected_process_patterns"] == [
        "python*train.py",
        "vllm*",
        "*serve*",
    ]


def test_config_profile_applies_defaults_and_allows_cli_overrides(capsys) -> None:
    result = main(
        [
            "config",
            "--profile",
            "conservative",
            "--target-util",
            "76",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["ok"] is True
    assert payload["config"]["profile"] == "conservative"
    assert payload["config"]["target_util"] == 76
    assert payload["config"]["mem"] == "10%"
    assert payload["config"]["max_duty_cycle"] == 0.6
    assert payload["config"]["max_held_gpus"] == 1


def test_config_file_profile_is_default_layer_not_forced_override(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "gpu-holder.toml"
    config_path.write_text(
        """
[guard]
profile = "quota"
target_util = 72
""".strip(),
        encoding="utf-8",
    )

    result = main(["config", "--config", str(config_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["config"]["profile"] == "quota"
    assert payload["config"]["target_util"] == 72
    assert payload["config"]["idle_util"] == 55
    assert payload["config"]["program"] == "random"


def test_config_explain_reports_profile_overrides(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "gpu-holder.toml"
    config_path.write_text(
        """
[guard]
profile = "quota"
target_util = 72
""".strip(),
        encoding="utf-8",
    )

    result = main(
        [
            "config",
            "--config",
            str(config_path),
            "--max-duty-cycle",
            "0.7",
            "--explain",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    explain = payload["explain"]
    assert result == 0
    assert explain["profile"] == "quota"
    assert explain["file_overrides"] == ["profile", "target_util"]
    assert "max_duty_cycle" in explain["cli_overrides"]
    assert explain["profile_defaults_applied"]["idle_util"] == 55
    assert explain["profile_defaults_overridden"]["target_util"] == {
        "profile_value": 80,
        "source": "config",
        "effective_value": 72,
    }
    assert explain["profile_defaults_overridden"]["max_duty_cycle"] == {
        "profile_value": 1.0,
        "source": "cli",
        "effective_value": 0.7,
    }


def test_config_explain_text_is_human_readable(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "gpu-holder.toml"
    config_path.write_text(
        """
[guard]
profile = "conservative"
mem = "15%"
""".strip(),
        encoding="utf-8",
    )

    result = main(["config", "--config", str(config_path), "--explain"])

    captured = capsys.readouterr()
    assert result == 0
    assert "config explain" in captured.out
    assert "profile=conservative" in captured.out
    assert "file_overrides=mem,profile" in captured.out
    assert "mem: profile=10% source=config effective=15%" in captured.out


def test_config_rejects_unknown_profile_from_config_file(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "gpu-holder.toml"
    config_path.write_text(
        """
[guard]
profile = "unknown"
""".strip(),
        encoding="utf-8",
    )

    result = main(["config", "--config", str(config_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert any("unsupported profile: unknown" in error for error in payload["errors"])


def test_config_json_accepts_duty_cli_overrides(capsys) -> None:
    result = main(
        [
            "config",
            "--min-duty-cycle",
            "0.25",
            "--max-duty-cycle",
            "0.75",
            "--current-gap-weight",
            "0.6",
            "--history-gap-weight",
            "0.1",
            "--machine-gap-weight",
            "0.3",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["ok"] is True
    assert payload["config"]["min_duty_cycle"] == 0.25
    assert payload["config"]["max_duty_cycle"] == 0.75
    assert payload["config"]["current_gap_weight"] == 0.6
    assert payload["config"]["history_gap_weight"] == 0.1
    assert payload["config"]["machine_gap_weight"] == 0.3


def test_config_json_accepts_compute_burst_overrides(capsys) -> None:
    result = main(
        [
            "config",
            "--compute-burst-seconds",
            "0.35",
            "--compute-burst-jitter",
            "0.4",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["config"]["compute_burst_seconds"] == 0.35
    assert payload["config"]["compute_burst_jitter"] == 0.4


def test_config_json_accepts_worker_update_duty_threshold(capsys) -> None:
    result = main(["config", "--worker-update-duty-threshold", "0.08", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["config"]["worker_update_duty_threshold"] == 0.08


def test_config_json_rejects_invalid_compute_burst(capsys) -> None:
    result = main(["config", "--compute-burst-seconds", "0", "--compute-burst-jitter", "1.5", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert "compute_burst_seconds must be positive" in payload["errors"]
    assert "compute_burst_jitter must be between 0 and 1" in payload["errors"]


def test_config_json_rejects_invalid_worker_update_duty_threshold(capsys) -> None:
    result = main(["config", "--worker-update-duty-threshold", "1.2", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert "worker_update_duty_threshold must be between 0 and 1" in payload["errors"]


def test_config_json_allows_disabling_thermal_limit(capsys) -> None:
    result = main(["config", "--max-gpu-temp", "0", "--thermal-resume-temp", "0", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["ok"] is True
    assert payload["config"]["max_gpu_temp"] is None
    assert payload["config"]["thermal_resume_temp"] is None


def test_config_json_reports_validation_errors(capsys) -> None:
    result = main(["config", "--target-util", "0", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is False
    assert "target_util must be between 1 and 100" in payload["errors"]


def test_config_json_rejects_unknown_config_file_key(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "gpu-holder.toml"
    config_path.write_text(
        """
[guard]
target_util = 80
taget_util = 70
""".strip(),
        encoding="utf-8",
    )

    result = main(["config", "--config", str(config_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is False
    assert "unknown config key: taget_util" in payload["errors"]


def test_config_json_reports_missing_config_file(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "missing.toml"

    result = main(["config", "--config", str(config_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is False
    assert payload["errors"]
    assert payload["errors"][0].startswith("failed to load config file")


def test_config_json_reports_invalid_config_value(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "gpu-holder.toml"
    config_path.write_text(
        """
[guard]
mem = "not-a-memory-size"
""".strip(),
        encoding="utf-8",
    )

    result = main(["config", "--config", str(config_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is False
    assert payload["errors"][0].startswith("invalid config value:")


def test_plan_rejects_unknown_config_file_key(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "gpu-holder.toml"
    config_path.write_text(
        """
[guard]
target_util = 80
unknown_setting = true
""".strip(),
        encoding="utf-8",
    )

    result = main(["plan", "--fake", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert result == 2
    assert "config error: unknown config key: unknown_setting" in captured.err


def test_config_json_rejects_negative_thermal_limit(capsys) -> None:
    result = main(["config", "--max-gpu-temp", "-1", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert "max_gpu_temp must be positive when enabled" in payload["errors"]


def test_config_json_rejects_invalid_thermal_hysteresis(capsys) -> None:
    result = main(["config", "--max-gpu-temp", "80", "--thermal-resume-temp", "80", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert "thermal_resume_temp must be lower than max_gpu_temp" in payload["errors"]


def test_config_reference_json_lists_documented_fields(capsys) -> None:
    result = main(["config-reference", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    fields = {field["key"]: field for field in payload["fields"]}
    assert result == 0
    assert payload["schema_version"] == 1
    assert fields["profile"]["cli_flag"] == "--profile"
    assert fields["profile"]["default"] == "balanced"
    assert fields["target_util"]["cli_flag"] == "--target-util"
    assert fields["target_util"]["default"] == 75
    assert fields["mem"]["example"] == "20%, 10GiB, 12000MiB"
    assert fields["program"]["example"] == "mixed or matmul,conv,fft"
    assert fields["compute_burst_seconds"]["cli_flag"] == "--compute-burst-seconds"
    assert fields["worker_update_duty_threshold"]["default"] == 0.05
    assert fields["thermal_resume_temp"]["default"] == 80
    assert fields["event_log_max_size"]["cli_flag"] == "--event-log-max-size"
    assert fields["protected_process_patterns"]["cli_flag"] == "--protected-process"


def test_config_reference_text_is_human_readable(capsys) -> None:
    result = main(["config-reference"])

    captured = capsys.readouterr()
    assert result == 0
    assert "gpu-holder config reference" in captured.out
    assert "[policy]" in captured.out
    assert "target_util (int percent) default=75 cli=--target-util" in captured.out
    assert "Memory allocation target" in captured.out


def test_profile_reference_lists_runtime_profiles(capsys) -> None:
    result = main(["profile-reference", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    profiles = {profile["name"]: profile for profile in payload["profiles"]}
    assert result == 0
    assert payload["schema_version"] == 1
    assert "conservative" in profiles
    assert profiles["quota"]["values"]["program"] == "random"
    assert profiles["compute-only"]["values"]["hold_mode"] == "compute-only"


def test_profile_reference_text_is_human_readable(capsys) -> None:
    result = main(["profile-reference"])

    captured = capsys.readouterr()
    assert result == 0
    assert "gpu-holder profile reference" in captured.out
    assert "conservative" in captured.out
    assert "values:" in captured.out


def test_reason_reference_json_lists_policy_reasons(capsys) -> None:
    result = main(["reason-reference", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    reasons = {item["reason"]: item for item in payload["reasons"]}
    assert result == 0
    assert payload["schema_version"] == 1
    assert reasons["busy_process"]["actions"] == ["release", "standby"]
    assert "threshold" in reasons["busy_process"]["meaning"]


def test_reason_reference_text_is_human_readable(capsys) -> None:
    result = main(["reason-reference"])

    captured = capsys.readouterr()
    assert result == 0
    assert "gpu-holder reason reference" in captured.out
    assert "thermal_limit actions=release,standby" in captured.out
    assert "guidance:" in captured.out


def test_service_command_prints_systemd_user_unit(capsys) -> None:
    result = main(["service", "--config", "/etc/gpu-holder.toml", "--state-dir", "/tmp/gpu-holder"])

    captured = capsys.readouterr()
    assert result == 0
    assert "[Unit]" in captured.out
    assert "ExecStart=gpu-holder guard --config /etc/gpu-holder.toml --state-dir /tmp/gpu-holder" in captured.out
    assert "Restart=always" in captured.out


def test_service_command_accepts_extra_guard_args(capsys) -> None:
    result = main(["service", "--", "--gpus", "0,1", "--dry-run"])

    captured = capsys.readouterr()
    assert result == 0
    assert "ExecStart=gpu-holder guard --gpus 0,1 --dry-run" in captured.out


def test_completion_command_prints_shell_script(capsys) -> None:
    result = main(["completion", "bash"])

    captured = capsys.readouterr()
    assert result == 0
    assert "_gpu_holder_completion" in captured.out


def test_alerts_command_prints_prometheus_rules(capsys) -> None:
    result = main(["alerts", "--group-name", "custom"])

    captured = capsys.readouterr()
    assert result == 0
    assert '- name: "custom"' in captured.out
    assert "GPUHolderQuotaForecastAtRisk" in captured.out
    assert "gpu_holder_target_forecast_at_risk == 1" in captured.out


def test_alerts_command_can_print_json(capsys) -> None:
    result = main(["alerts", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    alerts = [rule["alert"] for rule in payload["groups"][0]["rules"]]
    assert result == 0
    assert "GPUHolderStatusMissing" in alerts
    assert "GPUHolderQuotaForecastAtRisk" in alerts


def test_grafana_dashboard_command_prints_dashboard_json(capsys) -> None:
    result = main(
        [
            "grafana-dashboard",
            "--title",
            "Shared GPUs",
            "--uid",
            "shared-gpus",
            "--datasource",
            "Prometheus",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    panel_titles = [panel["title"] for panel in payload["panels"]]
    assert result == 0
    assert payload["title"] == "Shared GPUs"
    assert payload["uid"] == "shared-gpus"
    assert "Quota forecast" in panel_titles
    assert "Per-GPU utilization" in panel_titles
    assert "gpu_holder_target_forecast_at_risk" in captured.out


def test_monitoring_bundle_command_writes_files(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "bundle"

    result = main(
        [
            "monitoring-bundle",
            "--output-dir",
            str(output_dir),
            "--group-name",
            "shared-gpus",
            "--dashboard-title",
            "Shared GPUs",
            "--dashboard-uid",
            "shared-gpus",
            "--datasource",
            "Prometheus",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert Path(payload["files"]["alerts"]).exists()
    assert Path(payload["files"]["dashboard"]).exists()
    assert Path(payload["files"]["readme"]).exists()
    assert "GPUHolderQuotaForecastAtRisk" in Path(payload["files"]["alerts"]).read_text(
        encoding="utf-8"
    )
    dashboard = json.loads(Path(payload["files"]["dashboard"]).read_text(encoding="utf-8"))
    assert dashboard["title"] == "Shared GPUs"


def test_manual_command_prints_markdown(capsys) -> None:
    result = main(["manual"])

    captured = capsys.readouterr()
    assert result == 0
    assert "# gpu-holder manual" in captured.out
    assert "### `guard`" in captured.out


def test_manual_command_prints_roff(capsys) -> None:
    result = main(["manual", "--format", "man"])

    captured = capsys.readouterr()
    assert result == 0
    assert '.TH "GPU-HOLDER" "1"' in captured.out


def test_init_config_writes_default_config(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "gpu-holder.toml"

    result = main(["init-config", "--path", str(config_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert "wrote config" in captured.out
    content = config_path.read_text(encoding="utf-8")
    assert "[guard]" in content
    assert 'mem = "20%"' in content

    result = main(["config", "--config", str(config_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["ok"] is True
    assert payload["config"]["program"] == "mixed"


def test_init_config_stdout_prints_default_config_without_writing(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "gpu-holder.toml"

    result = main(["init-config", "--path", str(config_path), "--stdout"])

    captured = capsys.readouterr()
    assert result == 0
    assert "[guard]" in captured.out
    assert 'mem = "20%"' in captured.out
    assert not config_path.exists()


def test_init_config_profile_writes_minimal_profile_config(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "gpu-holder.toml"

    result = main(["init-config", "--path", str(config_path), "--profile", "quota"])

    captured = capsys.readouterr()
    assert result == 0
    assert "wrote config" in captured.out
    content = config_path.read_text(encoding="utf-8")
    assert 'profile = "quota"' in content
    assert "\ntarget_util = 75" not in content
    assert '# target_util = 75' in content

    capsys.readouterr()
    result = main(["config", "--config", str(config_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["config"]["profile"] == "quota"
    assert payload["config"]["target_util"] == 80
    assert payload["config"]["idle_util"] == 55


def test_init_config_minimal_keeps_balanced_profile_defaults(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "gpu-holder.toml"

    result = main(["init-config", "--path", str(config_path), "--minimal"])

    assert result == 0
    content = config_path.read_text(encoding="utf-8")
    assert 'profile = "balanced"' in content
    assert "\ntarget_util = 75" not in content
    capsys.readouterr()
    payload_result = main(["config", "--config", str(config_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload_result == 0
    assert payload["config"]["target_util"] == 75


def test_init_config_refuses_to_overwrite_existing_file(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "gpu-holder.toml"
    config_path.write_text("keep me\n", encoding="utf-8")

    result = main(["init-config", "--path", str(config_path)])

    captured = capsys.readouterr()
    assert result == 2
    assert "already exists" in captured.err
    assert config_path.read_text(encoding="utf-8") == "keep me\n"


def test_init_config_force_overwrites_existing_file(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "gpu-holder.toml"
    config_path.write_text("old\n", encoding="utf-8")

    result = main(["init-config", "--path", str(config_path), "--force"])

    captured = capsys.readouterr()
    assert result == 0
    assert "wrote config" in captured.out
    assert config_path.read_text(encoding="utf-8").startswith("[guard]\n")


def test_simulate_json_uses_built_in_trace(capsys) -> None:
    result = main(["simulate", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["scenario"] == "mixed"
    assert payload["trace"] is None
    assert payload["summary"]["steps"] == 4
    assert payload["summary"]["action_counts"]["assist"] >= 1


def test_simulate_json_accepts_built_in_scenario(capsys) -> None:
    result = main(["simulate", "--scenario", "thermal", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["scenario"] == "thermal"
    assert payload["timeline"][0]["decisions"][0]["reason"] == "thermal_limit"


def test_simulate_json_accepts_all_built_in_scenarios(capsys) -> None:
    result = main(["simulate", "--scenario", "all", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["scenario"] == "all"
    assert payload["trace"] is None
    assert payload["summary"]["scenario_count"] == 5
    assert set(payload["summary"]["scenarios"]) == {"mixed", "idle", "busy", "thermal", "quota-risk"}
    assert len(payload["scenarios"]) == 5
    assert payload["summary"]["action_counts"]["hold"] >= 1


def test_simulate_all_text_prints_suite_summary(capsys) -> None:
    result = main(["simulate", "--scenario", "all"])

    captured = capsys.readouterr()
    assert result == 0
    assert "scenarios=5" in captured.out
    assert "scenario=thermal" in captured.out
    assert "scenario=quota-risk" in captured.out


def test_simulate_json_reads_trace_file(tmp_path: Path, capsys) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            [
                {
                    "timestamp": 0,
                "gpus": [
                    {
                        "index": 0,
                        "utilization": 10,
                        "temperature": 79,
                        "memory_total": "80GiB",
                        "memory_used": "2GiB",
                    }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = main(["simulate", "--trace", str(trace_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["scenario"] is None
    assert payload["trace"] == str(trace_path)
    assert payload["summary"]["steps"] == 1
    assert payload["timeline"][0]["decisions"][0]["gpu_index"] == 0


def test_simulate_reports_missing_trace_file_without_traceback(tmp_path: Path, capsys) -> None:
    result = main(["simulate", "--trace", str(tmp_path / "missing.json")])

    captured = capsys.readouterr()
    assert result == 2
    assert "trace error:" in captured.err
    assert "Traceback" not in captured.err


def test_simulate_json_reports_invalid_trace_file_as_json(tmp_path: Path, capsys) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text("{not-json\n", encoding="utf-8")

    result = main(["simulate", "--trace", str(trace_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is False
    assert "trace error:" in payload["error"]


def test_simulate_json_reports_trace_schema_error_with_path(tmp_path: Path, capsys) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps([{"gpus": [{"index": 0}]}]),
        encoding="utf-8",
    )

    result = main(["simulate", "--trace", str(trace_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["ok"] is False
    assert "frames[0].gpus[0].utilization is required" in payload["error"]


def test_plan_rejects_invalid_resolved_config(capsys) -> None:
    result = main(["plan", "--fake", "--sample-interval", "0"])

    captured = capsys.readouterr()
    assert result == 2
    assert "config error: sample_interval must be positive" in captured.err


def test_start_removes_stale_pid_file_and_starts(tmp_path: Path, monkeypatch, capsys) -> None:
    pid_file = tmp_path / "gpu-holder.pid"
    pid_file.write_text("999999", encoding="utf-8")
    started_commands = []

    class FakePopen:
        pid = 4242

        def __init__(self, command, **kwargs) -> None:
            started_commands.append((command, kwargs))

    monkeypatch.setattr(cli, "_is_gpu_holder_guard_process", lambda pid: False)
    monkeypatch.setattr(cli, "_process_exists", lambda pid: False)
    monkeypatch.setattr(cli.subprocess, "Popen", FakePopen)

    result = main(["start", "--state-dir", str(tmp_path), "--dry-run"])

    captured = capsys.readouterr()
    assert result == 0
    assert "removed stale pid file" in captured.err
    assert pid_file.read_text(encoding="utf-8") == "4242"
    assert started_commands
    command, kwargs = started_commands[0]
    assert command[:3] == [cli.sys.executable, "-m", "gpu_holder.cli"]
    assert kwargs["env"].get("PYTHONPATH") == os.environ.get("PYTHONPATH")


def test_start_refuses_pid_file_for_live_non_holder_process(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    pid_file = tmp_path / "gpu-holder.pid"
    pid_file.write_text("12345", encoding="utf-8")

    monkeypatch.setattr(cli, "_is_gpu_holder_guard_process", lambda pid: False)
    monkeypatch.setattr(cli, "_process_exists", lambda pid: True)
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not start")),
    )

    result = main(["start", "--state-dir", str(tmp_path), "--dry-run"])

    captured = capsys.readouterr()
    assert result == 2
    assert "belongs to another process" in captured.err
    assert pid_file.read_text(encoding="utf-8") == "12345"


def test_start_refuses_existing_holder_process(tmp_path: Path, monkeypatch, capsys) -> None:
    pid_file = tmp_path / "gpu-holder.pid"
    pid_file.write_text("12345", encoding="utf-8")

    monkeypatch.setattr(cli, "_is_gpu_holder_guard_process", lambda pid: True)
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not start")),
    )

    result = main(["start", "--state-dir", str(tmp_path), "--dry-run"])

    captured = capsys.readouterr()
    assert result == 1
    assert "already running" in captured.err


def test_start_does_not_overwrite_pidfile_created_after_spawn(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    pid_file = tmp_path / "gpu-holder.pid"
    terminated = []

    class FakePopen:
        pid = 4242

        def __init__(self, command, **kwargs) -> None:
            pid_file.write_text("12345\n", encoding="utf-8")

        def terminate(self) -> None:
            terminated.append(self.pid)

        def wait(self, timeout=None) -> int:
            return 0

    def fake_process_exists(pid: int) -> bool:
        return pid == 12345

    monkeypatch.setattr(cli, "_is_gpu_holder_guard_process", lambda pid: False)
    monkeypatch.setattr(cli, "_process_exists", fake_process_exists)
    monkeypatch.setattr(cli.subprocess, "Popen", FakePopen)

    result = main(["start", "--state-dir", str(tmp_path), "--dry-run"])

    captured = capsys.readouterr()
    assert result == 2
    assert "belongs to another process" in captured.err
    assert pid_file.read_text(encoding="utf-8") == "12345\n"
    assert terminated == [4242]


def test_guard_refuses_pid_file_for_live_non_holder_process(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    pid_file = tmp_path / "gpu-holder.pid"
    pid_file.write_text("12345", encoding="utf-8")

    monkeypatch.setattr(cli, "_is_gpu_holder_guard_process", lambda pid: False)
    monkeypatch.setattr(cli, "_process_exists", lambda pid: True)
    monkeypatch.setattr(
        cli.GuardController,
        "run",
        lambda self: (_ for _ in ()).throw(AssertionError("must not run controller")),
    )

    result = main(["guard", "--state-dir", str(tmp_path), "--dry-run"])

    captured = capsys.readouterr()
    assert result == 2
    assert "belongs to another process" in captured.err
    assert pid_file.read_text(encoding="utf-8") == "12345"


def test_guard_allows_pid_file_for_current_process(tmp_path: Path, monkeypatch) -> None:
    pid_file = tmp_path / "gpu-holder.pid"
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    ran = []

    monkeypatch.setattr(cli, "_is_gpu_holder_guard_process", lambda pid: True)
    monkeypatch.setattr(cli.GuardController, "run", lambda self: ran.append(self.config.pid_file))

    result = main(["guard", "--state-dir", str(tmp_path), "--dry-run"])

    assert result == 0
    assert ran == [pid_file]


def test_guard_reserves_pid_file_before_controller_runs(tmp_path: Path, monkeypatch) -> None:
    seen_pidfile = []

    def fake_run(self) -> None:
        seen_pidfile.append(self.config.pid_file.read_text(encoding="utf-8").strip())

    monkeypatch.setattr(cli.GuardController, "run", fake_run)

    result = main(["guard", "--state-dir", str(tmp_path), "--dry-run"])

    assert result == 0
    assert seen_pidfile == [str(os.getpid())]


def test_guard_replaces_invalid_stale_pid_file_before_controller_runs(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    pid_file = tmp_path / "gpu-holder.pid"
    pid_file.write_text("not-a-pid", encoding="utf-8")
    seen_pidfile = []

    def fake_run(self) -> None:
        seen_pidfile.append(self.config.pid_file.read_text(encoding="utf-8").strip())

    monkeypatch.setattr(cli.GuardController, "run", fake_run)

    result = main(["guard", "--state-dir", str(tmp_path), "--dry-run"])

    captured = capsys.readouterr()
    assert result == 0
    assert "removed invalid stale pid file" in captured.err
    assert seen_pidfile == [str(os.getpid())]


def test_stop_refuses_pid_that_is_not_gpu_holder_guard(tmp_path: Path, monkeypatch, capsys) -> None:
    pid_file = tmp_path / "gpu-holder.pid"
    pid_file.write_text("12345", encoding="utf-8")

    monkeypatch.setattr(cli, "_is_gpu_holder_guard_process", lambda pid: False)

    result = main(["stop", "--state-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 2
    assert "refusing to stop" in captured.err


def test_guard_process_identity_check(monkeypatch, tmp_path: Path) -> None:
    proc = tmp_path / "proc" / "42"
    proc.mkdir(parents=True)
    (proc / "cmdline").write_bytes(b"python\0-m\0gpu_holder.cli\0guard\0")

    original_path = cli.Path

    def fake_path(value: str):
        if value == "/proc/42/cmdline":
            return proc / "cmdline"
        return original_path(value)

    monkeypatch.setattr(cli, "Path", fake_path)

    assert cli._is_gpu_holder_guard_process(42)


def test_guard_process_identity_rejects_non_holder(monkeypatch, tmp_path: Path) -> None:
    proc = tmp_path / "proc" / "42"
    proc.mkdir(parents=True)
    (proc / "cmdline").write_bytes(b"python\0train.py\0")

    original_path = cli.Path

    def fake_path(value: str):
        if value == "/proc/42/cmdline":
            return proc / "cmdline"
        return original_path(value)

    monkeypatch.setattr(cli, "Path", fake_path)

    assert not cli._is_gpu_holder_guard_process(42)


def test_pause_and_resume_commands_use_pause_file(tmp_path: Path, capsys) -> None:
    assert main(["pause", "--state-dir", str(tmp_path)]) == 0
    assert (tmp_path / "pause").exists()

    assert main(["resume", "--state-dir", str(tmp_path)]) == 0
    assert not (tmp_path / "pause").exists()

    captured = capsys.readouterr()
    assert "paused:" in captured.out
    assert "resumed:" in captured.out


def test_pause_command_accepts_auto_resume_duration(tmp_path: Path, capsys) -> None:
    assert main(["pause", "--state-dir", str(tmp_path), "--for", "10m"]) == 0

    payload = json.loads((tmp_path / "pause").read_text(encoding="utf-8"))
    assert payload["paused"] is True
    assert payload["pause_until"] > payload["created_at"]
    assert payload["pause_until"] - payload["created_at"] == 600

    captured = capsys.readouterr()
    assert "paused:" in captured.out
    assert "until=" in captured.out


def test_pause_command_rejects_invalid_duration(tmp_path: Path, capsys) -> None:
    assert main(["pause", "--state-dir", str(tmp_path), "--for", "0m"]) == 2
    assert not (tmp_path / "pause").exists()

    captured = capsys.readouterr()
    assert "invalid --for" in captured.err


def test_disable_and_enable_gpu_commands_update_runtime_state(tmp_path: Path, capsys) -> None:
    assert main(["disable-gpu", "--state-dir", str(tmp_path), "2"]) == 0
    assert main(["disable-gpu", "--state-dir", str(tmp_path), "0"]) == 0
    assert main(["enable-gpu", "--state-dir", str(tmp_path), "2"]) == 0

    captured = capsys.readouterr()
    assert "disabled gpu=2" in captured.out
    assert "enabled gpu=2" in captured.out
    assert json.loads((tmp_path / "disabled_gpus.json").read_text(encoding="utf-8")) == [0]


def test_disable_gpu_command_accepts_auto_enable_duration(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli.time, "time", lambda: 100.0)

    result = main(["disable-gpu", "--state-dir", str(tmp_path), "--for", "10m", "2"])

    captured = capsys.readouterr()
    payload = json.loads((tmp_path / "disabled_gpus.json").read_text(encoding="utf-8"))
    assert result == 0
    assert "disabled gpu=2" in captured.out
    assert "until=" in captured.out
    assert payload == {
        "version": 1,
        "disabled_gpus": [{"index": 2, "disabled_until": 700.0}],
    }


def test_disable_gpu_command_rejects_invalid_duration(tmp_path: Path, capsys) -> None:
    result = main(["disable-gpu", "--state-dir", str(tmp_path), "--for", "0m", "2"])

    captured = capsys.readouterr()
    assert result == 2
    assert "invalid --for" in captured.err
    assert not (tmp_path / "disabled_gpus.json").exists()


def test_status_json_outputs_status_file(tmp_path: Path, capsys) -> None:
    payload = {"timestamp": 1, "paused": False, "machine": {"average_utilization": 75}, "gpus": []}
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(["status", "--state-dir", str(tmp_path), "--json", "--stale-after", "0"])

    captured = capsys.readouterr()
    rendered = json.loads(captured.out)
    assert result == 0
    assert rendered["machine"]["average_utilization"] == 75
    assert rendered["status_stale"] is False
    assert rendered["status_stale_after_seconds"] == 0.0


def test_status_json_reports_missing_status_file_as_json(tmp_path: Path, capsys) -> None:
    result = main(["status", "--state-dir", str(tmp_path), "--json"])

    captured = capsys.readouterr()
    rendered = json.loads(captured.out)
    assert result == 1
    assert rendered["ok"] is False
    assert rendered["status_error"] == "no status file"


def test_metrics_outputs_prometheus_text(tmp_path: Path, capsys) -> None:
    payload = {
        "timestamp": 1,
        "paused": False,
        "config": {"target_util": 75},
        "machine": {"average_utilization": 75, "policy_average_utilization": 75},
        "gpus": [],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(["metrics", "--state-dir", str(tmp_path), "--stale-after", "0"])

    captured = capsys.readouterr()
    assert result == 0
    assert "# HELP gpu_holder_status_available" in captured.out
    assert "gpu_holder_status_available 1" in captured.out
    assert "gpu_holder_machine_average_utilization_percent 75" in captured.out


def test_metrics_output_writes_prometheus_text_atomically(tmp_path: Path, capsys) -> None:
    payload = {
        "timestamp": 1,
        "paused": False,
        "config": {"target_util": 75},
        "machine": {"average_utilization": 75, "policy_average_utilization": 75},
        "gpus": [],
    }
    output_path = tmp_path / "collector" / "gpu-holder.prom"
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(
        [
            "metrics",
            "--state-dir",
            str(tmp_path),
            "--stale-after",
            "0",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    rendered = output_path.read_text(encoding="utf-8")
    assert result == 0
    assert f"wrote metrics: {output_path}" in captured.out
    assert "gpu_holder_status_available 1" in rendered
    assert "gpu_holder_machine_average_utilization_percent 75" in rendered


def test_metrics_reports_missing_status(tmp_path: Path, capsys) -> None:
    result = main(["metrics", "--state-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 1
    assert "gpu_holder_status_available 0" in captured.out
    assert 'gpu_holder_status_error{error="no status file"} 1' in captured.out


def test_metrics_output_writes_missing_status_metrics(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "gpu-holder.prom"

    result = main(["metrics", "--state-dir", str(tmp_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    rendered = output_path.read_text(encoding="utf-8")
    assert result == 1
    assert f"wrote metrics: {output_path}" in captured.out
    assert "gpu_holder_status_available 0" in rendered
    assert 'gpu_holder_status_error{error="no status file"} 1' in rendered


def test_status_check_json_reports_missing_status_file_as_json(tmp_path: Path, capsys) -> None:
    result = main(["status", "--state-dir", str(tmp_path), "--check", "--json"])

    captured = capsys.readouterr()
    rendered = json.loads(captured.out)
    assert result == 1
    assert rendered == {
        "ok": False,
        "status_error": "no status file",
        "status_check": {
            "ok": False,
            "errors": ["no status file"],
        },
    }


def test_status_check_returns_zero_for_fresh_status(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    payload = {"timestamp": 100.0, "paused": False, "machine": {"average_utilization": 75}, "gpus": []}
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(cli.time, "time", lambda: 110.0)

    result = main(["status", "--state-dir", str(tmp_path), "--check", "--stale-after", "30"])

    captured = capsys.readouterr()
    assert result == 0
    assert "status_check=ok" in captured.out


def test_status_check_returns_nonzero_for_stale_status(tmp_path: Path, monkeypatch, capsys) -> None:
    payload = {"timestamp": 100.0, "paused": False, "machine": {"average_utilization": 75}, "gpus": []}
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(cli.time, "time", lambda: 200.0)

    result = main(["status", "--state-dir", str(tmp_path), "--check", "--stale-after", "30"])

    captured = capsys.readouterr()
    assert result == 2
    assert "status_check=failed" in captured.out
    assert "status file is stale" in captured.out


def test_status_check_json_includes_machine_readable_result(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    payload = {"timestamp": 100.0, "paused": False, "machine": {"average_utilization": 75}, "gpus": []}
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(cli.time, "time", lambda: 200.0)

    result = main(["status", "--state-dir", str(tmp_path), "--check", "--json", "--stale-after", "30"])

    captured = capsys.readouterr()
    rendered = json.loads(captured.out)
    assert result == 2
    assert rendered["status_check"] == {
        "ok": False,
        "errors": ["status file is stale"],
    }


def test_status_check_json_can_require_target_utilization(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    payload = {
        "timestamp": 100.0,
        "paused": False,
        "config": {"target_util": 75},
        "machine": {"policy_average_utilization": 68.5},
        "gpus": [],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(cli.time, "time", lambda: 110.0)

    result = main(
        [
            "status",
            "--state-dir",
            str(tmp_path),
            "--check",
            "--require-target",
            "--json",
            "--stale-after",
            "30",
        ]
    )

    captured = capsys.readouterr()
    rendered = json.loads(captured.out)
    assert result == 2
    assert rendered["utilization_target_health"] == {
        "ok": False,
        "status": "below_target",
        "target_util": 75.0,
        "policy_average_utilization": 68.5,
        "gap": -6.5,
    }
    assert rendered["status_check"] == {
        "ok": False,
        "errors": ["policy average utilization 68.5% is below target 75.0%"],
    }


def test_status_check_json_can_require_recoverable_forecast(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    payload = {
        "timestamp": 100.0,
        "paused": False,
        "config": {"target_util": 75},
        "machine": {
            "policy_average_utilization": 40.0,
            "policy_window_seconds": 3600.0,
            "policy_window_coverage_seconds": 3000.0,
            "policy_window_coverage_ratio": 3000.0 / 3600.0,
        },
        "gpus": [],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(cli.time, "time", lambda: 110.0)

    result = main(
        [
            "status",
            "--state-dir",
            str(tmp_path),
            "--check",
            "--require-forecast",
            "--json",
            "--stale-after",
            "30",
        ]
    )

    captured = capsys.readouterr()
    rendered = json.loads(captured.out)
    assert result == 2
    assert rendered["utilization_target_forecast"]["status"] == "at_risk"
    assert rendered["status_check"] == {
        "ok": False,
        "errors": [
            "utilization target forecast at_risk; remaining window requires 250.0% "
            "average utilization to reach target 75.0%"
        ],
    }


def test_status_check_text_accepts_recoverable_forecast(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    payload = {
        "timestamp": 100.0,
        "paused": False,
        "config": {"target_util": 75},
        "machine": {
            "policy_average_utilization": 50.0,
            "policy_window_seconds": 3600.0,
            "policy_window_coverage_seconds": 1800.0,
            "policy_window_coverage_ratio": 0.5,
        },
        "gpus": [],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(cli.time, "time", lambda: 110.0)

    result = main(
        [
            "status",
            "--state-dir",
            str(tmp_path),
            "--check",
            "--require-forecast",
            "--stale-after",
            "30",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "quota=recoverable required=100.0%" in captured.out
    assert "status_check=ok" in captured.out


def test_status_text_shows_utilization_target_health() -> None:
    lines: list[str] = []
    cli.print_status(
        {
            "timestamp": 100.0,
            "paused": False,
            "config": {"target_util": 75},
            "machine": {
                "average_utilization": 69.0,
                "policy_average_utilization": 68.5,
                "policy_window_seconds": 3600.0,
                "policy_window_sample_count": 2,
                "policy_window_coverage_seconds": 60.0,
                "policy_window_coverage_ratio": 60.0 / 3600.0,
            },
            "gpus": [],
        },
        stale_after=15.0,
        now=110.0,
        sink=lines.append,
    )

    assert "target_health=below_target" in lines[0]
    assert "target_gap=-6.5" in lines[0]
    assert "window=warming_up" in lines[0]
    assert "coverage=2%" in lines[0]
    assert "quota=recoverable" in lines[0]
    assert "required=75.1%" in lines[0]


def test_status_check_json_reports_invalid_status_file_as_json(
    tmp_path: Path,
    capsys,
) -> None:
    (tmp_path / "status.json").write_text("{not-json\n", encoding="utf-8")

    result = main(["status", "--state-dir", str(tmp_path), "--check", "--json"])

    captured = capsys.readouterr()
    rendered = json.loads(captured.out)
    assert result == 2
    assert rendered["ok"] is False
    assert "invalid status file" in rendered["status_error"]
    assert rendered["status_check"]["ok"] is False
    assert rendered["status_check"]["errors"] == [rendered["status_error"]]


def test_print_status_shows_age_and_stale_state() -> None:
    lines: list[str] = []
    cli.print_status(
        {"timestamp": 100.0, "paused": False, "machine": {"average_utilization": 75}, "gpus": []},
        stale_after=15.0,
        now=120.0,
        sink=lines.append,
    )

    assert "age=20s" in lines[0]
    assert "stale=True" in lines[0]


def test_status_text_shows_worker_backoff(tmp_path: Path, capsys) -> None:
    payload = {
        "timestamp": 1,
        "paused": False,
        "disabled_gpus": [],
        "machine": {"average_utilization": 10, "action_counts": {"hold": 1}},
        "gpus": [
            {
                "index": 0,
                "utilization": 10,
                "memory_used_human": "0.00B",
                "memory_total_human": "80.00GiB",
                "decision": {
                    "action": "hold",
                    "reason": "below_target",
                    "history_average": 10.0,
                },
                "worker": None,
                "worker_restart_backoff": {
                    "active": True,
                    "remaining_seconds": 29.7,
                },
            }
        ],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(["status", "--state-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert "worker=backoff=30s" in captured.out


def test_status_text_shows_disabled_gpu(tmp_path: Path, capsys) -> None:
    payload = {
        "timestamp": 1,
        "paused": False,
        "disabled_gpus": [0],
        "machine": {"average_utilization": 10, "action_counts": {"release": 1}},
        "gpus": [
            {
                "index": 0,
                "utilization": 10,
                "memory_used_human": "0.00B",
                "memory_total_human": "80.00GiB",
                "decision": {
                    "action": "release",
                    "reason": "gpu_disabled",
                    "history_average": 10.0,
                },
                "gpu_disabled": True,
                "worker": None,
            }
        ],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(["status", "--state-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert "disabled_gpus=[0]" in captured.out
    assert "worker=disabled" in captured.out


def test_status_text_shows_temporary_disabled_gpu_expiration(
    tmp_path: Path,
    capsys,
) -> None:
    payload = {
        "timestamp": 1,
        "paused": False,
        "disabled_gpus": [0],
        "disabled_gpu_expirations": {"0": 3600.0},
        "machine": {"average_utilization": 10, "action_counts": {"release": 1}},
        "gpus": [
            {
                "index": 0,
                "utilization": 10,
                "memory_used_human": "0.00B",
                "memory_total_human": "80.00GiB",
                "decision": {
                    "action": "release",
                    "reason": "gpu_disabled",
                    "history_average": 10.0,
                },
                "gpu_disabled": True,
                "gpu_disabled_until": 3600.0,
                "worker": None,
            }
        ],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(["status", "--state-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert "worker=disabled_until=" in captured.out
    assert "1970" in captured.out


def test_status_text_shows_process_grace(tmp_path: Path, capsys) -> None:
    payload = {
        "timestamp": 1,
        "paused": False,
        "machine": {"average_utilization": 10, "action_counts": {"release": 1}},
        "gpus": [
            {
                "index": 0,
                "utilization": 10,
                "memory_used_human": "0.00B",
                "memory_total_human": "80.00GiB",
                "decision": {
                    "action": "release",
                    "reason": "process_grace",
                    "history_average": 10.0,
                },
                "worker": None,
                "worker_restart_backoff": None,
                "process_grace": {
                    "active": True,
                    "remaining_seconds": 14.7,
                },
            }
        ],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(["status", "--state-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert "worker=process_grace=15s" in captured.out


def test_status_text_shows_thermal_block(tmp_path: Path, capsys) -> None:
    payload = {
        "timestamp": 1,
        "paused": False,
        "machine": {"average_utilization": 10, "action_counts": {"release": 1}},
        "gpus": [
            {
                "index": 0,
                "utilization": 10,
                "temperature_celsius": 83,
                "thermal_blocked": True,
                "memory_used_human": "0.00B",
                "memory_total_human": "80.00GiB",
                "decision": {
                    "action": "release",
                    "reason": "thermal_limit",
                    "history_average": 10.0,
                },
                "worker": None,
                "worker_restart_backoff": None,
                "process_grace": None,
            }
        ],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(["status", "--state-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert "temp=83C" in captured.out
    assert "worker=thermal" in captured.out


def test_status_reports_invalid_status_file_without_traceback(tmp_path: Path, capsys) -> None:
    (tmp_path / "status.json").write_text("{not-json\n", encoding="utf-8")

    result = main(["status", "--state-dir", str(tmp_path), "--check"])

    captured = capsys.readouterr()
    assert result == 2
    assert "invalid status file" in captured.out
    assert "Traceback" not in captured.err


def test_dashboard_once_renders_status_without_curses(tmp_path: Path, capsys) -> None:
    payload = {
        "timestamp": 1,
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
                "decision": {
                    "action": "hold",
                    "reason": "below_target",
                    "history_average": 40.0,
                },
                "worker": None,
                "processes": [],
            }
        ],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(["dashboard", "--state-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert "gpu-holder status snapshot" in captured.out
    assert "gpu=0 util=42%" in captured.out
    assert "reason=below_target" in captured.out


def test_dashboard_once_can_show_filtered_events(tmp_path: Path, capsys) -> None:
    payload = {
        "timestamp": 1,
        "paused": False,
        "machine": {"average_utilization": 42.0},
        "gpus": [
            {
                "index": 0,
                "name": "gpu-0",
                "utilization": 42,
                "memory_used_human": "2.00GiB",
                "memory_total_human": "80.00GiB",
                "decision": {"action": "hold", "reason": "below_target"},
                "worker": None,
                "processes": [],
            }
        ],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    from gpu_holder.events import write_event

    write_event(tmp_path / "events.jsonl", "decision", gpu_index=0, action="hold")
    write_event(tmp_path / "events.jsonl", "decision", gpu_index=1, action="standby")

    result = main(
        ["dashboard", "--state-dir", str(tmp_path), "--once", "--events", "--event-gpu", "0"]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Recent events for gpu=0" in captured.out
    assert "gpu_index=0" in captured.out
    assert "gpu_index=1" not in captured.out


def test_dashboard_once_can_show_history(tmp_path: Path, capsys) -> None:
    payload = {
        "timestamp": 1,
        "paused": False,
        "machine": {"average_utilization": 42.0},
        "gpus": [
            {
                "index": 0,
                "name": "gpu-0",
                "utilization": 42,
                "memory_used_human": "2.00GiB",
                "memory_total_human": "80.00GiB",
                "decision": {"action": "hold", "reason": "below_target"},
                "worker": None,
                "processes": [],
            }
        ],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    from gpu_holder.events import write_event

    write_event(tmp_path / "events.jsonl", "decision", gpu_index=0, action="hold", reason="below_target")
    write_event(tmp_path / "events.jsonl", "worker_start", gpu_index=0, pid=123)

    result = main(["dashboard", "--state-dir", str(tmp_path), "--once", "--history"])

    captured = capsys.readouterr()
    assert result == 0
    assert "Recent history" in captured.out
    assert "events=2 decisions=1 actions=hold:1" in captured.out
    assert "workers=start:1" in captured.out


def test_dashboard_once_can_show_explain(tmp_path: Path, capsys) -> None:
    payload = {
        "timestamp": 1,
        "paused": False,
        "config": {"target_util": 75},
        "machine": {"average_utilization": 42.0, "policy_average_utilization": 50.0},
        "gpus": [
            {
                "index": 0,
                "name": "gpu-0",
                "utilization": 42,
                "memory_used_human": "2.00GiB",
                "memory_total_human": "80.00GiB",
                "decision": {"action": "release", "reason": "busy_process"},
                "worker": None,
                "processes": [{"pid": 123, "is_holder": False}],
            }
        ],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(["dashboard", "--state-dir", str(tmp_path), "--once", "--explain"])

    captured = capsys.readouterr()
    assert result == 0
    assert "Current explanation" in captured.out
    assert "finding[warning]: below_target" in captured.out
    assert "finding[info]: busy_process" in captured.out


def test_dashboard_once_can_show_advice(tmp_path: Path, capsys) -> None:
    payload = {
        "timestamp": 1,
        "paused": False,
        "config": {
            "target_util": 60,
            "program": "matmul",
            "mem": "20%",
        },
        "machine": {"average_utilization": 42.0},
        "gpus": [],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(["dashboard", "--state-dir", str(tmp_path), "--once", "--advice"])

    captured = capsys.readouterr()
    assert result == 0
    assert "Tuning advice" in captured.out
    assert "grade=warn" in captured.out
    assert "check=program_diversity status=warn" in captured.out


def test_dashboard_once_reports_missing_status_file(tmp_path: Path, capsys) -> None:
    result = main(["dashboard", "--state-dir", str(tmp_path), "--once"])

    captured = capsys.readouterr()
    assert result == 1
    assert "no status file" in captured.out


def test_dashboard_once_reports_invalid_status_file(tmp_path: Path, capsys) -> None:
    (tmp_path / "status.json").write_text("{not-json\n", encoding="utf-8")

    result = main(["dashboard", "--state-dir", str(tmp_path), "--once"])

    captured = capsys.readouterr()
    assert result == 1
    assert "invalid status file" in captured.out
    assert "Traceback" not in captured.err


def test_status_watch_prints_multiple_snapshots(tmp_path: Path, capsys) -> None:
    payload = {
        "timestamp": 1,
        "paused": False,
        "machine": {"average_utilization": 75, "action_counts": {"standby": 1}},
        "gpus": [],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(
        [
            "status",
            "--state-dir",
            str(tmp_path),
            "--watch",
            "--count",
            "2",
            "--interval",
            "0",
            "--no-clear",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out.count("machine_avg=75") == 2


def test_status_watch_json_outputs_json_lines(tmp_path: Path, capsys) -> None:
    payload = {
        "timestamp": 1,
        "paused": False,
        "machine": {"average_utilization": 75},
        "gpus": [],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(
        [
            "status",
            "--state-dir",
            str(tmp_path),
            "--watch",
            "--count",
            "2",
            "--interval",
            "0",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
    assert result == 0
    assert [line["timestamp"] for line in lines] == [1, 1]


def test_status_watch_json_reports_missing_status_file_as_json(tmp_path: Path, capsys) -> None:
    result = main(
        [
            "status",
            "--state-dir",
            str(tmp_path),
            "--watch",
            "--count",
            "1",
            "--interval",
            "0",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
    assert result == 1
    assert lines == [{"ok": False, "status_error": "no status file"}]


def test_events_json_outputs_event_log(tmp_path: Path, capsys) -> None:
    events = [
        {"timestamp": 1, "type": "decision", "gpu_index": 0, "action": "hold"},
        {"timestamp": 2, "type": "worker_stop", "gpu_index": 0, "pid": 99},
    ]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    result = main(["events", "--state-dir", str(tmp_path), "--limit", "1", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert len(payload) == 1
    assert payload[0]["type"] == "worker_stop"


def test_events_json_filters_by_type_and_gpu_before_limit(tmp_path: Path, capsys) -> None:
    events = [
        {"timestamp": 1, "type": "decision", "gpu_index": 0, "action": "hold"},
        {"timestamp": 2, "type": "worker_stop", "gpu_index": 0, "pid": 99},
        {"timestamp": 3, "type": "decision", "gpu_index": 1, "action": "release"},
        {"timestamp": 4, "type": "decision", "gpu_index": 0, "action": "assist"},
    ]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    result = main(
        [
            "events",
            "--state-dir",
            str(tmp_path),
            "--event-type",
            "decision",
            "--gpu",
            "0",
            "--limit",
            "1",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload == [{"timestamp": 4, "type": "decision", "gpu_index": 0, "action": "assist"}]


def test_events_text_filters_multiple_event_types(tmp_path: Path, capsys) -> None:
    events = [
        {"timestamp": 1, "type": "decision", "gpu_index": 0, "action": "hold"},
        {"timestamp": 2, "type": "worker_stop", "gpu_index": 0, "pid": 99},
        {"timestamp": 3, "type": "thermal_limit_start", "gpu_index": 0},
    ]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    result = main(
        [
            "events",
            "--state-dir",
            str(tmp_path),
            "--event-type",
            "worker_stop,thermal_limit_start",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "type=decision" not in captured.out
    assert "type=worker_stop" in captured.out
    assert "type=thermal_limit_start" in captured.out


def test_history_command_summarizes_events(tmp_path: Path, capsys) -> None:
    events = [
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
    ]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    result = main(["history", "--state-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert "history events=4 decisions=2" in captured.out
    assert "actions=hold:1,release:1" in captured.out
    assert "gpu=0" in captured.out
    assert "workers=start:1" in captured.out


def test_history_command_json_filters_by_since_and_gpu(tmp_path: Path, capsys) -> None:
    events = [
        {"timestamp": 1, "type": "decision", "gpu_index": 0, "action": "hold"},
        {"timestamp": 8, "type": "decision", "gpu_index": 1, "action": "release"},
        {"timestamp": 10, "type": "worker_stop", "gpu_index": 0},
    ]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    result = main(
        ["history", "--state-dir", str(tmp_path), "--since", "5s", "--gpu", "0", "--json"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["event_count"] == 1
    assert payload["event_type_counts"] == {"worker_stop": 1}
    assert payload["gpu_indices"] == [0]


def test_history_command_rejects_invalid_since(tmp_path: Path, capsys) -> None:
    result = main(["history", "--state-dir", str(tmp_path), "--since", "0s"])

    captured = capsys.readouterr()
    assert result == 2
    assert "invalid --since" in captured.err


def test_explain_command_reports_status_findings(tmp_path: Path, capsys) -> None:
    payload = {
        "timestamp": 100.0,
        "paused": False,
        "config": {"target_util": 75},
        "machine": {"policy_average_utilization": 60},
        "disabled_gpus": [],
        "gpus": [
            {
                "index": 0,
                "decision": {"action": "release", "reason": "busy_process"},
                "processes": [{"pid": 123, "is_holder": False}],
            }
        ],
    }
    (tmp_path / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    result = main(["explain", "--state-dir", str(tmp_path), "--stale-after", "0"])

    captured = capsys.readouterr()
    assert result == 0
    assert "gpu-holder explain ok=True" in captured.out
    assert "finding[warning]: below_target" in captured.out
    assert "finding[info]: busy_process" in captured.out


def test_explain_command_json_reports_missing_status(tmp_path: Path, capsys) -> None:
    result = main(["explain", "--state-dir", str(tmp_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert result == 1
    assert payload["ok"] is False
    assert payload["findings"][0]["code"] == "status_missing"


def test_events_follow_reads_new_events_only(tmp_path: Path, capsys) -> None:
    first = {"timestamp": 1, "type": "decision", "gpu_index": 0, "action": "hold"}
    second = {"timestamp": 2, "type": "worker_stop", "gpu_index": 0, "pid": 99}
    event_log = tmp_path / "events.jsonl"
    event_log.write_text(json.dumps(first) + "\n", encoding="utf-8")
    appended = False

    def after_poll(iteration: int) -> None:
        nonlocal appended
        if iteration == 0 and not appended:
            with event_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(second) + "\n")
            appended = True

    result = cli.follow_events(
        event_log=event_log,
        limit=10,
        json_output=False,
        interval=0,
        count=2,
        sink=print,
        after_poll=after_poll,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "type=decision" in captured.out
    assert "type=worker_stop" in captured.out
    assert captured.out.count("type=decision") == 1


def test_events_follow_filters_new_events(tmp_path: Path, capsys) -> None:
    first = {"timestamp": 1, "type": "decision", "gpu_index": 0, "action": "hold"}
    ignored = {"timestamp": 2, "type": "worker_stop", "gpu_index": 0, "pid": 99}
    selected = {"timestamp": 3, "type": "decision", "gpu_index": 1, "action": "release"}
    event_log = tmp_path / "events.jsonl"
    event_log.write_text(json.dumps(first) + "\n", encoding="utf-8")
    appended = False

    def after_poll(iteration: int) -> None:
        nonlocal appended
        if iteration == 0 and not appended:
            with event_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(ignored) + "\n")
                handle.write(json.dumps(selected) + "\n")
            appended = True

    result = cli.follow_events(
        event_log=event_log,
        limit=10,
        json_output=False,
        interval=0,
        count=2,
        sink=print,
        after_poll=after_poll,
        event_types={"decision"},
        gpu_indices={1},
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "gpu_index=0" not in captured.out
    assert "gpu_index=1" in captured.out
    assert "type=worker_stop" not in captured.out


def test_events_follow_json_reads_new_events_as_json_lines(tmp_path: Path, capsys) -> None:
    event = {"timestamp": 1, "type": "decision", "gpu_index": 0, "action": "hold"}
    event_log = tmp_path / "events.jsonl"
    event_log.write_text(json.dumps(event) + "\n", encoding="utf-8")

    result = cli.follow_events(
        event_log=event_log,
        limit=10,
        json_output=True,
        interval=0,
        count=1,
        sink=print,
    )

    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
    assert result == 0
    assert lines == [event]


def test_report_json_collects_status_events_and_diagnostics(tmp_path: Path, monkeypatch, capsys) -> None:
    status = {
        "timestamp": 1,
        "paused": False,
        "holder_pids": [123],
        "machine": {
            "average_utilization": 75,
            "policy_average_utilization": 74,
            "policy_window_seconds": 3600.0,
            "policy_window_sample_count": 120,
            "policy_window_coverage_seconds": 3500.0,
            "policy_window_coverage_ratio": 3500.0 / 3600.0,
            "action_counts": {"hold": 1, "standby": 1},
        },
        "config": {"target_util": 75},
        "gpus": [
            {
                "index": 0,
                "decision": {"action": "hold", "reason": "below_target"},
                "worker": {"pid": 123},
                "processes": [
                    {"pid": 123, "name": "gpu-holder", "is_holder": True},
                    {"pid": 456, "name": "train.py", "is_holder": False},
                ],
            },
            {
                "index": 1,
                "decision": {"action": "standby", "reason": "target_met"},
                "worker": None,
                "processes": [],
            },
        ],
    }
    event = {"timestamp": 2, "type": "decision", "gpu_index": 0, "action": "hold"}
    event2 = {"timestamp": 3, "type": "worker_start", "gpu_index": 0, "pid": 123}
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(
        json.dumps(event) + "\n" + json.dumps(event2) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "gpu-holder.pid").write_text("12345", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "run_diagnostics",
        lambda: {"ok": True, "checks": [{"name": "python", "ok": True, "detail": "3.11"}]},
    )

    result = main(["report", "--state-dir", str(tmp_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["schema_version"] == 1
    assert payload["diagnostics"]["ok"] is True
    assert payload["status"]["machine"]["average_utilization"] == 75
    assert payload["events"] == [event, event2]
    assert payload["pid_file"]["pid"] == 12345
    assert payload["summary"] == {
        "diagnostics_ok": True,
        "status_present": True,
        "status_error": None,
        "event_count": 2,
        "event_type_counts": {"decision": 1, "worker_start": 1},
        "gpu_count": 2,
        "machine_average_utilization": 75,
        "machine_policy_average_utilization": 74,
        "policy_window_health": {
            "ok": True,
            "status": "ready",
            "window_seconds": 3600.0,
            "sample_count": 120,
            "coverage_seconds": 3500.0,
            "coverage_ratio": 0.972222,
        },
        "utilization_target_forecast": {
            "ok": False,
            "status": "at_risk",
            "target_util": 75.0,
            "policy_average_utilization": 74.0,
            "window_seconds": 3600.0,
            "coverage_seconds": 3500.0,
            "coverage_ratio": 0.972222,
            "remaining_seconds": 100.0,
            "required_remaining_average": 110.0,
        },
        "utilization_target_health": {
            "ok": False,
            "status": "below_target",
            "target_util": 75.0,
            "policy_average_utilization": 74.0,
            "gap": -1.0,
        },
        "action_counts": {"hold": 1, "standby": 1},
        "decision_reason_counts": {"below_target": 1, "target_met": 1},
        "owned_worker_count": 1,
        "holder_pid_count": 1,
        "external_process_count": 1,
        "disabled_gpu_count": 0,
    }


def test_report_json_can_redact_local_paths_and_process_identity(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    status = {
        "timestamp": 1,
        "paused": False,
        "holder_pids": [1111],
        "machine": {"average_utilization": 75},
        "config": {
            "target_util": 75,
            "state_dir": str(tmp_path),
            "status_file": str(tmp_path / "status.json"),
            "event_log_file": str(tmp_path / "events.jsonl"),
            "log_file": str(tmp_path / "gpu-holder.log"),
        },
        "gpus": [
            {
                "index": 0,
                "processes": [
                    {
                        "pid": 2222,
                        "name": "secret-train.py",
                        "used_memory_bytes": 1024,
                        "is_holder": False,
                    }
                ],
                "worker": {"pid": 3333, "hold_mode": "balanced"},
            }
        ],
    }
    event = {
        "timestamp": 2,
        "type": "process_grace_start",
        "gpu_index": 0,
        "pid": 2222,
        "pids": [2222],
    }
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    (tmp_path / "gpu-holder.pid").write_text("12345", encoding="utf-8")
    monkeypatch.setattr(cli, "run_diagnostics", lambda: {"ok": True, "checks": []})

    result = main(["report", "--state-dir", str(tmp_path), "--redact", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["redacted"] is True
    assert payload["summary"]["external_process_count"] == 1
    assert payload["state_dir"] == "<redacted>"
    assert payload["files"]["status_file"]["path"] == "<redacted>"
    assert payload["pid_file"]["pid"] == "<redacted>"
    assert payload["status"]["holder_pids"] == ["<redacted>"]
    assert payload["status"]["config"]["state_dir"] == "<redacted>"
    assert payload["status"]["config"]["status_file"] == "<redacted>"
    process = payload["status"]["gpus"][0]["processes"][0]
    assert process["pid"] == "<redacted>"
    assert process["name"] == "<redacted>"
    assert process["used_memory_bytes"] == 1024
    assert payload["status"]["gpus"][0]["worker"]["pid"] == "<redacted>"
    assert payload["events"][0]["pid"] == "<redacted>"
    assert payload["events"][0]["pids"] == ["<redacted>"]


def test_report_json_includes_invalid_status_file_error(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "status.json").write_text("{not-json\n", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "run_diagnostics",
        lambda: {"ok": True, "checks": []},
    )

    result = main(["report", "--state-dir", str(tmp_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["status"] is None
    assert "invalid status file" in payload["status_error"]


def test_report_text_shows_invalid_status_file_error(tmp_path: Path, monkeypatch, capsys) -> None:
    (tmp_path / "status.json").write_text("{not-json\n", encoding="utf-8")
    monkeypatch.setattr(cli, "run_diagnostics", lambda: {"ok": True, "checks": []})

    result = main(["report", "--state-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert "status_error=invalid status file" in captured.out


def test_report_can_skip_diagnostics(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "run_diagnostics",
        lambda: (_ for _ in ()).throw(AssertionError("diagnostics must be skipped")),
    )

    result = main(["report", "--state-dir", str(tmp_path), "--no-diagnostics", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["diagnostics"] is None
    assert payload["status"] is None
    assert payload["events"] == []


def test_report_text_summarizes_bundle(tmp_path: Path, monkeypatch, capsys) -> None:
    status = {
        "timestamp": 1,
        "paused": False,
        "config": {"target_util": 75},
        "machine": {
            "policy_average_utilization": 70.0,
            "policy_window_seconds": 3600.0,
            "policy_window_sample_count": 2,
            "policy_window_coverage_seconds": 60.0,
            "policy_window_coverage_ratio": 60.0 / 3600.0,
        },
        "gpus": [],
    }
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"timestamp": 1, "type": "controller_start"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "run_diagnostics", lambda: {"ok": False, "checks": []})

    result = main(["report", "--state-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert "gpu-holder report" in captured.out
    assert "diagnostics_ok=False" in captured.out
    assert "events=1" in captured.out
    assert "window=warming_up coverage=2%" in captured.out
    assert "quota=recoverable required=75.1%" in captured.out


def test_doctor_json_uses_diagnostics_payload(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "run_diagnostics",
        lambda: {"ok": True, "checks": [{"name": "python", "ok": True, "detail": "3.11"}]},
    )

    result = main(["doctor", "--json"])

    captured = capsys.readouterr()
    assert result == 0
    assert json.loads(captured.out)["checks"][0]["name"] == "python"


def test_doctor_text_prints_actionable_suggestions(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "run_diagnostics",
        lambda: {
            "ok": False,
            "checks": [
                {
                    "name": "pynvml",
                    "ok": False,
                    "detail": "ModuleNotFoundError: No module named pynvml",
                    "suggestion": "Install NVML bindings with: python -m pip install 'gpu-holder[monitor]'",
                }
            ],
        },
    )

    result = main(["doctor"])

    captured = capsys.readouterr()
    assert result == 1
    assert "pynvml: fail" in captured.out
    assert "suggestion: Install NVML bindings" in captured.out
