from __future__ import annotations

from gpu_holder.completion import command_reference, generate_completion


def test_command_reference_lists_manual_and_completion_metadata() -> None:
    commands = {command["name"]: command for command in command_reference()}

    assert commands["preflight"]["description"] == "validate startup readiness without workers"
    assert "--no-diagnostics" in commands["preflight"]["flags"]
    assert commands["profile-reference"]["description"] == "print documented runtime profiles"
    assert commands["profile-reference"]["flags"] == ["--json"]
    assert commands["reason-reference"]["description"] == "print documented policy decision reasons"
    assert commands["reason-reference"]["flags"] == ["--json"]
    assert commands["recipes"]["description"] == "print scenario-oriented config recipes"
    assert "--name" in commands["recipes"]["flags"]
    assert "--stdout" in commands["init-config"]["flags"]
    assert "--profile" in commands["init-config"]["flags"]
    assert "--minimal" in commands["init-config"]["flags"]
    assert "--explain" in commands["config"]["flags"]
    assert "--scenario" in commands["simulate"]["flags"]
    assert commands["tune"]["description"] == "score config against offline policy scenarios"
    assert "--json" in commands["tune"]["flags"]
    assert "--strict" in commands["tune"]["flags"]
    assert "--compare-profiles" in commands["tune"]["flags"]
    assert "--target-util" in commands["tune"]["flags"]
    assert commands["manual"]["description"] == "print generated command manual"
    assert "--format" in commands["manual"]["flags"]
    assert "--config" in commands["service"]["flags"]
    assert "--extra-args" in commands["service"]["flags"]
    assert "--redact" in commands["report"]["flags"]
    assert "--require-target" in commands["status"]["flags"]
    assert "--require-forecast" in commands["status"]["flags"]
    assert commands["metrics"]["description"] == "export latest status as Prometheus text"
    assert "--stale-after" in commands["metrics"]["flags"]
    assert "--output" in commands["metrics"]["flags"]
    assert commands["alerts"]["description"] == "print Prometheus alert rules"
    assert "--group-name" in commands["alerts"]["flags"]
    assert commands["grafana-dashboard"]["description"] == "print a Grafana dashboard JSON"
    assert "--datasource" in commands["grafana-dashboard"]["flags"]
    assert commands["monitoring-bundle"]["description"] == (
        "write Prometheus alerts and Grafana dashboard files"
    )
    assert "--output-dir" in commands["monitoring-bundle"]["flags"]
    assert "--for" in commands["pause"]["flags"]
    assert "--for" in commands["disable-gpu"]["flags"]
    assert commands["history"]["description"] == "summarize recent controller event history"
    assert "--since" in commands["history"]["flags"]
    assert commands["explain"]["description"] == "explain current guard state and suggested actions"
    assert "--stale-after" in commands["explain"]["flags"]
    assert "--history" in commands["dashboard"]["flags"]
    assert "--explain" in commands["dashboard"]["flags"]
    assert "--advice" in commands["dashboard"]["flags"]
    assert "--history-limit" in commands["dashboard"]["flags"]
    assert "--temporary-disable-duration" not in commands["dashboard"]["flags"]
    assert "--temporary-pause-duration" not in commands["dashboard"]["flags"]
    assert commands["completion"]["flags"] == ["bash", "zsh", "fish"]


def test_bash_completion_contains_commands_and_guard_options() -> None:
    script = generate_completion("bash")

    assert "complete -F _gpu_holder_completion gpu-holder" in script
    assert "guard plan preflight start config" in script
    assert "--target-util" in script
    assert "--profile" in script
    assert "--protected-process" in script
    assert "--thermal-resume-temp" in script
    assert "events)" in script
    assert "tune)" in script
    assert "metrics)" in script
    assert "alerts)" in script
    assert "grafana-dashboard)" in script
    assert "monitoring-bundle)" in script
    assert "--event-type" in script
    assert "history)" in script
    assert "--history-limit" in script
    assert "--explain" in script


def test_zsh_completion_contains_command_descriptions_and_options() -> None:
    script = generate_completion("zsh")

    assert "#compdef gpu-holder" in script
    assert "'dashboard:print a terminal status snapshot'" in script
    assert "'--state-dir'" in script
    assert "'--once'" in script
    assert "'--check'" in script
    assert "'--require-target'" in script
    assert "'--require-forecast'" in script


def test_fish_completion_contains_subcommands_and_long_options() -> None:
    script = generate_completion("fish")

    assert "complete -c gpu-holder -f" in script
    assert "-a 'doctor'" in script
    assert "-l max-gpu-temp" in script
    assert "-l event-type" in script


def test_completion_rejects_unsupported_shell() -> None:
    try:
        generate_completion("powershell")
    except ValueError as exc:
        assert "unsupported shell" in str(exc)
    else:
        raise AssertionError("unsupported shell must raise ValueError")
