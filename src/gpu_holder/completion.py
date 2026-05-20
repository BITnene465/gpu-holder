from __future__ import annotations

from .config import config_reference


COMMANDS: dict[str, str] = {
    "guard": "run the foreground controller",
    "plan": "print policy decisions without workers",
    "preflight": "validate startup readiness without workers",
    "start": "start a lightweight background daemon",
    "config": "validate and print the resolved guard config",
    "config-reference": "print documented guard config fields",
    "profile-reference": "print documented runtime profiles",
    "recipes": "print scenario-oriented config recipes",
    "reason-reference": "print documented policy decision reasons",
    "init-config": "write a default TOML config file",
    "simulate": "simulate policy decisions from a trace",
    "tune": "score config against offline policy scenarios",
    "stop": "stop the background daemon",
    "status": "show the latest status file",
    "metrics": "export latest status as Prometheus text",
    "alerts": "print Prometheus alert rules",
    "grafana-dashboard": "print a Grafana dashboard JSON",
    "monitoring-bundle": "write Prometheus alerts and Grafana dashboard files",
    "events": "show recent controller events",
    "history": "summarize recent controller event history",
    "explain": "explain current guard state and suggested actions",
    "dashboard": "print a terminal status snapshot",
    "pause": "pause all holder workers",
    "resume": "resume holder workers",
    "disable-gpu": "release and skip one GPU at runtime",
    "enable-gpu": "allow holder workers on one GPU again",
    "doctor": "diagnose NVML/PyTorch runtime readiness",
    "report": "build a read-only support report",
    "service": "print a systemd user service unit",
    "completion": "print shell completion script",
    "manual": "print generated command manual",
}


GUARD_LIKE_EXTRA_FLAGS = [
    "--config",
    "--dry-run",
    "--fake",
    "--json",
    "--log-interval",
]


COMMAND_FLAGS: dict[str, list[str]] = {
    "guard": [],
    "plan": [],
    "preflight": ["--no-diagnostics"],
    "start": [],
    "config": ["--explain"],
    "simulate": ["--trace", "--scenario"],
    "tune": ["--strict", "--compare-profiles"],
    "config-reference": ["--json"],
    "profile-reference": ["--json"],
    "recipes": ["--name", "--json"],
    "reason-reference": ["--json"],
    "init-config": ["--path", "--profile", "--minimal", "--force", "--stdout"],
    "stop": ["--state-dir"],
    "status": [
        "--state-dir",
        "--json",
        "--check",
        "--require-target",
        "--require-forecast",
        "--watch",
        "--interval",
        "--count",
        "--no-clear",
        "--stale-after",
    ],
    "metrics": ["--state-dir", "--stale-after", "--output"],
    "alerts": ["--json", "--group-name"],
    "grafana-dashboard": ["--title", "--uid", "--datasource"],
    "monitoring-bundle": [
        "--output-dir",
        "--group-name",
        "--dashboard-title",
        "--dashboard-uid",
        "--datasource",
        "--json",
    ],
    "events": ["--state-dir", "--limit", "--json", "--event-type", "--gpu", "--follow", "--interval", "--count"],
    "history": ["--state-dir", "--limit", "--since", "--gpu", "--json"],
    "explain": ["--state-dir", "--limit", "--stale-after", "--json"],
    "dashboard": [
        "--state-dir",
        "--once",
        "--events",
        "--history",
        "--explain",
        "--advice",
        "--history-limit",
        "--config",
        "--details",
        "--event-gpu",
    ],
    "pause": ["--state-dir", "--for"],
    "resume": ["--state-dir"],
    "disable-gpu": ["--state-dir", "--for"],
    "enable-gpu": ["--state-dir"],
    "doctor": ["--json"],
    "report": ["--state-dir", "--limit", "--no-diagnostics", "--redact", "--json"],
    "service": [
        "--config",
        "--state-dir",
        "--working-directory",
        "--executable",
        "--restart-sec",
        "--description",
        "--extra-args",
    ],
    "completion": ["bash", "zsh", "fish"],
    "manual": ["--format"],
}


def command_reference() -> list[dict[str, object]]:
    flags_by_command = _command_flags()
    return [
        {
            "name": name,
            "description": description,
            "flags": flags_by_command.get(name, []),
        }
        for name, description in COMMANDS.items()
    ]


def generate_completion(shell: str) -> str:
    if shell == "bash":
        return bash_completion()
    if shell == "zsh":
        return zsh_completion()
    if shell == "fish":
        return fish_completion()
    raise ValueError(f"unsupported shell: {shell}")


def bash_completion() -> str:
    commands = " ".join(COMMANDS)
    guard_like_options = " ".join(_guard_like_flags())
    cases = []
    for command, flags in _command_flags().items():
        options = " ".join(flags)
        cases.append(f"    {command}) opts=\"{options}\" ;;")
    cases_text = "\n".join(cases)
    return f"""# bash completion for gpu-holder
_gpu_holder_completion() {{
    local cur command opts
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    command="${{COMP_WORDS[1]}}"

    if [[ $COMP_CWORD -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "{commands}" -- "$cur") )
        return 0
    fi

    opts=""
    case "$command" in
{cases_text}
    esac

    if [[ -z "$opts" && "$command" =~ ^(guard|plan|preflight|start|config|simulate|tune)$ ]]; then
        opts="{guard_like_options}"
    fi

    COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
    return 0
}}
complete -F _gpu_holder_completion gpu-holder
"""


def zsh_completion() -> str:
    command_entries = " ".join(
        f"'{name}:{description}'" for name, description in COMMANDS.items()
    )
    option_entries = " ".join(f"'{flag}'" for flag in _all_flags())
    return f"""#compdef gpu-holder
# zsh completion for gpu-holder
_gpu_holder() {{
  local -a commands options
  commands=({command_entries})
  options=({option_entries})

  if (( CURRENT == 2 )); then
    _describe 'command' commands
  else
    _describe 'option' options
  fi
}}
_gpu_holder "$@"
"""


def fish_completion() -> str:
    lines = [
        "# fish completion for gpu-holder",
        "complete -c gpu-holder -f",
    ]
    for command, description in COMMANDS.items():
        lines.append(
            f"complete -c gpu-holder -n '__fish_use_subcommand' -a '{command}' -d '{description}'"
        )
    for flag in _all_flags():
        if flag.startswith("--"):
            lines.append(f"complete -c gpu-holder -l {flag[2:]}")
    return "\n".join(lines) + "\n"


def _guard_like_flags() -> list[str]:
    config_flags = [
        str(field["cli_flag"])
        for field in config_reference()
        if isinstance(field.get("cli_flag"), str)
    ]
    return sorted(set(config_flags + GUARD_LIKE_EXTRA_FLAGS))


def _command_flags() -> dict[str, list[str]]:
    flags = {command: list(values) for command, values in COMMAND_FLAGS.items()}
    for command in ("guard", "plan", "preflight", "start", "config", "simulate", "tune"):
        flags[command] = sorted(set(flags.get(command, []) + _guard_like_flags()))
    return flags


def _all_flags() -> list[str]:
    values: set[str] = set()
    for flags in _command_flags().values():
        values.update(flag for flag in flags if flag.startswith("--"))
    return sorted(values)
