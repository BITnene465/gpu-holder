from __future__ import annotations

from pathlib import Path


def generate_systemd_user_service(
    *,
    executable: str = "gpu-holder",
    config_path: str | None = None,
    state_dir: str | None = None,
    working_directory: str | None = None,
    restart_seconds: int = 10,
    extra_args: tuple[str, ...] = (),
    description: str = "gpu-holder guard",
) -> str:
    command = [_quote_systemd_arg(executable), "guard"]
    if config_path:
        command.extend(["--config", _quote_systemd_arg(str(Path(config_path).expanduser()))])
    if state_dir:
        command.extend(["--state-dir", _quote_systemd_arg(str(Path(state_dir).expanduser()))])
    command.extend(_quote_systemd_arg(arg) for arg in extra_args)

    lines = [
        "[Unit]",
        f"Description={description}",
        "After=default.target",
        "",
        "[Service]",
        "Type=simple",
    ]
    if working_directory:
        lines.append(f"WorkingDirectory={_quote_systemd_arg(str(Path(working_directory).expanduser()))}")
    lines.extend(
        [
            f"ExecStart={' '.join(command)}",
            "Restart=always",
            f"RestartSec={max(0, int(restart_seconds))}",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )
    return "\n".join(lines)


def _quote_systemd_arg(value: str) -> str:
    text = str(value)
    if text == "":
        return '""'
    if not any(character.isspace() or character in {'"', "\\"} for character in text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
