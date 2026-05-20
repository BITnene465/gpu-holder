from __future__ import annotations

from . import __version__
from .completion import command_reference


def generate_manual(output_format: str) -> str:
    if output_format == "markdown":
        return manual_markdown()
    if output_format == "man":
        return manual_roff()
    raise ValueError(f"unsupported manual format: {output_format}")


def manual_markdown() -> str:
    lines = [
        "# gpu-holder manual",
        "",
        f"Version: `{__version__}`",
        "",
        "A polite GPU guard for shared NVIDIA machines. It keeps utilization high",
        "while yielding to real workloads and never controls external GPU processes.",
        "",
        "## Synopsis",
        "",
        "```bash",
        "gpu-holder <command> [options]",
        "```",
        "",
        "## Commands",
        "",
    ]
    for command in command_reference():
        lines.extend(_markdown_command(command))
    lines.extend(
        [
            "## Safety",
            "",
            "`gpu-holder` starts and stops only worker processes that it owns.",
            "External GPU processes are read-only scheduling signals.",
            "",
        ]
    )
    return "\n".join(lines)


def manual_roff() -> str:
    lines = [
        '.TH "GPU-HOLDER" "1"',
        ".SH NAME",
        "gpu-holder \\- polite GPU utilization guard",
        ".SH SYNOPSIS",
        ".B gpu-holder",
        "<command> [options]",
        ".SH DESCRIPTION",
        "gpu-holder keeps shared NVIDIA GPU utilization high while yielding to real workloads.",
        "It starts and stops only worker processes that it owns.",
        ".SH COMMANDS",
    ]
    for command in command_reference():
        lines.extend(_roff_command(command))
    lines.extend(
        [
            ".SH SAFETY",
            "External GPU processes are read-only scheduling signals and are never control targets.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_command(command: dict[str, object]) -> list[str]:
    name = str(command["name"])
    description = str(command["description"])
    flags = command.get("flags")
    lines = [
        f"### `{name}`",
        "",
        description,
        "",
    ]
    if isinstance(flags, list) and flags:
        lines.append("Options:")
        lines.append("")
        for flag in flags:
            lines.append(f"- `{flag}`")
        lines.append("")
    return lines


def _roff_command(command: dict[str, object]) -> list[str]:
    name = str(command["name"])
    description = str(command["description"])
    flags = command.get("flags")
    lines = [
        f'.SS "{name}"',
        description,
    ]
    if isinstance(flags, list) and flags:
        lines.append(".P")
        lines.append("Options: " + ", ".join(str(flag) for flag in flags))
    return lines
