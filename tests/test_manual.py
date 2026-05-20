from __future__ import annotations

from gpu_holder.manual import generate_manual


def test_markdown_manual_lists_commands_options_and_safety() -> None:
    manual = generate_manual("markdown")

    assert "# gpu-holder manual" in manual
    assert "### `guard`" in manual
    assert "### `preflight`" in manual
    assert "- `--no-diagnostics`" in manual
    assert "### `simulate`" in manual
    assert "- `--scenario`" in manual
    assert "### `tune`" in manual
    assert "- `--strict`" in manual
    assert "- `--compare-profiles`" in manual
    assert "- `--target-util`" in manual
    assert "### `recipes`" in manual
    assert "- `--name`" in manual
    assert "### `reason-reference`" in manual
    assert "### `service`" in manual
    assert "- `--target-util`" in manual
    assert "- `--require-target`" in manual
    assert "- `--require-forecast`" in manual
    assert "- `--advice`" in manual
    assert "### `alerts`" in manual
    assert "- `--group-name`" in manual
    assert "### `grafana-dashboard`" in manual
    assert "- `--datasource`" in manual
    assert "### `monitoring-bundle`" in manual
    assert "- `--output-dir`" in manual
    assert "### `events`" in manual
    assert "- `--event-type`" in manual
    assert "### `report`" in manual
    assert "- `--redact`" in manual
    assert "External GPU processes are read-only scheduling signals" in manual


def test_roff_manual_contains_manpage_sections() -> None:
    manual = generate_manual("man")

    assert '.TH "GPU-HOLDER" "1"' in manual
    assert ".SH COMMANDS" in manual
    assert '.SS "dashboard"' in manual
    assert ".SH SAFETY" in manual


def test_manual_rejects_unsupported_format() -> None:
    try:
        generate_manual("html")
    except ValueError as exc:
        assert "unsupported manual format" in str(exc)
    else:
        raise AssertionError("unsupported manual format must raise ValueError")
