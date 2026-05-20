from __future__ import annotations

from pathlib import Path
import re

from gpu_holder.cli import build_parser


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_exposes_small_cli_metadata() -> None:
    content = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'gpu-holder = "gpu_holder.cli:main"' in content
    assert 'requires-python = ">=3.10,<3.11"' in content
    assert '"Programming Language :: Python :: 3.10"' in content
    assert '"Programming Language :: Python :: 3.11"' not in content
    assert '"build>=1.0.0"' in content
    assert re.search(r'^license\s*=\s*\{ text = "MIT" \}', content, re.MULTILINE)


def test_source_tree_is_trimmed_to_cli_and_worker() -> None:
    files = {
        path.relative_to(ROOT / "src" / "gpu_holder").as_posix()
        for path in (ROOT / "src" / "gpu_holder").glob("*.py")
    }

    assert files == {"__init__.py", "__main__.py", "cli.py", "worker.py"}


def test_parser_only_exposes_current_commands() -> None:
    parser = build_parser()
    command_actions = [
        action for action in parser._actions if getattr(action, "dest", None) == "command"
    ]

    assert len(command_actions) == 1
    assert set(command_actions[0].choices) == {
        "guard",
        "start",
        "stop",
        "status",
        "dashboard",
        "doctor",
    }


def test_readme_does_not_document_deleted_features() -> None:
    content = (ROOT / "README.md").read_text(encoding="utf-8").lower()

    assert "gpu-holder guard --gpus 0-7 --mem 0.2 --target-util 0.75" in content
    for deleted_feature in [
        "plan --fake",
        "preflight",
        "tune",
        "init-config",
        "recipes",
        "grafana",
        "monitoring-bundle",
        "alerts",
        "metrics",
    ]:
        assert deleted_feature not in content
