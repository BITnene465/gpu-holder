from __future__ import annotations

from pathlib import Path
import re

from gpu_holder.cli import build_parser


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_exposes_small_cli_metadata() -> None:
    content = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'gpu-holder = "gpu_holder.cli:main"' in content
    assert 'requires-python = ">=3.10,<3.11"' in content
    assert 'Repository = "https://github.com/BITnene465/gpu-holder"' in content
    assert '{ name = "BITnene465", email = "nene465suki@gmail.com" }' in content
    assert '"Programming Language :: Python :: 3.10"' in content
    assert '"Programming Language :: Python :: 3.11"' not in content
    assert '"build>=1.0.0"' in content
    assert re.search(r'^license\s*=\s*"MIT"', content, re.MULTILINE)


def test_source_tree_is_trimmed_to_cli_and_worker() -> None:
    files = {
        path.relative_to(ROOT / "src" / "gpu_holder").as_posix()
        for path in (ROOT / "src" / "gpu_holder").glob("*.py")
    }

    assert files == {
        "__init__.py",
        "__main__.py",
        "backends.py",
        "cli.py",
        "driver_backend.py",
        "models.py",
        "policy.py",
        "telemetry.py",
        "torch_backend.py",
        "worker.py",
    }


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

    assert (
        "gpu-holder guard --gpus 0-7 --risk-util 0.6 "
        "--target-util 0.9 --mem 0.05 --backend torch"
    ) in content
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
        "gpu-holder events",
        "gpu-holder report",
    ]:
        assert deleted_feature not in content


def test_repository_has_github_ready_safety_docs() -> None:
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert (ROOT / "CONTRIBUTING.md").exists()
    assert (ROOT / "MANIFEST.in").exists()
    assert (ROOT / "ROADMAP.md").exists()
    assert (ROOT / "SECURITY.md").exists()
    assert (ROOT / ".github" / "workflows" / "ci.yml").exists()
    assert (ROOT / ".github" / "pull_request_template.md").exists()
    assert "External CUDA processes are read-only scheduling signals." in contributing
    assert "policy.py" in contributing
    assert "driver_backend.py" in contributing
    assert "telemetry.py" in contributing
    assert "ROADMAP.md" in readme
    assert "Near-Term Priorities" in roadmap
    assert "Non-Goals" in roadmap
    assert "Compatibility Reality" in roadmap
    assert "It is not a promise that every machine can run it." in roadmap
    assert "embedded PTX smoke" in roadmap
    assert "ctypes + libcuda.so.1 + embedded conservative PTX" in roadmap
    assert "Managing, killing, suspending, renicing" in roadmap
    assert "include ROADMAP.md" in manifest
    assert "recursive-include docs *.md" in manifest


def test_github_templates_are_english_and_safety_focused() -> None:
    bug = (ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").read_text(encoding="utf-8")
    feature = (ROOT / ".github" / "ISSUE_TEMPLATE" / "feature_request.yml").read_text(encoding="utf-8")
    support = (ROOT / ".github" / "ISSUE_TEMPLATE" / "support.yml").read_text(encoding="utf-8")
    pr = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")

    assert "Report scheduling, CLI, daemon, or diagnostic behavior issues." in bug
    assert "Propose policy, worker, CLI, diagnostics" in feature
    assert "Prefer read-only commands for support requests" in support
    assert "External GPU process metadata remains a read-only scheduling signal." in pr


def test_backend_strategy_documents_compatibility_boundary() -> None:
    content = (ROOT / "docs" / "backend_strategy.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "ctypes + libcuda.so.1 + embedded conservative PTX" in content
    assert "gpu-holder doctor --backend torch" in content
    assert "gpu-holder doctor --backend driver" in content
    assert "gpu-holder guard --backend driver" in content
    assert "experimental but usable as a worker" in content
    assert "tiny embedded PTX kernel can be JIT-compiled" in content
    assert "Driver API worker intentionally uses one conservative spin kernel" in content
    assert "Works on Linux machines with an NVIDIA driver and accessible CUDA devices." in content
    assert 'It should not be documented as "works on every machine".' in content
    assert "CUDA_VISIBLE_DEVICES" in content
    assert "docs/backend_strategy.md" in readme
