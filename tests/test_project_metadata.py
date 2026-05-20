from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_release_checklist_documents_required_verification() -> None:
    content = (ROOT / "docs" / "release.md").read_text(encoding="utf-8")

    assert "python -m ruff check src tests" in content
    assert "python -m pytest -q" in content
    assert "Do not kill existing user GPU jobs" in content


def test_docs_use_plain_pytest_from_project_root() -> None:
    checked_paths = [
        ROOT / "README.md",
        ROOT / "docs" / "testing.md",
        ROOT / "docs" / "release.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in checked_paths)

    assert "python -m pytest -q" in combined
    assert "PYTHONPATH=src python -m pytest" not in combined


def test_changelog_has_unreleased_section_and_safety_note() -> None:
    content = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "## [Unreleased]" in content
    assert "External GPU processes are read-only scheduling signals" in content


def test_github_templates_preserve_safety_invariant() -> None:
    template_dir = ROOT / ".github" / "ISSUE_TEMPLATE"
    template_names = {
        path.name
        for path in template_dir.glob("*.yml")
    }

    assert {"bug_report.yml", "feature_request.yml", "support.yml"}.issubset(template_names)
    bug_report = (template_dir / "bug_report.yml").read_text(encoding="utf-8")
    feature_request = (template_dir / "feature_request.yml").read_text(encoding="utf-8")
    pr_template = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    combined = "\n".join([bug_report, feature_request, pr_template])
    assert "external GPU processes" in combined
    assert "read-only scheduling" in combined
    assert "kill" in combined.lower()


def test_project_metadata_is_publishable_without_template_placeholders() -> None:
    checked_paths = [
        ROOT / "pyproject.toml",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml",
        ROOT / "README.md",
        ROOT / "docs" / "release.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in checked_paths)

    assert "your-name" not in combined
    assert "example.com" not in combined
    assert "placeholder" not in combined.lower()
    assert "https://github.com/nene/gpu-holder" in combined


def test_monitoring_examples_document_metrics_alerts_and_dashboard() -> None:
    monitoring_dir = ROOT / "examples" / "monitoring"
    readme = (monitoring_dir / "README.md").read_text(encoding="utf-8")
    prometheus = (monitoring_dir / "prometheus.yml").read_text(encoding="utf-8")
    service = (monitoring_dir / "gpu-holder-metrics.service").read_text(encoding="utf-8")
    timer = (monitoring_dir / "gpu-holder-metrics.timer").read_text(encoding="utf-8")

    assert "gpu-holder metrics --output" in service
    assert "gpu-holder alerts > gpu-holder-alerts.yml" in readme
    assert "gpu-holder grafana-dashboard > gpu-holder-dashboard.json" in readme
    assert "gpu-holder-alerts.yml" in prometheus
    assert "localhost:9100" in prometheus
    assert "OnUnitActiveSec=15s" in timer


def test_pyproject_exposes_cli_and_build_metadata() -> None:
    content = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'gpu-holder = "gpu_holder.cli:main"' in content
    assert (ROOT / "src" / "gpu_holder" / "__main__.py").exists()
    assert '"build>=1.0.0"' in content
    assert '"Programming Language :: Python :: 3.12"' in content
    assert re.search(r"^license\s*=\s*\"MIT\"", content, re.MULTILINE)


def test_ci_runs_lint_tests_compile_and_package_build() -> None:
    content = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert 'python-version: ["3.10", "3.11", "3.12"]' in content
    assert "ruff check src tests" in content
    assert "pytest -q" in content
    assert "python -m compileall -q src tests" in content
    assert "python -m build" in content
