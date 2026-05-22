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
        "worker_controls.py",
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
    content = (
        (ROOT / "README.md").read_text(encoding="utf-8")
        + (ROOT / "README.en.md").read_text(encoding="utf-8")
    ).lower()

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
    readme_en = (ROOT / "README.en.md").read_text(encoding="utf-8")
    hero = (ROOT / "assets" / "hero.svg").read_text(encoding="utf-8")
    development_log = (ROOT / "docs" / "development_log.md").read_text(encoding="utf-8")

    assert (ROOT / "README.en.md").exists()
    assert not (ROOT / "assets" / "logo.png").exists()
    assert (ROOT / "assets" / "hero.svg").exists()
    assert (ROOT / "CONTRIBUTING.md").exists()
    assert (ROOT / "MANIFEST.in").exists()
    assert (ROOT / "ROADMAP.md").exists()
    assert (ROOT / "SECURITY.md").exists()
    assert (ROOT / "docs" / "development_log.md").exists()
    assert (ROOT / ".github" / "workflows" / "ci.yml").exists()
    assert (ROOT / ".github" / "pull_request_template.md").exists()
    assert '<img src="assets/hero.svg"' in readme
    assert '<img src="assets/hero.svg"' in readme_en
    assert 'width="190"' in hero
    assert "PyTorch fallback" in hero
    assert '<a href="README.en.md">English</a>' in readme
    assert '<a href="README.md">中文</a>' in readme_en
    assert "## English" not in readme
    assert "## 中文" not in readme
    assert "### Quick Start" in readme_en
    assert "### 快速开始" in readme
    assert "## 一分钟选择" in readme
    assert "## One-Minute Choice" in readme_en
    assert "--dry-run --explain" in readme
    assert "--dry-run --explain" in readme_en
    assert "外部 CUDA 进程只是只读调度信号。" in contributing
    assert "policy.py" in contributing
    assert "driver_backend.py" in contributing
    assert "telemetry.py" in contributing
    assert "worker_controls.py" in contributing
    assert "worker_controls.py # backend-neutral duty cycle、jitter、hold-mode 控制" in readme
    assert "worker_controls.py # backend-neutral duty cycle, jitter, and hold-mode controls" in readme_en
    assert "gpu-holder guard --gpus 0 --risk-util 0.6 --target-util 0.9 --mem 0 --backend driver --once" in contributing
    assert "include README.en.md" in manifest
    assert "recursive-include assets *.svg" in manifest
    assert "ROADMAP.md" in readme
    assert "docs/development_log.md" in readme
    assert "avoid PyTorch and use the experimental Driver API backend" in readme_en
    assert "近期优先级" in roadmap
    assert "非目标" in roadmap
    assert "兼容性现实" in roadmap
    assert "所有机器\n都能跑" in roadmap
    assert "内置 PTX smoke" in roadmap
    assert "ctypes + libcuda.so.1 + embedded conservative PTX" in roadmap
    assert "管理、杀死、挂起、renice" in roadmap
    assert "include ROADMAP.md" in manifest
    assert "recursive-include docs *.md" in manifest
    assert "Driver backend 兼容性边界" in development_log
    assert "在有 NVIDIA driver 且 CUDA 设备可访问的 Linux 机器上工作" in development_log
    assert "worker_controls.py" in development_log


def test_github_templates_are_chinese_and_safety_focused() -> None:
    bug = (ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").read_text(encoding="utf-8")
    feature = (ROOT / ".github" / "ISSUE_TEMPLATE" / "feature_request.yml").read_text(encoding="utf-8")
    support = (ROOT / ".github" / "ISSUE_TEMPLATE" / "support.yml").read_text(encoding="utf-8")
    pr = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")

    assert "报告调度、CLI、daemon 或诊断行为问题。" in bug
    assert "提议 policy、worker、CLI、诊断、打包或文档改进。" in feature
    assert "支持请求优先使用只读命令" in support
    assert "外部 GPU 进程 metadata 仍然只是只读调度信号。" in pr


def test_backend_strategy_documents_compatibility_boundary() -> None:
    content = (ROOT / "docs" / "backend_strategy.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "ctypes + libcuda.so.1 + embedded conservative PTX" in content
    assert "gpu-holder doctor --backend torch" in content
    assert "gpu-holder doctor --backend driver" in content
    assert "gpu-holder guard --backend driver" in content
    assert "实验性能力，但已经可作为 worker 使用" in content
    assert "很小的内置 PTX kernel 是否能 JIT" in content
    assert "Driver API worker 故意只使用一个保守的 spin\nkernel" in content
    assert "worker_controls.py" in content
    assert "backend-neutral duty-cycle、burst jitter 和 hold-mode 语义" in content
    assert "在有 NVIDIA driver 且 CUDA 设备可访问的 Linux 机器上工作。" in content
    assert "不应写成“所有机器都能跑”。" in content
    assert "CUDA_VISIBLE_DEVICES" in content
    assert "docs/backend_strategy.md" in readme
