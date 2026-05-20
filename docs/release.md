# Release Checklist

Use this checklist before publishing a new `gpu-holder` release.

## Preflight

- Confirm the release scope is documented in `CHANGELOG.md`.
- Confirm `README.md`, `docs/`, and `examples/gpu-holder.toml` match the
  current CLI behavior.
- Confirm `examples/monitoring/` still matches metrics, alerts, and Grafana
  dashboard commands.
- Confirm no code path controls external GPU processes. External PIDs must
  remain read-only scheduling signals.
- Confirm default dependencies remain minimal and CUDA/PyTorch imports stay
  behind optional runtime paths.

## Local Verification

Run these from the project root:

```bash
python -m ruff check src tests
python -m pytest -q
PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache python -m compileall -q src tests
PYTHONPATH=src python -m gpu_holder --version
PYTHONPATH=src python -m gpu_holder init-config --stdout
PYTHONPATH=src python -m gpu_holder init-config --profile quota --stdout
PYTHONPATH=src python -m gpu_holder recipes
PYTHONPATH=src python -m gpu_holder recipes --name busy-shared
PYTHONPATH=src python -m gpu_holder plan --fake --json
PYTHONPATH=src python -m gpu_holder preflight --fake --no-diagnostics --json
PYTHONPATH=src python -m gpu_holder simulate --scenario all --json
PYTHONPATH=src python -m gpu_holder tune --strict --json
PYTHONPATH=src python -m gpu_holder tune --compare-profiles --json
PYTHONPATH=src python -m gpu_holder config --profile quota --explain --json
PYTHONPATH=src python -m gpu_holder config-reference --json
PYTHONPATH=src python -m gpu_holder profile-reference --json
PYTHONPATH=src python -m gpu_holder reason-reference --json
PYTHONPATH=src python -m gpu_holder completion bash
PYTHONPATH=src python -m gpu_holder manual --format man
PYTHONPATH=src python -m gpu_holder alerts --json
PYTHONPATH=src python -m gpu_holder grafana-dashboard
PYTHONPATH=src python -m gpu_holder monitoring-bundle --output-dir /tmp/gpu-holder-monitoring --json
PYTHONPATH=src python -m gpu_holder service --config /tmp/gpu-holder.toml
python -m build
```

Optional read-only runtime check:

```bash
PYTHONPATH=src python -m gpu_holder doctor --json
PYTHONPATH=src python -m gpu_holder status --check
PYTHONPATH=src python -m gpu_holder status --check --require-target
PYTHONPATH=src python -m gpu_holder status --check --require-forecast
PYTHONPATH=src python -m gpu_holder metrics
PYTHONPATH=src python -m gpu_holder metrics --output /tmp/gpu-holder.prom
PYTHONPATH=src python -m gpu_holder dashboard --once
PYTHONPATH=src python -m gpu_holder dashboard --once --history
PYTHONPATH=src python -m gpu_holder dashboard --once --explain
PYTHONPATH=src python -m gpu_holder dashboard --once --advice
PYTHONPATH=src python -m gpu_holder explain
```

`doctor` may return a non-zero exit code on machines without both monitoring
backends (`pynvml`/NVML and `nvidia-smi`) or without CUDA-capable PyTorch.
Missing NVML Python bindings alone are acceptable when `doctor` reports the
`nvidia-smi` fallback backend as healthy. `status --check` and
`dashboard --once` return non-zero when no status file exists.
`status --check --require-target` can also return non-zero when the recorded
utilization target is not met. `status --check --require-forecast` can return
non-zero when the remaining-window target forecast is unrecoverable. Those are
acceptable if the output is accurate.

## Manual GPU Smoke

Only run real holder workers on an idle GPU or a GPU you explicitly own.

Recommended smoke:

```bash
gpu-holder plan --fake
gpu-holder guard --gpus 0 --dry-run --tui
```

For real CUDA worker validation, use a short supervised tmux session on an idle
GPU and verify:

- status updates every sample interval
- `gpu-holder pause` releases owned workers
- `gpu-holder pause --for 1m` auto-expires without manual resume
- `gpu-holder history --since 5m` summarizes recent events without touching workers
- `gpu-holder explain` reports operator guidance without touching workers
- `gpu-holder disable-gpu 0` releases and skips that GPU
- `gpu-holder disable-gpu 0 --for 1m` auto-expires without manual enable
- `gpu-holder stop` refuses to stop non-holder PIDs

Do not kill existing user GPU jobs during release validation.

## Publish

- Update version in `pyproject.toml`.
- Move `CHANGELOG.md` entries from `Unreleased` to the version section.
- Confirm `pyproject.toml` URLs point at the published GitHub repository.
- Tag the release.
- Confirm CI is green for supported Python versions.
- Attach notes that mention safety behavior, supported Python versions, and
  optional CUDA dependencies.
