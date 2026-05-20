# Contributing

Thanks for improving `gpu-holder`.

## Development

```bash
python -m pip install -e ".[dev]"
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m gpu_holder.cli plan --fake
```

CUDA smoke tests must be run manually on an idle or explicitly selected GPU.
Never run real holder workers on a busy shared machine unless you own the target
GPU.

## Safety Invariant

External GPU processes are read-only scheduling signals. Contributions must not
add code that kills, suspends, renices, or otherwise controls non-holder PIDs.

Acceptable:

- stop a worker process created by `gpu-holder`
- read NVML process metadata
- write status files

Not acceptable:

- kill a training process
- free another process' memory
- shell out to destructive process-management commands

## Pull Request Checklist

- Add or update tests for policy changes.
- Keep default dependencies minimal.
- Put CUDA-specific imports behind runtime paths or optional extras.
- Update `README.md` and `docs/` when behavior changes.
- Update `CHANGELOG.md` for user-visible changes.
- Check `docs/release.md` before tagging a release.
- Confirm `pytest -q` and `ruff check src tests` pass.
