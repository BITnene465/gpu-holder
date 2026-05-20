## Summary

- 

## Verification

- [ ] `python -m ruff check src tests`
- [ ] `PYTHONPATH=src python -m pytest -q`
- [ ] `python -m compileall -q src tests`
- [ ] Updated docs/examples when behavior changed

## Safety

- [ ] This change does not kill, suspend, renice, or otherwise control external
      GPU processes.
- [ ] External GPU process metadata is used only as a read-only scheduling
      signal.
- [ ] Any real CUDA smoke test was run only on an idle GPU or a GPU explicitly
      owned by the tester.
