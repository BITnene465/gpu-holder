## Summary

- 

## Verification

- [ ] `python -m ruff check --no-cache src tests`
- [ ] `python -m pytest -q`
- [ ] `PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache python -m compileall -q src tests`
- [ ] Behavior changes are documented in README, docs, or examples

## Safety

- [ ] This change does not kill, suspend, renice, or otherwise control external GPU processes.
- [ ] External GPU process metadata remains a read-only scheduling signal.
- [ ] Real CUDA smoke tests were only run on idle GPUs or GPUs I explicitly own.
