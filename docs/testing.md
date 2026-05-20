# Testing

## No-GPU Tests

These are safe on busy machines:

```bash
python -m pytest -q
PYTHONPATH=src python -m gpu_holder --version
PYTHONPATH=src python -m gpu_holder init-config --stdout
PYTHONPATH=src python -m gpu_holder recipes
PYTHONPATH=src python -m gpu_holder recipes --name strict-quota
PYTHONPATH=src python -m gpu_holder plan --fake
PYTHONPATH=src python -m gpu_holder plan --fake --json
PYTHONPATH=src python -m gpu_holder preflight --fake --no-diagnostics
PYTHONPATH=src python -m gpu_holder preflight --fake --no-diagnostics --json
PYTHONPATH=src python -m gpu_holder simulate --json
PYTHONPATH=src python -m gpu_holder simulate --scenario all --json
PYTHONPATH=src python -m gpu_holder tune --json
PYTHONPATH=src python -m gpu_holder tune --strict --json
PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache python -m compileall -q src tests
```

`pytest` is configured to include `src/` on the test import path when run from
the project root. These checks do not import PyTorch, allocate CUDA memory, or
start holder workers.

## Offline Policy Simulation

`simulate` evaluates policy behavior over time. It is useful for tuning
`idle-window`, `target-util`, and busy-process thresholds before running a real
guard:

```bash
PYTHONPATH=src python -m gpu_holder simulate
PYTHONPATH=src python -m gpu_holder simulate --scenario thermal --json
PYTHONPATH=src python -m gpu_holder simulate --scenario all --json
PYTHONPATH=src python -m gpu_holder simulate --json
PYTHONPATH=src python -m gpu_holder simulate --trace trace.json --json
```

Invalid or missing trace files return exit code `2`; JSON mode emits an
`ok=false` error payload instead of a traceback. Schema errors include the
frame/GPU/process path, for example `frames[0].gpus[0].utilization is required`.
Built-in scenarios include `mixed`, `idle`, `busy`, `thermal`, and `quota-risk`.
Use `--scenario all` so release checks can exercise every built-in policy
branch without real GPU workers.

`tune` wraps the all-scenario simulation with config-level advice:

```bash
PYTHONPATH=src python -m gpu_holder tune
PYTHONPATH=src python -m gpu_holder tune --profile quota --json
PYTHONPATH=src python -m gpu_holder tune --strict --json
PYTHONPATH=src python -m gpu_holder tune --compare-profiles
PYTHONPATH=src python -m gpu_holder tune --target-util 60 --program matmul
```

It checks target utilization, low-util emergency intervention, busy-process
yielding, thermal yielding, memory hold size, compute program diversity, and
sample cadence without touching real GPUs. `--strict` exits non-zero for
warnings as well as failures, which is useful for release or CI gates.
`--compare-profiles` scores every built-in profile and recommends a starting
profile without touching real GPUs.

Trace files are JSON lists, or objects with a `frames` list:

```json
{
  "frames": [
    {
      "timestamp": 0,
      "gpus": [
        {
          "index": 0,
          "utilization": 10,
          "memory_total": "80GiB",
          "memory_used": "2GiB",
          "processes": [
            {"pid": 1234, "used_memory": "12GiB", "name": "train.py"}
          ]
        }
      ]
    }
  ]
}
```

## Read-Only GPU Tests

These require a monitoring backend, either NVML Python bindings or `nvidia-smi`,
but do not start workers:

```bash
gpu-holder doctor
gpu-holder preflight
gpu-holder dashboard --once
gpu-holder dashboard --once --events
gpu-holder dashboard --once --advice
gpu-holder dashboard --temporary-pause-duration 20m
gpu-holder dashboard --temporary-disable-duration 1h
gpu-holder plan
gpu-holder status --check
gpu-holder status --check --require-target
gpu-holder metrics
gpu-holder service --config /tmp/gpu-holder.toml
gpu-holder init-config --stdout
gpu-holder init-config --profile quota --stdout
gpu-holder dashboard
gpu-holder status --watch
gpu-holder events --follow
gpu-holder report
gpu-holder metrics --output /tmp/gpu-holder.prom
gpu-holder init-config --path /tmp/gpu-holder.toml
gpu-holder disable-gpu --state-dir /tmp/gpu-holder-state 0
gpu-holder disable-gpu --state-dir /tmp/gpu-holder-state --for 1m 0
gpu-holder enable-gpu --state-dir /tmp/gpu-holder-state 0
```

`doctor`, `preflight`, and `plan` are read-only. `doctor` reports the active
monitoring backend and treats NVML bindings as preferred but optional when
`nvidia-smi` fallback is available. `preflight` combines config
validation, diagnostics, GPU snapshot reading, and would-be policy decisions
without starting workers. `dashboard --once` and `dashboard` only read
`~/.gpu-holder/status.json` and optionally recent events/history/explanations. `status --watch` and
`status --check` only poll state files under `~/.gpu-holder`; `status --check`
returns non-zero when the status file is missing or stale. With
`--require-target`, it also returns non-zero when the recorded policy/window
average utilization is below `target_util`. With `--require-forecast`, it
returns non-zero only when the remaining-window forecast is unrecoverable.
`explain` reads status and recent events to generate guidance, but does not
change runtime state. `metrics` reads
status and emits Prometheus text without changing runtime state;
`metrics --output` writes only the requested collector textfile via atomic
replace. `events --follow` only polls event files. `service` only prints a unit file. `report` reads the same state files and can run
diagnostics, but it does not start or stop workers. `init-config` only
writes the requested TOML path and refuses to overwrite unless `--force` is
used; `init-config --stdout` only prints the selected template. `disable-gpu` and
`enable-gpu` only update the runtime state file. `disable-gpu --for` writes an
auto-expiring runtime state entry.

## Real CUDA Smoke Test

Run only on an explicitly safe GPU:

```bash
gpu-holder guard --gpus 7 --mem 1GiB --target-util 20 --sample-interval 2
```

Stop with `Ctrl-C`. Do not run this on a GPU that has a production or shared
training job unless you intentionally want the guard policy to interact with it.
