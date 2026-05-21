from __future__ import annotations

from gpu_holder.cli import (
    Config,
    GpuProcess,
    GpuSnapshot,
    Guard,
    ProcessKillTarget,
    build_parser,
    child_args,
    config_from_args,
    decide,
    format_gpus,
    make_hold,
    make_release,
    parse_gpus,
    parse_ratio,
    resolve_memory_ratio,
    process_signature,
    status_payload,
)


def test_ratio_options_accept_new_float_and_legacy_percent() -> None:
    assert parse_ratio("0.2") == 0.2
    assert parse_ratio("20%") == 0.2
    assert parse_ratio("75") == 0.75
    assert parse_ratio("75%") == 0.75


def test_child_args_emit_normalized_ratio_values() -> None:
    config = Config(
        gpus=parse_gpus("0,1,2,3,4,5,6,7"),
        target_util=parse_ratio("75%"),
        risk_util=parse_ratio("50%"),
        mem=parse_ratio("20%"),
        backend="torch",
    )
    args = child_args(config)

    assert args[args.index("--gpus") + 1] == "0-7"
    assert args[args.index("--target-util") + 1] == "0.75"
    assert args[args.index("--risk-util") + 1] == "0.5"
    assert args[args.index("--mem") + 1] == "0.2"
    assert args[args.index("--backend") + 1] == "torch"


def test_guard_parser_defaults_use_short_target_duty_cycle() -> None:
    args = build_parser().parse_args(["guard", "--target-util", "0.9"])
    config = config_from_args(args)

    assert config.target_util == 0.9
    assert config.backend == "torch"
    assert config.min_duty_cycle == 0.0
    assert config.max_duty_cycle == 1.0
    assert config.compute_burst_seconds == 0.2
    assert config.process_grace_window == 120.0


def test_guard_parser_exposes_worker_backend() -> None:
    args = build_parser().parse_args(["guard", "--backend", "torch"])
    config = config_from_args(args)

    assert config.backend == "torch"


def test_doctor_parser_accepts_driver_diagnostic_backend() -> None:
    args = build_parser().parse_args(["doctor", "--backend", "driver"])

    assert args.backend == "driver"


def test_status_payload_reports_selected_backend() -> None:
    config = Config(backend="torch")
    gpu = GpuSnapshot(
        index=0,
        uuid="gpu-0",
        name="GPU 0",
        utilization=90,
        memory_total=80 * 1024**3,
        memory_used=1 * 1024**3,
        memory_free=79 * 1024**3,
        temperature=None,
        processes=[],
    )
    decision = make_release(gpu, "risk_clear")

    payload = status_payload([gpu], [decision], {}, config)

    assert payload["config"]["backend"] == "torch"


def test_gpu_option_accepts_ranges_and_mixed_lists() -> None:
    assert parse_gpus("all") == "all"
    assert parse_gpus("0-7") == tuple(range(8))
    assert parse_gpus("0-3,6,7") == (0, 1, 2, 3, 6, 7)
    assert parse_gpus("0,1,1,2") == (0, 1, 2)
    assert format_gpus((0, 1, 2, 3, 6, 7)) == "0-3,6-7"


def test_memory_ratio_resolves_against_total_and_free_memory() -> None:
    total = 80 * 1024**3
    free = 10 * 1024**3
    reserve = 2 * 1024**3

    assert resolve_memory_ratio(0.2, total=total, free=free, reserve=reserve) == 8 * 1024**3


def test_single_low_gpu_holds_even_when_other_gpu_is_busy() -> None:
    config = Config(risk_util=0.5, target_util=0.75)
    busy_gpu = GpuSnapshot(
        index=0,
        uuid="gpu-0",
        name="GPU 0",
        utilization=95,
        memory_total=80 * 1024**3,
        memory_used=10 * 1024**3,
        memory_free=70 * 1024**3,
        temperature=None,
        processes=[],
    )
    risk_gpu = GpuSnapshot(
        index=1,
        uuid="gpu-1",
        name="GPU 1",
        utilization=40,
        memory_total=80 * 1024**3,
        memory_used=1 * 1024**3,
        memory_free=79 * 1024**3,
        temperature=None,
        processes=[],
    )

    decisions = decide([busy_gpu, risk_gpu], config)

    assert [decision.reason for decision in decisions] == ["risk_clear", "below_risk"]
    assert [decision.action for decision in decisions] == ["release", "hold"]


def test_new_holder_starts_only_below_per_gpu_risk_level() -> None:
    config = Config(risk_util=0.5, target_util=0.75)
    gpu = GpuSnapshot(
        index=0,
        uuid="gpu-0",
        name="GPU 0",
        utilization=60,
        memory_total=80 * 1024**3,
        memory_used=1 * 1024**3,
        memory_free=79 * 1024**3,
        temperature=None,
        processes=[],
    )

    decision = decide([gpu], config)[0]

    assert decision.action == "release"
    assert decision.reason == "risk_clear"

    gpu.utilization = 49
    decision = decide([gpu], config)[0]

    assert decision.action == "hold"
    assert decision.reason == "below_risk"


def test_running_holder_keeps_holding_without_external_process() -> None:
    config = Config(risk_util=0.5, target_util=0.75)
    gpu = GpuSnapshot(
        index=0,
        uuid="gpu-0",
        name="GPU 0",
        utilization=76,
        memory_total=80 * 1024**3,
        memory_used=10 * 1024**3,
        memory_free=70 * 1024**3,
        temperature=None,
        processes=[GpuProcess(pid=123, used_memory=4 * 1024**3, name="gpu-holder", is_holder=True)],
    )

    decision = decide([gpu], config)[0]

    assert decision.action == "hold"
    assert decision.reason == "holder_running"

    gpu.utilization = 99
    decision = decide([gpu], config)[0]

    assert decision.action == "hold"
    assert decision.reason == "holder_running"


def test_target_util_sets_holder_duty_cycle() -> None:
    config = Config(target_util=0.75, min_duty_cycle=0.0, max_duty_cycle=1.0)
    gpu = GpuSnapshot(
        index=0,
        uuid="gpu-0",
        name="GPU 0",
        utilization=40,
        memory_total=80 * 1024**3,
        memory_used=1 * 1024**3,
        memory_free=79 * 1024**3,
        temperature=None,
        processes=[],
    )

    decision = make_hold(gpu, config, reason="below_risk", assist=False)

    assert decision.duty_cycle == 0.75


def test_active_training_process_takes_priority_over_hold() -> None:
    config = Config(risk_util=0.5, target_util=0.75)
    gpu = GpuSnapshot(
        index=0,
        uuid="gpu-0",
        name="GPU 0",
        utilization=80,
        memory_total=80 * 1024**3,
        memory_used=30 * 1024**3,
        memory_free=50 * 1024**3,
        temperature=None,
        processes=[GpuProcess(pid=456, used_memory=30 * 1024**3, name="python", is_holder=False)],
    )

    decision = decide([gpu], config)[0]

    assert decision.action == "release"
    assert decision.reason == "busy_process_grace"


def test_active_training_process_releases_existing_holder() -> None:
    config = Config(risk_util=0.5, target_util=0.75)
    gpu = GpuSnapshot(
        index=0,
        uuid="gpu-0",
        name="GPU 0",
        utilization=80,
        memory_total=80 * 1024**3,
        memory_used=31 * 1024**3,
        memory_free=49 * 1024**3,
        temperature=None,
        processes=[
            GpuProcess(pid=456, used_memory=30 * 1024**3, name="python", is_holder=False),
            GpuProcess(pid=789, used_memory=1024**3, name="gpu-holder", is_holder=True),
        ],
    )

    decision = decide([gpu], config)[0]

    assert decision.action == "release"
    assert decision.reason == "busy_process_grace"


def test_busy_process_releases_existing_holder_during_grace_window() -> None:
    config = Config(risk_util=0.5, target_util=0.75, assist_mem="512MiB")
    external = GpuProcess(pid=456, used_memory=30 * 1024**3, name="[Not Found]", is_holder=False)
    gpu = GpuSnapshot(
        index=0,
        uuid="gpu-0",
        name="GPU 0",
        utilization=80,
        memory_total=80 * 1024**3,
        memory_used=31 * 1024**3,
        memory_free=49 * 1024**3,
        temperature=None,
        processes=[
            external,
            GpuProcess(pid=789, used_memory=1024**3, name="gpu-holder", is_holder=True),
        ],
    )

    decision = decide([gpu], config)[0]

    assert decision.action == "release"
    assert decision.reason == "busy_process_grace"

    decision = decide(
        [gpu],
        config,
        external_process_first_seen={0: {process_signature([external]): 0.0}},
        now=121.0,
    )[0]

    assert decision.action == "release"
    assert decision.reason == "busy_process"


def test_busy_process_gets_assist_after_grace_if_under_risk() -> None:
    config = Config(risk_util=0.5, target_util=0.75, assist_mem="512MiB")
    external = GpuProcess(pid=456, used_memory=30 * 1024**3, name="[Not Found]", is_holder=False)
    gpu = GpuSnapshot(
        index=0,
        uuid="gpu-0",
        name="GPU 0",
        utilization=10,
        memory_total=80 * 1024**3,
        memory_used=31 * 1024**3,
        memory_free=49 * 1024**3,
        temperature=None,
        processes=[external],
    )

    decision = decide(
        [gpu],
        config,
        external_process_first_seen={0: {process_signature([external]): 0.0}},
        now=121.0,
    )[0]

    assert decision.action == "hold"
    assert decision.reason == "busy_process_idle"


def test_idle_busy_process_gets_assist_hold() -> None:
    config = Config(risk_util=0.5, target_util=0.75, assist_mem="512MiB", process_grace_window=0.0)
    gpu = GpuSnapshot(
        index=5,
        uuid="gpu-5",
        name="GPU 5",
        utilization=0,
        memory_total=80 * 1024**3,
        memory_used=20 * 1024**3,
        memory_free=60 * 1024**3,
        temperature=None,
        processes=[GpuProcess(pid=456, used_memory=20 * 1024**3, name="[Not Found]", is_holder=False)],
    )

    external = [process for process in gpu.processes if not process.is_holder]
    decision = decide(
        [gpu],
        config,
        external_process_first_seen={5: {process_signature(external): 0.0}},
        now=1.0,
    )[0]

    assert decision.action == "hold"
    assert decision.reason == "busy_process_idle"
    assert decision.hold_mode == "assist"
    assert decision.memory_bytes == 512 * 1024**2


def test_busy_process_assist_releases_after_util_rises() -> None:
    config = Config(risk_util=0.5, target_util=0.75, assist_mem="512MiB", process_grace_window=0.0)
    gpu = GpuSnapshot(
        index=5,
        uuid="gpu-5",
        name="GPU 5",
        utilization=10,
        memory_total=80 * 1024**3,
        memory_used=21 * 1024**3,
        memory_free=59 * 1024**3,
        temperature=None,
        processes=[
            GpuProcess(pid=456, used_memory=20 * 1024**3, name="[Not Found]", is_holder=False),
            GpuProcess(pid=789, used_memory=512 * 1024**2, name="gpu-holder", is_holder=True),
        ],
    )

    external = [process for process in gpu.processes if not process.is_holder]
    decision = decide(
        [gpu],
        config,
        external_process_first_seen={5: {process_signature(external): 0.0}},
        now=1.0,
    )[0]

    assert decision.action == "hold"
    assert decision.reason == "busy_process_idle"
    assert decision.hold_mode == "assist"

    gpu.utilization = 79
    decision = decide(
        [gpu],
        config,
        external_process_first_seen={5: {process_signature(external): 0.0}},
        now=1.0,
    )[0]

    assert decision.action == "release"
    assert decision.reason == "busy_process"


def test_training_process_releases_only_the_affected_gpu() -> None:
    config = Config(risk_util=0.5, target_util=0.75)
    training_gpu = GpuSnapshot(
        index=0,
        uuid="gpu-0",
        name="GPU 0",
        utilization=95,
        memory_total=80 * 1024**3,
        memory_used=30 * 1024**3,
        memory_free=50 * 1024**3,
        temperature=None,
        processes=[GpuProcess(pid=456, used_memory=30 * 1024**3, name="python", is_holder=False)],
    )
    idle_gpu = GpuSnapshot(
        index=1,
        uuid="gpu-1",
        name="GPU 1",
        utilization=10,
        memory_total=80 * 1024**3,
        memory_used=1 * 1024**3,
        memory_free=79 * 1024**3,
        temperature=None,
        processes=[],
    )

    decisions = decide([training_gpu, idle_gpu], config)

    assert [(decision.gpu_index, decision.action, decision.reason) for decision in decisions] == [
        (0, "release", "busy_process_grace"),
        (1, "hold", "below_risk"),
    ]


def test_training_end_reclaims_gpu_immediately_below_risk_level() -> None:
    config = Config(risk_util=0.5, target_util=0.75)
    gpu = GpuSnapshot(
        index=0,
        uuid="gpu-0",
        name="GPU 0",
        utilization=10,
        memory_total=80 * 1024**3,
        memory_used=1 * 1024**3,
        memory_free=79 * 1024**3,
        temperature=None,
        processes=[],
    )

    decision = decide([gpu], config)[0]

    assert decision.action == "hold"
    assert decision.reason == "below_risk"


def test_guard_apply_stops_for_training_and_restarts_after_training(monkeypatch, tmp_path) -> None:
    events: list[tuple[str, int]] = []

    class FakeWorker:
        pid = 123

        def __init__(self, gpu_index: int, **kwargs: object) -> None:
            del kwargs
            self.gpu_index = gpu_index
            self.alive = False

        def start(self, timeout: float = 10) -> None:
            del timeout
            self.alive = True
            events.append(("start", self.gpu_index))

        def stop(self, timeout: float = 0.2) -> None:
            del timeout
            self.alive = False
            events.append(("stop", self.gpu_index))

        def is_alive(self) -> bool:
            return self.alive

    monkeypatch.setattr("gpu_holder.cli.WorkerProcess", FakeWorker)
    config = Config(state_dir=tmp_path)
    guard = Guard(config)
    gpu = GpuSnapshot(
        index=0,
        uuid="gpu-0",
        name="GPU 0",
        utilization=76,
        memory_total=80 * 1024**3,
        memory_used=1 * 1024**3,
        memory_free=79 * 1024**3,
        temperature=None,
        processes=[],
    )

    guard.apply([make_hold(gpu, config, reason="target_margin", assist=False)])
    guard.apply([make_release(gpu, "busy_process")])
    guard.apply([make_hold(gpu, config, reason="target_margin", assist=False)])

    assert events == [("start", 0), ("stop", 0), ("start", 0)]


def test_guard_dry_run_prints_decisions_without_state_or_workers(monkeypatch, tmp_path, capsys) -> None:
    def fake_read_snapshots(config: Config, workers: dict[int, object]) -> list[GpuSnapshot]:
        assert workers == {}
        return [
            GpuSnapshot(
                index=0,
                uuid="gpu-0",
                name="GPU 0",
                utilization=10,
                memory_total=80 * 1024**3,
                memory_used=1 * 1024**3,
                memory_free=79 * 1024**3,
                temperature=None,
                processes=[],
            )
        ]

    def fail_start_worker(decision: object) -> None:
        raise AssertionError(f"dry-run must not start worker: {decision}")

    config = Config(state_dir=tmp_path, dry_run=True, risk_util=0.5)
    guard = Guard(config)
    monkeypatch.setattr("gpu_holder.cli.read_snapshots", fake_read_snapshots)
    monkeypatch.setattr(guard, "start_worker", fail_start_worker)

    assert guard.run() == 0

    output = capsys.readouterr().out
    assert "gpu=0 util=10%" in output
    assert "action=hold reason=below_risk" in output
    assert not config.pid_file.exists()
    assert not config.status_file.exists()


def test_guard_once_runs_one_iteration_and_cleans_pidfile(monkeypatch, tmp_path) -> None:
    calls = 0

    def fake_read_snapshots(config: Config, workers: dict[int, object]) -> list[GpuSnapshot]:
        nonlocal calls
        del config, workers
        calls += 1
        return [
            GpuSnapshot(
                index=0,
                uuid="gpu-0",
                name="GPU 0",
                utilization=90,
                memory_total=80 * 1024**3,
                memory_used=1 * 1024**3,
                memory_free=79 * 1024**3,
                temperature=None,
                processes=[],
            )
        ]

    config = Config(state_dir=tmp_path, once=True)
    guard = Guard(config)
    monkeypatch.setattr("gpu_holder.cli.read_snapshots", fake_read_snapshots)

    assert guard.run() == 0

    assert calls == 1
    assert not config.pid_file.exists()
    assert config.status_file.exists()


def test_background_stop_targets_process_group_for_session_leader(monkeypatch) -> None:
    calls: list[tuple[str, int, int]] = []

    monkeypatch.setattr("gpu_holder.cli.os.getsid", lambda pid: pid)
    monkeypatch.setattr(
        "gpu_holder.cli.os.killpg",
        lambda pid, signum: calls.append(("killpg", pid, signum)),
    )
    monkeypatch.setattr(
        "gpu_holder.cli.os.kill",
        lambda pid, signum: calls.append(("kill", pid, signum)),
    )

    target = ProcessKillTarget(123)
    target.terminate()
    target.kill()

    assert calls == [("killpg", 123, 15), ("killpg", 123, 9)]


def test_foreground_stop_targets_only_process_when_not_session_leader(monkeypatch) -> None:
    calls: list[tuple[str, int, int]] = []

    monkeypatch.setattr("gpu_holder.cli.os.getsid", lambda pid: 999)
    monkeypatch.setattr(
        "gpu_holder.cli.os.killpg",
        lambda pid, signum: calls.append(("killpg", pid, signum)),
    )
    monkeypatch.setattr(
        "gpu_holder.cli.os.kill",
        lambda pid, signum: calls.append(("kill", pid, signum)),
    )

    target = ProcessKillTarget(123)
    target.terminate()
    target.kill()

    assert calls == [("kill", 123, 15), ("kill", 123, 9)]
