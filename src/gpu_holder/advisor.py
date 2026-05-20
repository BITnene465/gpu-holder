from __future__ import annotations

from pathlib import Path

from .config import CONFIG_PROFILES, GuardConfig, config_payload, profile_defaults
from .simulator import DEFAULT_TOTAL_MEMORY, simulate_scenarios
from .units import parse_memory_spec


PASS = "pass"
WARN = "warn"
FAIL = "fail"
GRADE_ORDER = {
    PASS: 0,
    WARN: 1,
    FAIL: 2,
}


def config_from_status_config(raw_config: object) -> GuardConfig:
    if not isinstance(raw_config, dict):
        raise ValueError("status config is unavailable")
    default = GuardConfig()
    return GuardConfig(
        profile=str(raw_config.get("profile", default.profile)),
        gpus=_parse_gpus(raw_config.get("gpus", default.gpus)),
        target_util=int(raw_config.get("target_util", default.target_util)),
        idle_util=int(raw_config.get("idle_util", default.idle_util)),
        idle_window=float(raw_config.get("idle_window", default.idle_window)),
        machine_window=float(raw_config.get("machine_window", default.machine_window)),
        mem=parse_memory_spec(str(raw_config.get("mem", default.mem.raw))),
        reserve=parse_memory_spec(str(raw_config.get("reserve", default.reserve.raw))),
        busy_process_mem_threshold=parse_memory_spec(
            str(raw_config.get("busy_process_mem_threshold", default.busy_process_mem_threshold.raw))
        ),
        assist_mem=parse_memory_spec(str(raw_config.get("assist_mem", default.assist_mem.raw))),
        max_gpu_temp=_optional_int(raw_config.get("max_gpu_temp", default.max_gpu_temp)),
        thermal_resume_temp=_optional_int(
            raw_config.get("thermal_resume_temp", default.thermal_resume_temp)
        ),
        sample_interval=float(raw_config.get("sample_interval", default.sample_interval)),
        program=str(raw_config.get("program", default.program)),
        hold_mode=str(raw_config.get("hold_mode", default.hold_mode)),
        compute_burst_seconds=float(
            raw_config.get("compute_burst_seconds", default.compute_burst_seconds)
        ),
        compute_burst_jitter=float(
            raw_config.get("compute_burst_jitter", default.compute_burst_jitter)
        ),
        max_duty_cycle=float(raw_config.get("max_duty_cycle", default.max_duty_cycle)),
        min_duty_cycle=float(raw_config.get("min_duty_cycle", default.min_duty_cycle)),
        current_gap_weight=float(raw_config.get("current_gap_weight", default.current_gap_weight)),
        history_gap_weight=float(raw_config.get("history_gap_weight", default.history_gap_weight)),
        machine_gap_weight=float(raw_config.get("machine_gap_weight", default.machine_gap_weight)),
        event_log_max_bytes=int(
            raw_config.get("event_log_max_bytes", default.event_log_max_bytes)
        ),
        event_log_backup_count=int(
            raw_config.get("event_log_backup_count", default.event_log_backup_count)
        ),
        worker_restart_backoff=float(
            raw_config.get("worker_restart_backoff", default.worker_restart_backoff)
        ),
        worker_start_timeout=float(
            raw_config.get("worker_start_timeout", default.worker_start_timeout)
        ),
        worker_update_duty_threshold=float(
            raw_config.get("worker_update_duty_threshold", default.worker_update_duty_threshold)
        ),
        process_grace_window=float(
            raw_config.get("process_grace_window", default.process_grace_window)
        ),
        protected_process_patterns=_parse_patterns(
            raw_config.get("protected_process_patterns", default.protected_process_patterns)
        ),
        max_held_gpus=_optional_int(raw_config.get("max_held_gpus", default.max_held_gpus)),
        state_dir=Path(str(raw_config.get("state_dir", default.state_dir))).expanduser(),
        pause_file=_optional_path(raw_config.get("pause_file", default.pause_file)),
        dry_run=bool(raw_config.get("dry_run", default.dry_run)),
        tui=bool(raw_config.get("tui", default.tui)),
    )


def tuning_advice(config: GuardConfig) -> dict[str, object]:
    simulation = simulate_scenarios(config)
    checks = [
        _target_util_floor_check(config),
        _idle_emergency_check(simulation),
        _busy_process_yield_check(simulation),
        _thermal_yield_check(config, simulation),
        _memory_hold_size_check(config),
        _program_diversity_check(config),
        _sample_cadence_check(config),
    ]
    fail_count = sum(1 for check in checks if check["status"] == FAIL)
    warn_count = sum(1 for check in checks if check["status"] == WARN)
    score = max(0, 100 - fail_count * 25 - warn_count * 8)
    grade = FAIL if fail_count else WARN if warn_count else PASS
    recommendations = [
        str(check["recommendation"])
        for check in checks
        if check["status"] in {WARN, FAIL} and check.get("recommendation")
    ]
    return {
        "schema_version": 1,
        "ok": fail_count == 0,
        "grade": grade,
        "score": score,
        "summary": {
            "checks": len(checks),
            "passed": sum(1 for check in checks if check["status"] == PASS),
            "warnings": warn_count,
            "failures": fail_count,
            "scenario_count": simulation["summary"]["scenario_count"],
        },
        "checks": checks,
        "recommendations": recommendations,
        "simulation": _compact_simulation(simulation),
        "config": config_payload(config),
    }


def profile_comparison() -> dict[str, object]:
    profiles = [_profile_score(profile_name) for profile_name in CONFIG_PROFILES]
    profiles.sort(
        key=lambda item: (
            GRADE_ORDER.get(str(item.get("grade")), 99),
            -int(item.get("score", 0)),
            str(item.get("profile")),
        )
    )
    recommended_profile = profiles[0]["profile"] if profiles else None
    return {
        "recommended_profile": recommended_profile,
        "profiles": profiles,
    }


def _target_util_floor_check(config: GuardConfig) -> dict[str, object]:
    if config.target_util < 70:
        return _check(
            "target_util_floor",
            WARN,
            f"target_util={config.target_util}% is below the common 70% reclaim threshold",
            recommendation="Set --target-util to 75 or use --profile quota for strict hourly quotas.",
            details={"target_util": config.target_util, "minimum_recommended": 70},
        )
    if config.target_util > 90:
        return _check(
            "target_util_floor",
            WARN,
            f"target_util={config.target_util}% may be too aggressive for shared machines",
            recommendation="Keep --target-util in the 70-85 range unless the machine is dedicated.",
            details={"target_util": config.target_util, "maximum_recommended": 90},
        )
    return _check(
        "target_util_floor",
        PASS,
        f"target_util={config.target_util}% is suitable for quota protection",
        details={"target_util": config.target_util},
    )


def _idle_emergency_check(simulation: dict[str, object]) -> dict[str, object]:
    reasons = _scenario_reasons(simulation, "idle")
    if "low_util_emergency" in reasons:
        return _check(
            "idle_emergency_intervention",
            PASS,
            "idle scenario triggers emergency intervention after the low-util window",
            details={"reasons": sorted(reasons)},
        )
    return _check(
        "idle_emergency_intervention",
        FAIL,
        "idle scenario never reaches low-util emergency intervention",
        recommendation="Lower --idle-window or raise --idle-util so idle GPUs are held promptly.",
        details={"reasons": sorted(reasons)},
    )


def _busy_process_yield_check(simulation: dict[str, object]) -> dict[str, object]:
    actions = _scenario_action_counts(simulation, "busy")
    intrusive = int(actions.get("hold", 0)) + int(actions.get("assist", 0))
    if intrusive == 0 and int(actions.get("standby", 0)) > 0:
        return _check(
            "busy_process_yield",
            PASS,
            "busy-process scenario yields instead of starting holder workers",
            details={"action_counts": actions},
        )
    return _check(
        "busy_process_yield",
        FAIL,
        "busy-process scenario would start holder workers",
        recommendation=(
            "Increase --busy-process-mem-threshold only carefully; verify real jobs are above it."
        ),
        details={"action_counts": actions},
    )


def _thermal_yield_check(config: GuardConfig, simulation: dict[str, object]) -> dict[str, object]:
    if config.max_gpu_temp is None:
        return _check(
            "thermal_yield",
            WARN,
            "thermal guard is disabled",
            recommendation="Set --max-gpu-temp and --thermal-resume-temp for long-running daemons.",
            details={"max_gpu_temp": None},
        )
    decisions = _scenario_decisions(simulation, "thermal")
    thermal_decisions = [
        decision for decision in decisions if decision.get("reason") == "thermal_limit"
    ]
    intrusive = [
        decision
        for decision in thermal_decisions
        if decision.get("action") in {"hold", "assist"}
    ]
    if thermal_decisions and not intrusive:
        return _check(
            "thermal_yield",
            PASS,
            "thermal scenario yields on hot GPUs",
            details={
                "max_gpu_temp": config.max_gpu_temp,
                "thermal_resume_temp": config.thermal_resume_temp,
                "thermal_decisions": len(thermal_decisions),
            },
        )
    return _check(
        "thermal_yield",
        FAIL,
        "thermal scenario does not reliably yield",
        recommendation="Keep --max-gpu-temp below the hardware throttle range.",
        details={"thermal_decisions": len(thermal_decisions)},
    )


def _memory_hold_size_check(config: GuardConfig) -> dict[str, object]:
    estimated_percent = _estimated_memory_hold_percent(config)
    if estimated_percent > 80:
        return _check(
            "memory_hold_size",
            FAIL,
            f"mem={config.mem.raw} can reserve most of a GPU",
            recommendation="Use the default --mem 20% or a compute-only profile on shared machines.",
            details={"mem": config.mem.raw, "estimated_percent": estimated_percent},
        )
    if estimated_percent > 50:
        return _check(
            "memory_hold_size",
            WARN,
            f"mem={config.mem.raw} is high for shared machines",
            recommendation="Prefer --mem 20-30% unless you explicitly need memory residency.",
            details={"mem": config.mem.raw, "estimated_percent": estimated_percent},
        )
    return _check(
        "memory_hold_size",
        PASS,
        f"mem={config.mem.raw} is conservative enough for shared machines",
        details={"mem": config.mem.raw, "estimated_percent": estimated_percent},
    )


def _program_diversity_check(config: GuardConfig) -> dict[str, object]:
    programs = [part.strip() for part in config.program.split(",") if part.strip()]
    diverse_programs = {"mixed", "random"}
    if config.program in diverse_programs or len(set(programs)) > 1:
        return _check(
            "program_diversity",
            PASS,
            f"program={config.program} uses varied compute patterns",
            details={"program": config.program},
        )
    return _check(
        "program_diversity",
        WARN,
        f"program={config.program} uses a single compute pattern",
        recommendation="Use --program mixed, --program random, or a comma rotation such as matmul,conv,fft.",
        details={"program": config.program},
    )


def _sample_cadence_check(config: GuardConfig) -> dict[str, object]:
    recommended_max = max(1.0, min(10.0, config.idle_window / 4.0))
    if config.sample_interval <= recommended_max:
        return _check(
            "sample_cadence",
            PASS,
            f"sample_interval={config.sample_interval:g}s is responsive enough",
            details={
                "sample_interval": config.sample_interval,
                "recommended_max": recommended_max,
            },
        )
    return _check(
        "sample_cadence",
        WARN,
        f"sample_interval={config.sample_interval:g}s may react slowly to new workloads",
        recommendation="Keep --sample-interval at 2-5s for shared interactive machines.",
        details={
            "sample_interval": config.sample_interval,
            "recommended_max": recommended_max,
        },
    )


def _check(
    name: str,
    status: str,
    message: str,
    *,
    recommendation: str | None = None,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": name,
        "status": status,
        "message": message,
    }
    if recommendation:
        payload["recommendation"] = recommendation
    if details:
        payload["details"] = details
    return payload


def _profile_score(profile_name: str) -> dict[str, object]:
    config = config_from_status_config(
        {
            "profile": profile_name,
            **profile_defaults(profile_name),
        }
    )
    advice = tuning_advice(config)
    summary = advice.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    config_data = advice.get("config", {})
    if not isinstance(config_data, dict):
        config_data = {}
    return {
        "profile": profile_name,
        "description": CONFIG_PROFILES[profile_name].description,
        "grade": advice.get("grade"),
        "score": advice.get("score"),
        "ok": advice.get("ok"),
        "warnings": summary.get("warnings"),
        "failures": summary.get("failures"),
        "target_util": config_data.get("target_util"),
        "mem": config_data.get("mem"),
        "program": config_data.get("program"),
        "hold_mode": config_data.get("hold_mode"),
        "max_held_gpus": config_data.get("max_held_gpus"),
    }


def _parse_gpus(raw: object) -> tuple[int, ...] | str:
    if raw == "all":
        return "all"
    if isinstance(raw, list | tuple):
        return tuple(int(item) for item in raw)
    if isinstance(raw, str):
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    raise ValueError("status config gpus must be 'all', a list, or a comma-separated string")


def _parse_patterns(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list | tuple):
        return tuple(str(item) for item in raw)
    raise ValueError("status config protected_process_patterns must be a list or string")


def _optional_int(raw: object) -> int | None:
    if raw is None:
        return None
    return int(raw)


def _optional_path(raw: object) -> Path | None:
    if raw is None:
        return None
    return Path(str(raw)).expanduser()


def _estimated_memory_hold_percent(config: GuardConfig) -> float:
    if config.mem.percent is not None:
        return float(config.mem.percent)
    resolved = config.mem.resolve(DEFAULT_TOTAL_MEMORY)
    return resolved * 100.0 / DEFAULT_TOTAL_MEMORY


def _compact_simulation(simulation: dict[str, object]) -> dict[str, object]:
    scenarios = simulation.get("scenarios", [])
    compact_scenarios: list[dict[str, object]] = []
    if isinstance(scenarios, list):
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            compact_scenarios.append(
                {
                    "scenario": scenario.get("scenario"),
                    "summary": scenario.get("summary", {}),
                }
            )
    return {
        "summary": simulation.get("summary", {}),
        "scenarios": compact_scenarios,
    }


def _scenario_action_counts(simulation: dict[str, object], name: str) -> dict[str, int]:
    scenario = _scenario_payload(simulation, name)
    summary = scenario.get("summary", {})
    if not isinstance(summary, dict):
        return {}
    action_counts = summary.get("action_counts", {})
    if not isinstance(action_counts, dict):
        return {}
    return {str(action): int(count) for action, count in action_counts.items()}


def _scenario_reasons(simulation: dict[str, object], name: str) -> set[str]:
    return {
        str(decision["reason"])
        for decision in _scenario_decisions(simulation, name)
        if "reason" in decision
    }


def _scenario_decisions(simulation: dict[str, object], name: str) -> list[dict[str, object]]:
    scenario = _scenario_payload(simulation, name)
    timeline = scenario.get("timeline", [])
    if not isinstance(timeline, list):
        return []
    decisions: list[dict[str, object]] = []
    for item in timeline:
        if not isinstance(item, dict):
            continue
        item_decisions = item.get("decisions", [])
        if not isinstance(item_decisions, list):
            continue
        decisions.extend(
            decision for decision in item_decisions if isinstance(decision, dict)
        )
    return decisions


def _scenario_payload(simulation: dict[str, object], name: str) -> dict[str, object]:
    scenarios = simulation.get("scenarios", [])
    if not isinstance(scenarios, list):
        return {}
    for scenario in scenarios:
        if isinstance(scenario, dict) and scenario.get("scenario") == name:
            return scenario
    return {}
