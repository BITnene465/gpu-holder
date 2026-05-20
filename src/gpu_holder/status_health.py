from __future__ import annotations

import time


def status_age_seconds(payload: dict[str, object], *, now: float | None = None) -> float | None:
    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        return None
    current_time = time.time() if now is None else float(now)
    return max(0.0, current_time - float(timestamp))


def status_is_stale(
    payload: dict[str, object],
    *,
    stale_after: float | None = 15.0,
    now: float | None = None,
) -> bool:
    if stale_after is None or stale_after <= 0:
        return False
    age = status_age_seconds(payload, now=now)
    return age is not None and age > float(stale_after)


def enrich_status_health(
    payload: dict[str, object],
    *,
    stale_after: float | None = 15.0,
    now: float | None = None,
) -> dict[str, object]:
    enriched = dict(payload)
    age = status_age_seconds(payload, now=now)
    enriched["status_age_seconds"] = age
    enriched["status_stale"] = status_is_stale(payload, stale_after=stale_after, now=now)
    enriched["status_stale_after_seconds"] = stale_after
    enriched["utilization_target_health"] = utilization_target_health(payload)
    enriched["policy_window_health"] = policy_window_health(payload)
    enriched["utilization_target_forecast"] = utilization_target_forecast(payload)
    return enriched


def status_check(
    payload: dict[str, object],
    *,
    stale_after: float | None = 15.0,
    now: float | None = None,
    require_target: bool = False,
    require_forecast: bool = False,
) -> dict[str, object]:
    errors: list[str] = []
    if status_age_seconds(payload, now=now) is None:
        errors.append("status file has no timestamp")
    elif status_is_stale(payload, stale_after=stale_after, now=now):
        errors.append("status file is stale")
    if require_target:
        target_health = utilization_target_health(payload)
        if target_health["status"] == "unknown":
            errors.append("utilization target health is unknown")
        elif not target_health["ok"]:
            policy_average = target_health["policy_average_utilization"]
            target = target_health["target_util"]
            errors.append(
                f"policy average utilization {policy_average:.1f}% is below target {target:.1f}%"
            )
    if require_forecast:
        forecast = utilization_target_forecast(payload)
        status = forecast["status"]
        if status == "unknown":
            errors.append("utilization target forecast is unknown")
        elif status in {"at_risk", "complete_miss"}:
            required = forecast["required_remaining_average"]
            target = forecast["target_util"]
            if isinstance(required, (int, float)):
                errors.append(
                    "utilization target forecast "
                    f"{status}; remaining window requires {float(required):.1f}% "
                    f"average utilization to reach target {float(target):.1f}%"
                )
            else:
                policy_average = forecast["policy_average_utilization"]
                errors.append(
                    "utilization target forecast "
                    f"{status}; policy average utilization {float(policy_average):.1f}% "
                    f"is below target {float(target):.1f}%"
                )
    return {
        "ok": not errors,
        "errors": errors,
    }


def utilization_target_health(payload: dict[str, object]) -> dict[str, object]:
    target = _target_util(payload)
    policy_average = _policy_average_utilization(payload)
    if target is None or policy_average is None:
        return {
            "ok": False,
            "status": "unknown",
            "target_util": target,
            "policy_average_utilization": policy_average,
            "gap": None,
        }
    gap = round(float(policy_average) - float(target), 3)
    return {
        "ok": gap >= 0,
        "status": "above_target" if gap >= 0 else "below_target",
        "target_util": float(target),
        "policy_average_utilization": float(policy_average),
        "gap": gap,
    }


def policy_window_health(payload: dict[str, object]) -> dict[str, object]:
    machine = payload.get("machine")
    if not isinstance(machine, dict):
        return _unknown_policy_window_health()
    window_seconds = _float_or_none(machine.get("policy_window_seconds"))
    sample_count = _int_or_none(machine.get("policy_window_sample_count"))
    coverage_seconds = _float_or_none(machine.get("policy_window_coverage_seconds"))
    coverage_ratio = _float_or_none(machine.get("policy_window_coverage_ratio"))
    if window_seconds is None or coverage_seconds is None or coverage_ratio is None:
        return _unknown_policy_window_health(
            window_seconds=window_seconds,
            sample_count=sample_count,
            coverage_seconds=coverage_seconds,
            coverage_ratio=coverage_ratio,
        )
    ready = coverage_ratio >= 0.95
    return {
        "ok": ready,
        "status": "ready" if ready else "warming_up",
        "window_seconds": window_seconds,
        "sample_count": sample_count,
        "coverage_seconds": coverage_seconds,
        "coverage_ratio": round(coverage_ratio, 6),
    }


def utilization_target_forecast(payload: dict[str, object]) -> dict[str, object]:
    target_health = utilization_target_health(payload)
    window_health = policy_window_health(payload)
    target = _float_or_none(target_health.get("target_util"))
    policy_average = _float_or_none(target_health.get("policy_average_utilization"))
    window_seconds = _float_or_none(window_health.get("window_seconds"))
    coverage_seconds = _float_or_none(window_health.get("coverage_seconds"))
    coverage_ratio = _float_or_none(window_health.get("coverage_ratio"))
    if (
        target is None
        or policy_average is None
        or window_seconds is None
        or coverage_seconds is None
        or coverage_ratio is None
        or window_seconds <= 0
    ):
        return _unknown_utilization_target_forecast(
            target_util=target,
            policy_average_utilization=policy_average,
            window_seconds=window_seconds,
            coverage_seconds=coverage_seconds,
            coverage_ratio=coverage_ratio,
        )

    bounded_coverage = min(window_seconds, max(0.0, coverage_seconds))
    remaining_seconds = max(0.0, window_seconds - bounded_coverage)
    observed_area = policy_average * bounded_coverage
    required_area = target * window_seconds
    remaining_required_area = required_area - observed_area
    if remaining_seconds <= 0:
        status = "complete_met" if policy_average >= target else "complete_miss"
        return {
            "ok": policy_average >= target,
            "status": status,
            "target_util": target,
            "policy_average_utilization": policy_average,
            "window_seconds": window_seconds,
            "coverage_seconds": bounded_coverage,
            "coverage_ratio": round(min(1.0, max(0.0, coverage_ratio)), 6),
            "remaining_seconds": 0.0,
            "required_remaining_average": None,
        }

    required_remaining_average = remaining_required_area / remaining_seconds
    if required_remaining_average <= 0:
        status = "target_locked"
    elif policy_average >= target and required_remaining_average <= target:
        status = "on_track"
    elif required_remaining_average <= 100:
        status = "recoverable"
    else:
        status = "at_risk"
    return {
        "ok": status != "at_risk",
        "status": status,
        "target_util": target,
        "policy_average_utilization": policy_average,
        "window_seconds": window_seconds,
        "coverage_seconds": bounded_coverage,
        "coverage_ratio": round(min(1.0, max(0.0, coverage_ratio)), 6),
        "remaining_seconds": remaining_seconds,
        "required_remaining_average": round(required_remaining_average, 6),
    }


def format_policy_window_health(raw_health: object) -> str:
    if not isinstance(raw_health, dict):
        return "window=unknown coverage=-"
    status = raw_health.get("status") or "unknown"
    ratio = raw_health.get("coverage_ratio")
    if isinstance(ratio, (int, float)):
        coverage_text = f"{float(ratio) * 100:.0f}%"
    else:
        coverage_text = "-"
    return f"window={status} coverage={coverage_text}"


def format_utilization_target_forecast(raw_forecast: object) -> str:
    if not isinstance(raw_forecast, dict):
        return "quota=unknown required=-"
    status = raw_forecast.get("status") or "unknown"
    required = raw_forecast.get("required_remaining_average")
    if isinstance(required, (int, float)):
        required_text = f"{float(required):.1f}%"
    else:
        required_text = "-"
    return f"quota={status} required={required_text}"


def format_status_age(age: object) -> str:
    if not isinstance(age, (int, float)):
        return "-"
    seconds = float(age)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _target_util(payload: dict[str, object]) -> float | None:
    config = payload.get("config")
    if not isinstance(config, dict):
        return None
    return _float_or_none(config.get("target_util"))


def _policy_average_utilization(payload: dict[str, object]) -> float | None:
    machine = payload.get("machine")
    if not isinstance(machine, dict):
        return None
    for key in (
        "policy_average_utilization",
        "window_average_utilization",
        "average_utilization",
    ):
        value = _float_or_none(machine.get(key))
        if value is not None:
            return value
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, int):
        return None
    return int(value)


def _unknown_policy_window_health(
    *,
    window_seconds: float | None = None,
    sample_count: int | None = None,
    coverage_seconds: float | None = None,
    coverage_ratio: float | None = None,
) -> dict[str, object]:
    return {
        "ok": False,
        "status": "unknown",
        "window_seconds": window_seconds,
        "sample_count": sample_count,
        "coverage_seconds": coverage_seconds,
        "coverage_ratio": coverage_ratio,
    }


def _unknown_utilization_target_forecast(
    *,
    target_util: float | None = None,
    policy_average_utilization: float | None = None,
    window_seconds: float | None = None,
    coverage_seconds: float | None = None,
    coverage_ratio: float | None = None,
) -> dict[str, object]:
    return {
        "ok": False,
        "status": "unknown",
        "target_util": target_util,
        "policy_average_utilization": policy_average_utilization,
        "window_seconds": window_seconds,
        "coverage_seconds": coverage_seconds,
        "coverage_ratio": coverage_ratio,
        "remaining_seconds": None,
        "required_remaining_average": None,
    }
