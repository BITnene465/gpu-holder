from __future__ import annotations

from gpu_holder.status_health import (
    enrich_status_health,
    format_policy_window_health,
    policy_window_health,
    format_utilization_target_forecast,
    format_status_age,
    status_check,
    status_age_seconds,
    status_is_stale,
    utilization_target_forecast,
    utilization_target_health,
)


def test_status_age_seconds_uses_payload_timestamp() -> None:
    payload = {"timestamp": 100.0}

    assert status_age_seconds(payload, now=112.5) == 12.5


def test_status_age_seconds_handles_missing_timestamp() -> None:
    assert status_age_seconds({}, now=100.0) is None


def test_status_is_stale_respects_threshold_and_disable() -> None:
    payload = {"timestamp": 100.0}

    assert status_is_stale(payload, stale_after=15.0, now=116.0)
    assert not status_is_stale(payload, stale_after=20.0, now=116.0)
    assert not status_is_stale(payload, stale_after=0.0, now=1000.0)


def test_enrich_status_health_adds_derived_fields() -> None:
    enriched = enrich_status_health(
        {
            "timestamp": 100.0,
            "config": {"target_util": 75},
            "machine": {"policy_average_utilization": 80.0},
        },
        stale_after=15.0,
        now=120.0,
    )

    assert enriched["status_age_seconds"] == 20.0
    assert enriched["status_stale"] is True
    assert enriched["status_stale_after_seconds"] == 15.0
    assert enriched["utilization_target_health"]["ok"] is True
    assert enriched["utilization_target_health"]["status"] == "above_target"
    assert enriched["policy_window_health"]["status"] == "unknown"
    assert enriched["utilization_target_forecast"]["status"] == "unknown"


def test_status_check_reports_ok_and_stale_errors() -> None:
    ok = status_check({"timestamp": 100.0}, stale_after=15.0, now=110.0)
    stale = status_check({"timestamp": 100.0}, stale_after=15.0, now=120.0)
    missing_timestamp = status_check({}, stale_after=15.0, now=120.0)

    assert ok == {"ok": True, "errors": []}
    assert stale == {"ok": False, "errors": ["status file is stale"]}
    assert missing_timestamp == {"ok": False, "errors": ["status file has no timestamp"]}


def test_utilization_target_health_reports_above_and_below_target() -> None:
    above = utilization_target_health(
        {
            "config": {"target_util": 75},
            "machine": {"policy_average_utilization": 80.0},
        }
    )
    below = utilization_target_health(
        {
            "config": {"target_util": 75},
            "machine": {"window_average_utilization": 68.5},
        }
    )

    assert above == {
        "ok": True,
        "status": "above_target",
        "target_util": 75.0,
        "policy_average_utilization": 80.0,
        "gap": 5.0,
    }
    assert below == {
        "ok": False,
        "status": "below_target",
        "target_util": 75.0,
        "policy_average_utilization": 68.5,
        "gap": -6.5,
    }


def test_utilization_target_health_reports_unknown_when_status_lacks_target_data() -> None:
    health = utilization_target_health({"machine": {"average_utilization": 70.0}})

    assert health == {
        "ok": False,
        "status": "unknown",
        "target_util": None,
        "policy_average_utilization": 70.0,
        "gap": None,
    }


def test_policy_window_health_reports_ready_and_warming_up() -> None:
    ready = policy_window_health(
        {
            "machine": {
                "policy_window_seconds": 3600.0,
                "policy_window_sample_count": 120,
                "policy_window_coverage_seconds": 3500.0,
                "policy_window_coverage_ratio": 3500.0 / 3600.0,
            },
        }
    )
    warming_up = policy_window_health(
        {
            "machine": {
                "policy_window_seconds": 3600.0,
                "policy_window_sample_count": 3,
                "policy_window_coverage_seconds": 120.0,
                "policy_window_coverage_ratio": 120.0 / 3600.0,
            },
        }
    )

    assert ready == {
        "ok": True,
        "status": "ready",
        "window_seconds": 3600.0,
        "sample_count": 120,
        "coverage_seconds": 3500.0,
        "coverage_ratio": 0.972222,
    }
    assert warming_up["ok"] is False
    assert warming_up["status"] == "warming_up"
    assert warming_up["coverage_ratio"] == 0.033333


def test_format_policy_window_health_uses_readable_status_and_coverage() -> None:
    assert format_policy_window_health({}) == "window=unknown coverage=-"
    assert (
        format_policy_window_health({"status": "warming_up", "coverage_ratio": 0.25})
        == "window=warming_up coverage=25%"
    )


def test_utilization_target_forecast_reports_on_track_recoverable_and_at_risk() -> None:
    on_track = utilization_target_forecast(
        {
            "config": {"target_util": 75},
            "machine": {
                "policy_average_utilization": 80.0,
                "policy_window_seconds": 3600.0,
                "policy_window_sample_count": 10,
                "policy_window_coverage_seconds": 1800.0,
                "policy_window_coverage_ratio": 0.5,
            },
        }
    )
    recoverable = utilization_target_forecast(
        {
            "config": {"target_util": 75},
            "machine": {
                "policy_average_utilization": 50.0,
                "policy_window_seconds": 3600.0,
                "policy_window_sample_count": 10,
                "policy_window_coverage_seconds": 1800.0,
                "policy_window_coverage_ratio": 0.5,
            },
        }
    )
    at_risk = utilization_target_forecast(
        {
            "config": {"target_util": 75},
            "machine": {
                "policy_average_utilization": 40.0,
                "policy_window_seconds": 3600.0,
                "policy_window_sample_count": 10,
                "policy_window_coverage_seconds": 3000.0,
                "policy_window_coverage_ratio": 3000.0 / 3600.0,
            },
        }
    )

    assert on_track["status"] == "on_track"
    assert on_track["required_remaining_average"] == 70.0
    assert recoverable["status"] == "recoverable"
    assert recoverable["required_remaining_average"] == 100.0
    assert at_risk["ok"] is False
    assert at_risk["status"] == "at_risk"
    assert at_risk["required_remaining_average"] == 250.0


def test_utilization_target_forecast_reports_complete_windows() -> None:
    met = utilization_target_forecast(
        {
            "config": {"target_util": 75},
            "machine": {
                "policy_average_utilization": 75.0,
                "policy_window_seconds": 3600.0,
                "policy_window_coverage_seconds": 3600.0,
                "policy_window_coverage_ratio": 1.0,
            },
        }
    )
    missed = utilization_target_forecast(
        {
            "config": {"target_util": 75},
            "machine": {
                "policy_average_utilization": 74.0,
                "policy_window_seconds": 3600.0,
                "policy_window_coverage_seconds": 3600.0,
                "policy_window_coverage_ratio": 1.0,
            },
        }
    )

    assert met["status"] == "complete_met"
    assert met["required_remaining_average"] is None
    assert missed["ok"] is False
    assert missed["status"] == "complete_miss"


def test_format_utilization_target_forecast_uses_required_remaining_average() -> None:
    assert format_utilization_target_forecast({}) == "quota=unknown required=-"
    assert (
        format_utilization_target_forecast(
            {"status": "recoverable", "required_remaining_average": 87.25}
        )
        == "quota=recoverable required=87.2%"
    )


def test_status_check_can_require_utilization_target() -> None:
    payload = {
        "timestamp": 100.0,
        "config": {"target_util": 75},
        "machine": {"policy_average_utilization": 68.5},
    }

    check = status_check(payload, stale_after=15.0, now=110.0, require_target=True)

    assert check == {
        "ok": False,
        "errors": ["policy average utilization 68.5% is below target 75.0%"],
    }


def test_status_check_can_require_utilization_forecast() -> None:
    payload = {
        "timestamp": 100.0,
        "config": {"target_util": 75},
        "machine": {
            "policy_average_utilization": 40.0,
            "policy_window_seconds": 3600.0,
            "policy_window_coverage_seconds": 3000.0,
            "policy_window_coverage_ratio": 3000.0 / 3600.0,
        },
    }

    check = status_check(
        payload,
        stale_after=15.0,
        now=110.0,
        require_forecast=True,
    )

    assert check == {
        "ok": False,
        "errors": [
            "utilization target forecast at_risk; remaining window requires 250.0% "
            "average utilization to reach target 75.0%"
        ],
    }


def test_status_check_forecast_accepts_recoverable_windows() -> None:
    payload = {
        "timestamp": 100.0,
        "config": {"target_util": 75},
        "machine": {
            "policy_average_utilization": 50.0,
            "policy_window_seconds": 3600.0,
            "policy_window_coverage_seconds": 1800.0,
            "policy_window_coverage_ratio": 0.5,
        },
    }

    check = status_check(
        payload,
        stale_after=15.0,
        now=110.0,
        require_forecast=True,
    )

    assert check == {"ok": True, "errors": []}


def test_format_status_age_uses_readable_units() -> None:
    assert format_status_age(None) == "-"
    assert format_status_age(12.3) == "12s"
    assert format_status_age(90.0) == "1.5m"
    assert format_status_age(7200.0) == "2.0h"
