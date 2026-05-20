from __future__ import annotations

from gpu_holder.alerts import alert_rules_payload, render_prometheus_alert_rules


def test_alert_rules_payload_contains_core_quota_and_safety_alerts() -> None:
    payload = alert_rules_payload(group_name="custom-gpu-holder")
    group = payload["groups"][0]
    rules = {rule["alert"]: rule for rule in group["rules"]}

    assert group["name"] == "custom-gpu-holder"
    assert "GPUHolderStatusMissing" in rules
    assert "GPUHolderQuotaForecastAtRisk" in rules
    assert "GPUHolderWorkerStartBackoff" in rules
    assert "GPUHolderThermalLimit" in rules
    assert rules["GPUHolderQuotaForecastAtRisk"]["expr"] == (
        "gpu_holder_target_forecast_at_risk == 1"
    )
    assert rules["GPUHolderQuotaForecastAtRisk"]["labels"]["severity"] == "warning"


def test_render_prometheus_alert_rules_is_yaml_like_text() -> None:
    rendered = render_prometheus_alert_rules(group_name="custom")

    assert rendered.startswith("groups:\n")
    assert '- name: "custom"' in rendered
    assert "  - alert: GPUHolderStatusStale" in rendered
    assert '    expr: "gpu_holder_status_stale == 1"' in rendered
    assert "gpu_holder_target_required_remaining_average_percent" not in rendered
