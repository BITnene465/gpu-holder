from __future__ import annotations

from gpu_holder.grafana import grafana_dashboard_payload


def test_grafana_dashboard_payload_contains_core_panels_and_metrics() -> None:
    payload = grafana_dashboard_payload(
        title="Shared GPUs",
        uid="shared-gpus",
        datasource="Prometheus",
    )
    panels = {panel["title"]: panel for panel in payload["panels"]}

    assert payload["title"] == "Shared GPUs"
    assert payload["uid"] == "shared-gpus"
    assert payload["refresh"] == "30s"
    assert "Machine utilization" in panels
    assert "Quota forecast" in panels
    assert "Per-GPU utilization" in panels
    assert panels["Quota forecast"]["targets"][0]["expr"] == (
        "gpu_holder_target_required_remaining_average_percent"
    )
    assert panels["Policy window coverage"]["targets"][0]["expr"] == (
        "gpu_holder_machine_policy_window_coverage_ratio * 100"
    )
    assert payload["templating"]["list"][0]["query"] == "prometheus"


def test_grafana_dashboard_payload_uses_datasource_on_panels_and_targets() -> None:
    payload = grafana_dashboard_payload(datasource="Prometheus")

    first_panel = payload["panels"][0]

    assert first_panel["datasource"] == "Prometheus"
    assert first_panel["targets"][0]["datasource"] == "Prometheus"
