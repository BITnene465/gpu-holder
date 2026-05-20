from __future__ import annotations

import json
from pathlib import Path

from gpu_holder.monitoring_bundle import write_monitoring_bundle


def test_write_monitoring_bundle_writes_alerts_dashboard_and_readme(tmp_path: Path) -> None:
    payload = write_monitoring_bundle(
        tmp_path / "monitoring",
        group_name="shared-gpus",
        dashboard_title="Shared GPUs",
        dashboard_uid="shared-gpus",
        datasource="Prometheus",
    )

    files = payload["files"]
    alerts = Path(files["alerts"]).read_text(encoding="utf-8")
    dashboard = json.loads(Path(files["dashboard"]).read_text(encoding="utf-8"))
    readme = Path(files["readme"]).read_text(encoding="utf-8")

    assert payload["output_dir"].endswith("monitoring")
    assert '- name: "shared-gpus"' in alerts
    assert "GPUHolderQuotaForecastAtRisk" in alerts
    assert dashboard["title"] == "Shared GPUs"
    assert dashboard["uid"] == "shared-gpus"
    assert "gpu_holder_target_forecast_at_risk" in json.dumps(dashboard)
    assert "alert group: `shared-gpus`" in readme
    assert "datasource: `Prometheus`" in readme
