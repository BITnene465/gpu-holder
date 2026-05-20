from __future__ import annotations

import json
from pathlib import Path

from .alerts import render_prometheus_alert_rules
from .grafana import grafana_dashboard_payload
from .state import write_text_atomically


def write_monitoring_bundle(
    output_dir: Path,
    *,
    group_name: str = "gpu-holder",
    dashboard_title: str = "gpu-holder",
    dashboard_uid: str = "gpu-holder",
    datasource: str = "${DS_PROMETHEUS}",
) -> dict[str, object]:
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    alerts_path = write_text_atomically(
        output_dir / "gpu-holder-alerts.yml",
        render_prometheus_alert_rules(group_name=group_name),
    )
    dashboard_path = write_text_atomically(
        output_dir / "gpu-holder-dashboard.json",
        json.dumps(
            grafana_dashboard_payload(
                title=dashboard_title,
                uid=dashboard_uid,
                datasource=datasource,
            ),
            indent=2,
        )
        + "\n",
    )
    readme_path = write_text_atomically(
        output_dir / "README.md",
        _bundle_readme(
            group_name=group_name,
            dashboard_title=dashboard_title,
            dashboard_uid=dashboard_uid,
            datasource=datasource,
        ),
    )
    return {
        "output_dir": str(output_dir),
        "files": {
            "alerts": str(alerts_path),
            "dashboard": str(dashboard_path),
            "readme": str(readme_path),
        },
    }


def _bundle_readme(
    *,
    group_name: str,
    dashboard_title: str,
    dashboard_uid: str,
    datasource: str,
) -> str:
    return "\n".join(
        [
            "# gpu-holder monitoring bundle",
            "",
            "Generated files:",
            "",
            "- `gpu-holder-alerts.yml`: Prometheus alert rules",
            "- `gpu-holder-dashboard.json`: Grafana dashboard JSON",
            "",
            "Generation settings:",
            "",
            f"- alert group: `{group_name}`",
            f"- dashboard title: `{dashboard_title}`",
            f"- dashboard uid: `{dashboard_uid}`",
            f"- datasource: `{datasource}`",
            "",
            "Import `gpu-holder-alerts.yml` with Prometheus `rule_files`, then import",
            "`gpu-holder-dashboard.json` in Grafana.",
            "",
        ]
    )
