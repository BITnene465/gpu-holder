from __future__ import annotations


DEFAULT_DASHBOARD_UID = "gpu-holder"
DEFAULT_DASHBOARD_TITLE = "gpu-holder"
DEFAULT_DATASOURCE = "${DS_PROMETHEUS}"


def grafana_dashboard_payload(
    *,
    title: str = DEFAULT_DASHBOARD_TITLE,
    uid: str = DEFAULT_DASHBOARD_UID,
    datasource: str = DEFAULT_DATASOURCE,
) -> dict[str, object]:
    panels = [
        _stat_panel(
            1,
            "Status available",
            "gpu_holder_status_available",
            datasource=datasource,
            x=0,
            y=0,
            width=4,
            height=4,
        ),
        _stat_panel(
            2,
            "Quota forecast at risk",
            "gpu_holder_target_forecast_at_risk",
            datasource=datasource,
            x=4,
            y=0,
            width=4,
            height=4,
        ),
        _stat_panel(
            3,
            "Worker backoffs",
            "gpu_holder_worker_backoff_count",
            datasource=datasource,
            x=8,
            y=0,
            width=4,
            height=4,
        ),
        _stat_panel(
            4,
            "Thermal limits",
            "gpu_holder_thermal_limit_count",
            datasource=datasource,
            x=12,
            y=0,
            width=4,
            height=4,
        ),
        _stat_panel(
            5,
            "Owned workers",
            "gpu_holder_owned_worker_count",
            datasource=datasource,
            x=16,
            y=0,
            width=4,
            height=4,
        ),
        _stat_panel(
            6,
            "Policy window coverage",
            "gpu_holder_machine_policy_window_coverage_ratio * 100",
            datasource=datasource,
            x=20,
            y=0,
            width=4,
            height=4,
            unit="percent",
        ),
        _timeseries_panel(
            7,
            "Machine utilization",
            [
                ("current", "gpu_holder_machine_average_utilization_percent"),
                ("policy_window", "gpu_holder_machine_policy_average_utilization_percent"),
                ("target", "gpu_holder_target_utilization_percent"),
            ],
            datasource=datasource,
            x=0,
            y=4,
            width=12,
            height=8,
            unit="percent",
        ),
        _timeseries_panel(
            8,
            "Quota forecast",
            [
                ("required_remaining", "gpu_holder_target_required_remaining_average_percent"),
                ("target_gap", "gpu_holder_target_gap_percent"),
            ],
            datasource=datasource,
            x=12,
            y=4,
            width=12,
            height=8,
            unit="percent",
        ),
        _timeseries_panel(
            9,
            "Per-GPU utilization",
            [("{{gpu}} {{name}}", "gpu_holder_gpu_utilization_percent")],
            datasource=datasource,
            x=0,
            y=12,
            width=12,
            height=8,
            unit="percent",
        ),
        _timeseries_panel(
            10,
            "Per-GPU memory",
            [("{{gpu}} {{name}}", "gpu_holder_gpu_memory_used_bytes")],
            datasource=datasource,
            x=12,
            y=12,
            width=12,
            height=8,
            unit="bytes",
        ),
        _timeseries_panel(
            11,
            "Per-GPU temperature",
            [("{{gpu}} {{name}}", "gpu_holder_gpu_temperature_celsius")],
            datasource=datasource,
            x=0,
            y=20,
            width=12,
            height=8,
            unit="celsius",
        ),
        _timeseries_panel(
            12,
            "External and protected processes",
            [
                ("external {{gpu}}", "gpu_holder_gpu_external_process_count"),
                ("protected {{gpu}}", "gpu_holder_gpu_protected_process_count"),
            ],
            datasource=datasource,
            x=12,
            y=20,
            width=12,
            height=8,
        ),
    ]
    return {
        "annotations": {"list": []},
        "editable": True,
        "graphTooltip": 1,
        "panels": panels,
        "refresh": "30s",
        "schemaVersion": 39,
        "style": "dark",
        "tags": ["gpu-holder", "gpu", "cuda"],
        "templating": {
            "list": [
                {
                    "name": "DS_PROMETHEUS",
                    "type": "datasource",
                    "query": "prometheus",
                    "current": {"text": datasource, "value": datasource},
                }
            ]
        },
        "time": {"from": "now-6h", "to": "now"},
        "timezone": "browser",
        "title": title,
        "uid": uid,
        "version": 1,
    }


def _stat_panel(
    panel_id: int,
    title: str,
    expression: str,
    *,
    datasource: str,
    x: int,
    y: int,
    width: int,
    height: int,
    unit: str = "short",
) -> dict[str, object]:
    panel = _base_panel(panel_id, title, datasource=datasource, x=x, y=y, width=width, height=height)
    panel.update(
        {
            "type": "stat",
            "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
            "options": {
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto",
                "orientation": "auto",
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                "textMode": "auto",
            },
            "targets": [_target(expression, legend=title, datasource=datasource)],
        }
    )
    return panel


def _timeseries_panel(
    panel_id: int,
    title: str,
    series: list[tuple[str, str]],
    *,
    datasource: str,
    x: int,
    y: int,
    width: int,
    height: int,
    unit: str = "short",
) -> dict[str, object]:
    panel = _base_panel(panel_id, title, datasource=datasource, x=x, y=y, width=width, height=height)
    panel.update(
        {
            "type": "timeseries",
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "lineInterpolation": "linear",
                        "lineWidth": 2,
                        "showPoints": "never",
                    },
                    "unit": unit,
                },
                "overrides": [],
            },
            "options": {
                "legend": {"calcs": ["lastNotNull"], "displayMode": "table", "placement": "bottom"},
                "tooltip": {"mode": "multi", "sort": "none"},
            },
            "targets": [
                _target(expression, legend=legend, datasource=datasource)
                for legend, expression in series
            ],
        }
    )
    return panel


def _base_panel(
    panel_id: int,
    title: str,
    *,
    datasource: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> dict[str, object]:
    return {
        "datasource": datasource,
        "gridPos": {"h": height, "w": width, "x": x, "y": y},
        "id": panel_id,
        "title": title,
    }


def _target(expression: str, *, legend: str, datasource: str) -> dict[str, object]:
    return {
        "datasource": datasource,
        "editorMode": "code",
        "expr": expression,
        "legendFormat": legend,
        "range": True,
        "refId": _ref_id(legend),
    }


def _ref_id(value: str) -> str:
    letters = [char for char in value.upper() if char.isalpha()]
    return "".join(letters[:2] or ["A"])
