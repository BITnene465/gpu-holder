from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlertRule:
    name: str
    expression: str
    duration: str
    severity: str
    summary: str
    description: str


DEFAULT_ALERT_GROUP = "gpu-holder"


ALERT_RULES: tuple[AlertRule, ...] = (
    AlertRule(
        name="GPUHolderStatusMissing",
        expression="gpu_holder_status_available == 0",
        duration="2m",
        severity="critical",
        summary="gpu-holder status is missing",
        description="No valid gpu-holder status payload is available to Prometheus.",
    ),
    AlertRule(
        name="GPUHolderStatusStale",
        expression="gpu_holder_status_stale == 1",
        duration="2m",
        severity="warning",
        summary="gpu-holder status is stale",
        description="gpu-holder status has not been refreshed within the configured stale window.",
    ),
    AlertRule(
        name="GPUHolderQuotaForecastAtRisk",
        expression="gpu_holder_target_forecast_at_risk == 1",
        duration="5m",
        severity="warning",
        summary="GPU utilization target forecast is unrecoverable",
        description=(
            "The remaining policy window would need more than 100 percent average "
            "utilization to reach the configured target."
        ),
    ),
    AlertRule(
        name="GPUHolderTargetBelow",
        expression="gpu_holder_target_gap_percent < 0",
        duration="15m",
        severity="warning",
        summary="GPU utilization policy average is below target",
        description="The policy/window machine-average utilization is below target_util.",
    ),
    AlertRule(
        name="GPUHolderWorkerStartBackoff",
        expression="gpu_holder_worker_backoff_count > 0",
        duration="5m",
        severity="warning",
        summary="gpu-holder worker startup backoff is active",
        description="One or more holder workers failed to start and are in restart backoff.",
    ),
    AlertRule(
        name="GPUHolderThermalLimit",
        expression="gpu_holder_thermal_limit_count > 0",
        duration="5m",
        severity="info",
        summary="gpu-holder is yielding because of thermal limits",
        description="At least one GPU is thermally blocked by max_gpu_temp or hysteresis.",
    ),
)


def alert_rules_payload(*, group_name: str = DEFAULT_ALERT_GROUP) -> dict[str, object]:
    return {
        "groups": [
            {
                "name": group_name,
                "rules": [
                    {
                        "alert": rule.name,
                        "expr": rule.expression,
                        "for": rule.duration,
                        "labels": {"severity": rule.severity},
                        "annotations": {
                            "summary": rule.summary,
                            "description": rule.description,
                        },
                    }
                    for rule in ALERT_RULES
                ],
            }
        ]
    }


def render_prometheus_alert_rules(*, group_name: str = DEFAULT_ALERT_GROUP) -> str:
    lines = ["groups:", f"- name: {_quote_yaml(group_name)}", "  rules:"]
    for rule in ALERT_RULES:
        lines.extend(
            [
                f"  - alert: {rule.name}",
                f"    expr: {_quote_yaml(rule.expression)}",
                f"    for: {_quote_yaml(rule.duration)}",
                "    labels:",
                f"      severity: {_quote_yaml(rule.severity)}",
                "    annotations:",
                f"      summary: {_quote_yaml(rule.summary)}",
                f"      description: {_quote_yaml(rule.description)}",
            ]
        )
    return "\n".join(lines) + "\n"


def _quote_yaml(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
