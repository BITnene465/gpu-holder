from __future__ import annotations

from gpu_holder.service import generate_systemd_user_service


def test_generate_systemd_user_service_defaults_to_guard_command() -> None:
    unit = generate_systemd_user_service()

    assert "[Unit]" in unit
    assert "Description=gpu-holder guard" in unit
    assert "[Service]" in unit
    assert "Type=simple" in unit
    assert "ExecStart=gpu-holder guard" in unit
    assert "Restart=always" in unit
    assert "RestartSec=10" in unit
    assert "[Install]" in unit
    assert "WantedBy=default.target" in unit


def test_generate_systemd_user_service_includes_runtime_options() -> None:
    unit = generate_systemd_user_service(
        executable="/opt/gpu holder/bin/gpu-holder",
        config_path="/etc/gpu-holder.toml",
        state_dir="/tmp/gpu holder",
        working_directory="/srv/gpu holder",
        restart_seconds=30,
        extra_args=("--gpus", "0,1", "--dry-run"),
    )

    assert 'WorkingDirectory="/srv/gpu holder"' in unit
    assert '"/opt/gpu holder/bin/gpu-holder" guard' in unit
    assert "--config /etc/gpu-holder.toml" in unit
    assert '--state-dir "/tmp/gpu holder"' in unit
    assert "--gpus 0,1 --dry-run" in unit
    assert "RestartSec=30" in unit
