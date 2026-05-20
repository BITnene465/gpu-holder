from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import ast

from .units import MemorySpec, parse_memory_spec
from .worker import _program_sequence


DEFAULT_STATE_DIR = Path.home() / ".gpu-holder"


DEFAULT_CONFIG_TEMPLATE = """[guard]
# Runtime profile. Profile values are defaults; explicit config and CLI values
# still override them.
profile = "balanced"

# Select GPUs by index list or use "all".
gpus = "all"

# Keep the machine above a safe utilization target.
target_util = 75
idle_util = 50
idle_window = 60
machine_window = 3600

# Conservative default memory hold.
mem = "20%"
reserve = "2GiB"
assist_mem = "512MiB"
busy_process_mem_threshold = "10GiB"

# Thermal safety. Set to 0 to disable the thermal release guard.
max_gpu_temp = 85
# Resume holders only after cooling to this temperature. Set to 0 to resume
# as soon as the GPU falls below max_gpu_temp.
thermal_resume_temp = 80

# Worker behavior.
sample_interval = 2
program = "mixed"
hold_mode = "balanced"
compute_burst_seconds = 0.20
compute_burst_jitter = 0.20

# Adaptive duty-cycle control. Weights must sum to 1.0.
min_duty_cycle = 0.10
max_duty_cycle = 1.0
current_gap_weight = 0.45
history_gap_weight = 0.25
machine_gap_weight = 0.30

# Yield briefly when a new non-holder GPU process appears.
process_grace_window = 15
# Always yield to non-holder processes whose names match these shell-style
# patterns. Matching is case-insensitive.
protected_process_patterns = []
# Set to an integer to cap concurrent holder workers, e.g. 2.
# max_held_gpus = 2

# Runtime files.
state_dir = "~/.gpu-holder"
log_interval = 10
event_log_max_size = "10MiB"
event_log_backup_count = 3
worker_restart_backoff = 30
worker_start_timeout = 10
worker_update_duty_threshold = 0.05
"""


def config_template(*, profile: str = "balanced", minimal: bool = False) -> str:
    if profile not in CONFIG_PROFILES:
        supported = ", ".join(CONFIG_PROFILES)
        raise ValueError(f"unsupported profile: {profile}; supported: {supported}")
    if not minimal and profile == "balanced":
        return DEFAULT_CONFIG_TEMPLATE
    return "\n".join(
        [
            "[guard]",
            "# Runtime profile. Profile values are defaults; explicit config and CLI values",
            "# still override them.",
            f'profile = "{profile}"',
            "",
            "# Select GPUs by index list or use \"all\".",
            '# gpus = "all"',
            "",
            "# Uncomment only the settings you want to override from the selected profile.",
            '# target_util = 75',
            '# mem = "20%"',
            '# max_held_gpus = 2',
            "",
            "# Runtime files.",
            '# state_dir = "~/.gpu-holder"',
            "",
        ]
    )


@dataclass(frozen=True)
class ConfigField:
    key: str
    cli_flag: str | None
    value_type: str
    default: object
    category: str
    description: str
    example: str | None = None


@dataclass(frozen=True)
class ConfigProfile:
    name: str
    description: str
    values: dict[str, object]


@dataclass(frozen=True)
class ConfigRecipe:
    name: str
    description: str
    profile: str
    values: dict[str, object]


CONFIG_PROFILES = {
    "balanced": ConfigProfile(
        name="balanced",
        description="Default quota-oriented behavior with polite yielding and mixed workloads.",
        values={},
    ),
    "conservative": ConfigProfile(
        name="conservative",
        description="Lower-impact holders for busy shared machines or first-time rollout.",
        values={
            "target_util": 70,
            "mem": "10%",
            "max_duty_cycle": 0.60,
            "max_held_gpus": 1,
            "process_grace_window": 30,
            "compute_burst_jitter": 0.35,
        },
    ),
    "quota": ConfigProfile(
        name="quota",
        description="More assertive defaults for machines with strict hourly utilization reclaim rules.",
        values={
            "target_util": 80,
            "idle_util": 55,
            "mem": "20%",
            "program": "random",
            "max_duty_cycle": 1.0,
            "process_grace_window": 10,
        },
    ),
    "compute-only": ConfigProfile(
        name="compute-only",
        description="Avoids large memory holds and focuses on compute utilization.",
        values={
            "hold_mode": "compute-only",
            "mem": "0%",
            "assist_mem": "256MiB",
            "program": "random",
            "compute_burst_jitter": 0.40,
        },
    ),
}


CONFIG_RECIPES = {
    "first-run": ConfigRecipe(
        name="first-run",
        description="Low-impact first rollout for a busy shared machine.",
        profile="conservative",
        values={
            "target_util": 70,
            "mem": "10%",
            "max_held_gpus": 1,
            "program": "mixed",
            "process_grace_window": 30,
        },
    ),
    "strict-quota": ConfigRecipe(
        name="strict-quota",
        description="More assertive setup for machines with strict hourly reclaim rules.",
        profile="quota",
        values={
            "target_util": 80,
            "idle_util": 55,
            "mem": "20%",
            "program": "random",
            "process_grace_window": 10,
        },
    ),
    "busy-shared": ConfigRecipe(
        name="busy-shared",
        description="Yield-heavy setup for machines that often run real user jobs.",
        profile="conservative",
        values={
            "target_util": 70,
            "mem": "5%",
            "reserve": "4GiB",
            "max_held_gpus": 1,
            "process_grace_window": 45,
            "protected_process_patterns": ["*train*", "*serve*", "*vllm*"],
        },
    ),
    "compute-only": ConfigRecipe(
        name="compute-only",
        description="Minimal-memory setup that focuses on compute utilization.",
        profile="compute-only",
        values={
            "hold_mode": "compute-only",
            "mem": "0%",
            "assist_mem": "256MiB",
            "program": "random",
            "compute_burst_jitter": 0.40,
        },
    ),
}


@dataclass(frozen=True)
class GuardConfig:
    profile: str = "balanced"
    gpus: tuple[int, ...] | str = "all"
    target_util: int = 75
    idle_util: int = 50
    idle_window: float = 60.0
    machine_window: float = 3600.0
    mem: MemorySpec = parse_memory_spec("20%")
    reserve: MemorySpec = parse_memory_spec("2GiB")
    busy_process_mem_threshold: MemorySpec = parse_memory_spec("10GiB")
    assist_mem: MemorySpec = parse_memory_spec("512MiB")
    max_gpu_temp: int | None = 85
    thermal_resume_temp: int | None = 80
    sample_interval: float = 2.0
    program: str = "mixed"
    hold_mode: str = "balanced"
    compute_burst_seconds: float = 0.20
    compute_burst_jitter: float = 0.20
    max_duty_cycle: float = 1.0
    min_duty_cycle: float = 0.10
    current_gap_weight: float = 0.45
    history_gap_weight: float = 0.25
    machine_gap_weight: float = 0.30
    event_log_max_bytes: int = 10 * 1024**2
    event_log_backup_count: int = 3
    worker_restart_backoff: float = 30.0
    worker_start_timeout: float = 10.0
    worker_update_duty_threshold: float = 0.05
    process_grace_window: float = 15.0
    protected_process_patterns: tuple[str, ...] = ()
    max_held_gpus: int | None = None
    state_dir: Path = DEFAULT_STATE_DIR
    pause_file: Path | None = None
    log_interval: float = 10.0
    dry_run: bool = False
    source_errors: tuple[str, ...] = ()

    @property
    def resolved_pause_file(self) -> Path:
        return self.pause_file or (self.state_dir / "pause")

    @property
    def pid_file(self) -> Path:
        return self.state_dir / "gpu-holder.pid"

    @property
    def status_file(self) -> Path:
        return self.state_dir / "status.json"

    @property
    def log_file(self) -> Path:
        return self.state_dir / "gpu-holder.log"

    @property
    def event_log_file(self) -> Path:
        return self.state_dir / "events.jsonl"


def config_reference() -> list[dict[str, object]]:
    default = GuardConfig()
    fields = [
        ConfigField(
            "profile",
            "--profile",
            "balanced | conservative | quota | compute-only",
            default.profile,
            "policy",
            "Runtime profile. Profile values are defaults that explicit config and CLI values can override.",
            "conservative",
        ),
        ConfigField(
            "gpus",
            "--gpus",
            "all | comma-list | TOML array",
            "all",
            "selection",
            "GPU indices to manage. Use all to select every visible NVIDIA GPU.",
            "all or 0,1,3",
        ),
        ConfigField(
            "target_util",
            "--target-util",
            "int percent",
            default.target_util,
            "policy",
            "Machine-average utilization target. Holders start when the policy average is below this value.",
            "75",
        ),
        ConfigField(
            "idle_util",
            "--idle-util",
            "int percent",
            default.idle_util,
            "policy",
            "Per-GPU low-util emergency threshold.",
            "50",
        ),
        ConfigField(
            "idle_window",
            "--idle-window",
            "seconds",
            default.idle_window,
            "policy",
            "A GPU must stay below idle_util for this full window before emergency intervention.",
            "60",
        ),
        ConfigField(
            "machine_window",
            "--machine-window",
            "seconds",
            default.machine_window,
            "policy",
            "Rolling machine-average window used for quota-oriented target decisions.",
            "3600",
        ),
        ConfigField(
            "mem",
            "--mem",
            "memory spec",
            default.mem.raw,
            "memory",
            "Memory allocation target for balanced and memory-only holder workers.",
            "20%, 10GiB, 12000MiB",
        ),
        ConfigField(
            "reserve",
            "--reserve",
            "memory spec",
            default.reserve.raw,
            "memory",
            "Free memory to leave unallocated on each GPU.",
            "2GiB",
        ),
        ConfigField(
            "assist_mem",
            "--assist-mem",
            "memory spec",
            default.assist_mem.raw,
            "memory",
            "Smaller memory target used by assist mode during low-util emergency on busy GPUs.",
            "512MiB",
        ),
        ConfigField(
            "busy_process_mem_threshold",
            "--busy-process-mem-threshold",
            "memory spec",
            default.busy_process_mem_threshold.raw,
            "safety",
            "Non-holder GPU process memory threshold that normally makes gpu-holder yield the GPU.",
            "10GiB",
        ),
        ConfigField(
            "max_gpu_temp",
            "--max-gpu-temp",
            "int Celsius | 0",
            default.max_gpu_temp,
            "safety",
            "Release and block holder workers at or above this GPU temperature. Use 0 to disable.",
            "85",
        ),
        ConfigField(
            "thermal_resume_temp",
            "--thermal-resume-temp",
            "int Celsius | 0",
            default.thermal_resume_temp,
            "safety",
            "Resume after a thermal block only once the GPU cools to this temperature. Use 0 for no hysteresis.",
            "80",
        ),
        ConfigField(
            "sample_interval",
            "--sample-interval",
            "seconds",
            default.sample_interval,
            "runtime",
            "Controller polling interval.",
            "2",
        ),
        ConfigField(
            "program",
            "--program",
            "mixed | random | matmul | conv | fft | elementwise | comma-list",
            default.program,
            "worker",
            "CUDA compute program used by holder workers. Comma lists rotate through the selected programs.",
            "mixed or matmul,conv,fft",
        ),
        ConfigField(
            "hold_mode",
            "--hold-mode",
            "balanced | memory-only | compute-only",
            default.hold_mode,
            "worker",
            "Holder shape. Balanced uses memory and compute; memory-only avoids compute; compute-only avoids large memory hold.",
            "balanced",
        ),
        ConfigField(
            "compute_burst_seconds",
            "--compute-burst-seconds",
            "seconds",
            default.compute_burst_seconds,
            "worker",
            "Base compute burst length before duty-cycle sleep is applied.",
            "0.20",
        ),
        ConfigField(
            "compute_burst_jitter",
            "--compute-burst-jitter",
            "float 0..1",
            default.compute_burst_jitter,
            "worker",
            "Symmetric random jitter applied to each compute burst length.",
            "0.20",
        ),
        ConfigField(
            "min_duty_cycle",
            "--min-duty-cycle",
            "float 0..1",
            default.min_duty_cycle,
            "adaptive-load",
            "Lower bound for adaptive compute duty cycle.",
            "0.10",
        ),
        ConfigField(
            "max_duty_cycle",
            "--max-duty-cycle",
            "float 0..1",
            default.max_duty_cycle,
            "adaptive-load",
            "Upper bound for adaptive compute duty cycle.",
            "1.0",
        ),
        ConfigField(
            "current_gap_weight",
            "--current-gap-weight",
            "float",
            default.current_gap_weight,
            "adaptive-load",
            "Duty-cycle weight for the current per-GPU utilization gap.",
            "0.45",
        ),
        ConfigField(
            "history_gap_weight",
            "--history-gap-weight",
            "float",
            default.history_gap_weight,
            "adaptive-load",
            "Duty-cycle weight for the rolling per-GPU history gap.",
            "0.25",
        ),
        ConfigField(
            "machine_gap_weight",
            "--machine-gap-weight",
            "float",
            default.machine_gap_weight,
            "adaptive-load",
            "Duty-cycle weight for the rolling machine-average gap.",
            "0.30",
        ),
        ConfigField(
            "process_grace_window",
            "--process-grace-window",
            "seconds",
            default.process_grace_window,
            "safety",
            "Yield period after a new non-holder GPU process appears.",
            "15",
        ),
        ConfigField(
            "protected_process_patterns",
            "--protected-process",
            "string list | comma-list",
            list(default.protected_process_patterns),
            "safety",
            "Case-insensitive shell-style process name patterns that always make gpu-holder yield that GPU.",
            "python train.py, vllm*, *serve*",
        ),
        ConfigField(
            "max_held_gpus",
            "--max-held-gpus",
            "int | unset",
            default.max_held_gpus,
            "policy",
            "Optional cap on concurrent holder workers.",
            "2",
        ),
        ConfigField(
            "state_dir",
            "--state-dir",
            "path",
            str(default.state_dir),
            "runtime",
            "Directory for pid, status, event log, pause, and runtime-disable state files.",
            "~/.gpu-holder",
        ),
        ConfigField(
            "pause_file",
            "--pause-file",
            "path | unset",
            default.pause_file,
            "runtime",
            "Optional custom pause file. When it exists, holder workers are released.",
            None,
        ),
        ConfigField(
            "log_interval",
            "--log-interval",
            "seconds",
            default.log_interval,
            "observability",
            "Interval for compact runtime summary logs. Use 0 to log every controller sample.",
            "10",
        ),
        ConfigField(
            "event_log_max_size",
            "--event-log-max-size",
            "absolute memory spec",
            "10MiB",
            "observability",
            "Rotate events.jsonl after this size.",
            "10MiB",
        ),
        ConfigField(
            "event_log_backup_count",
            "--event-log-backup-count",
            "int",
            default.event_log_backup_count,
            "observability",
            "Number of rotated event log backups to keep.",
            "3",
        ),
        ConfigField(
            "worker_restart_backoff",
            "--worker-restart-backoff",
            "seconds",
            default.worker_restart_backoff,
            "runtime",
            "Per-GPU backoff after worker startup failure.",
            "30",
        ),
        ConfigField(
            "worker_start_timeout",
            "--worker-start-timeout",
            "seconds",
            default.worker_start_timeout,
            "runtime",
            "Maximum time to wait for a worker to report CUDA readiness before stopping it.",
            "10",
        ),
        ConfigField(
            "worker_update_duty_threshold",
            "--worker-update-duty-threshold",
            "float 0..1",
            default.worker_update_duty_threshold,
            "runtime",
            "Restart an owned worker only when the requested duty cycle differs by at least this amount.",
            "0.05",
        ),
    ]
    return [asdict(field) for field in fields]


def profile_reference() -> list[dict[str, object]]:
    return [
        {
            "name": profile.name,
            "description": profile.description,
            "values": dict(profile.values),
        }
        for profile in CONFIG_PROFILES.values()
    ]


def profile_defaults(profile_name: str) -> dict[str, object]:
    try:
        return dict(CONFIG_PROFILES[profile_name].values)
    except KeyError as exc:
        supported = ", ".join(CONFIG_PROFILES)
        raise ValueError(f"unsupported profile: {profile_name}; supported: {supported}") from exc


def recipe_reference() -> list[dict[str, object]]:
    return [
        {
            "name": recipe.name,
            "description": recipe.description,
            "profile": recipe.profile,
            "values": dict(recipe.values),
        }
        for recipe in CONFIG_RECIPES.values()
    ]


def recipe_template(recipe_name: str) -> str:
    try:
        recipe = CONFIG_RECIPES[recipe_name]
    except KeyError as exc:
        supported = ", ".join(CONFIG_RECIPES)
        raise ValueError(f"unsupported recipe: {recipe_name}; supported: {supported}") from exc
    lines = [
        "[guard]",
        f"# {recipe.description}",
        f'profile = "{recipe.profile}"',
        "",
    ]
    for key, value in recipe.values.items():
        if key == "profile":
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    lines.append("")
    return "\n".join(lines)


def config_payload(config: GuardConfig) -> dict[str, object]:
    return {
        "profile": config.profile,
        "gpus": list(config.gpus) if config.gpus != "all" else "all",
        "target_util": config.target_util,
        "idle_util": config.idle_util,
        "idle_window": config.idle_window,
        "machine_window": config.machine_window,
        "mem": config.mem.raw,
        "reserve": config.reserve.raw,
        "busy_process_mem_threshold": config.busy_process_mem_threshold.raw,
        "assist_mem": config.assist_mem.raw,
        "max_gpu_temp": config.max_gpu_temp,
        "thermal_resume_temp": config.thermal_resume_temp,
        "sample_interval": config.sample_interval,
        "program": config.program,
        "hold_mode": config.hold_mode,
        "compute_burst_seconds": config.compute_burst_seconds,
        "compute_burst_jitter": config.compute_burst_jitter,
        "max_duty_cycle": config.max_duty_cycle,
        "min_duty_cycle": config.min_duty_cycle,
        "current_gap_weight": config.current_gap_weight,
        "history_gap_weight": config.history_gap_weight,
        "machine_gap_weight": config.machine_gap_weight,
        "event_log_max_bytes": config.event_log_max_bytes,
        "event_log_backup_count": config.event_log_backup_count,
        "worker_restart_backoff": config.worker_restart_backoff,
        "worker_start_timeout": config.worker_start_timeout,
        "worker_update_duty_threshold": config.worker_update_duty_threshold,
        "process_grace_window": config.process_grace_window,
        "protected_process_patterns": list(config.protected_process_patterns),
        "max_held_gpus": config.max_held_gpus,
        "state_dir": str(config.state_dir),
        "pause_file": str(config.pause_file) if config.pause_file else None,
        "log_interval": config.log_interval,
        "pid_file": str(config.pid_file),
        "status_file": str(config.status_file),
        "event_log_file": str(config.event_log_file),
        "log_file": str(config.log_file),
        "dry_run": config.dry_run,
    }


def validate_config(config: GuardConfig) -> tuple[list[str], list[str]]:
    errors: list[str] = list(config.source_errors)
    warnings: list[str] = []

    if config.profile not in CONFIG_PROFILES:
        errors.append(f"unsupported profile: {config.profile}")

    if config.gpus != "all":
        if not config.gpus:
            errors.append("gpus must not be empty")
        if len(set(config.gpus)) != len(config.gpus):
            errors.append("gpus must not contain duplicate indices")
        if any(index < 0 for index in config.gpus):
            errors.append("gpu indices must be non-negative")

    if not 1 <= config.target_util <= 100:
        errors.append("target_util must be between 1 and 100")
    if not 0 <= config.idle_util <= 100:
        errors.append("idle_util must be between 0 and 100")
    if config.idle_util >= config.target_util:
        warnings.append("idle_util is greater than or equal to target_util; emergency mode may dominate")
    if config.idle_window <= 0:
        errors.append("idle_window must be positive")
    if config.machine_window <= 0:
        errors.append("machine_window must be positive")
    if config.sample_interval <= 0:
        errors.append("sample_interval must be positive")
    if config.sample_interval > config.idle_window:
        warnings.append("sample_interval is greater than idle_window; emergency detection will be coarse")
    if config.max_gpu_temp is not None and config.max_gpu_temp <= 0:
        errors.append("max_gpu_temp must be positive when enabled")
    if config.thermal_resume_temp is not None and config.thermal_resume_temp <= 0:
        errors.append("thermal_resume_temp must be positive when enabled")
    if (
        config.max_gpu_temp is not None
        and config.thermal_resume_temp is not None
        and config.thermal_resume_temp >= config.max_gpu_temp
    ):
        errors.append("thermal_resume_temp must be lower than max_gpu_temp")

    try:
        _program_sequence(config.program)
    except ValueError as exc:
        errors.append(f"unsupported program: {exc}")
    if config.hold_mode not in {"balanced", "memory-only", "compute-only"}:
        errors.append(f"unsupported hold_mode: {config.hold_mode}")
    if config.compute_burst_seconds <= 0:
        errors.append("compute_burst_seconds must be positive")
    if not 0 <= config.compute_burst_jitter <= 1:
        errors.append("compute_burst_jitter must be between 0 and 1")

    if not 0 <= config.min_duty_cycle <= config.max_duty_cycle <= 1:
        errors.append("duty cycle bounds must satisfy 0 <= min <= max <= 1")
    weight_sum = (
        config.current_gap_weight + config.history_gap_weight + config.machine_gap_weight
    )
    if any(
        weight < 0
        for weight in (
            config.current_gap_weight,
            config.history_gap_weight,
            config.machine_gap_weight,
        )
    ):
        errors.append("duty cycle weights must be non-negative")
    if abs(weight_sum - 1.0) > 0.001:
        warnings.append(f"duty cycle weights sum to {weight_sum:.3f}, not 1.0")
    if config.event_log_max_bytes <= 0:
        errors.append("event_log_max_bytes must be positive")
    if config.log_interval < 0:
        errors.append("log_interval must be non-negative")
    if config.event_log_backup_count < 0:
        errors.append("event_log_backup_count must be non-negative")
    if config.worker_restart_backoff < 0:
        errors.append("worker_restart_backoff must be non-negative")
    if config.worker_start_timeout <= 0:
        errors.append("worker_start_timeout must be positive")
    if not 0 <= config.worker_update_duty_threshold <= 1:
        errors.append("worker_update_duty_threshold must be between 0 and 1")
    if config.process_grace_window < 0:
        errors.append("process_grace_window must be non-negative")
    if any(not pattern.strip() for pattern in config.protected_process_patterns):
        errors.append("protected_process_patterns must not contain empty patterns")
    if config.max_held_gpus is not None and config.max_held_gpus < 0:
        errors.append("max_held_gpus must be non-negative")

    return errors, warnings


def load_config_file(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    payload = _load_toml(config_path)
    guard = payload.get("guard", payload)
    if not isinstance(guard, dict):
        raise ValueError("config file must contain a table or [guard] table")
    return dict(guard)


def validate_config_keys(file_config: dict[str, object]) -> list[str]:
    known_keys = _known_config_keys()
    errors: list[str] = []
    for key in file_config:
        if key not in known_keys:
            errors.append(f"unknown config key: {key}")
    return errors


def _known_config_keys() -> set[str]:
    keys: set[str] = set()
    for field in config_reference():
        key = field.get("key")
        if not isinstance(key, str):
            continue
        keys.add(key)
        keys.add(key.replace("_", "-"))
    return keys


def _toml_value(value: object) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if value is None:
        return '""'
    return str(value)


def _load_toml(path: Path) -> dict[str, object]:
    try:
        import tomllib
    except ModuleNotFoundError:
        return _load_simple_toml(path)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_simple_toml(path: Path) -> dict[str, object]:
    root: dict[str, object] = {}
    current = root
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            table_name = line[1:-1].strip()
            table: dict[str, object] = {}
            root[table_name] = table
            current = table
            continue
        if "=" not in line:
            raise ValueError(f"unsupported TOML line: {raw_line!r}")
        key, value = line.split("=", 1)
        current[key.strip()] = _parse_simple_toml_value(value.strip())
    return root


def _parse_simple_toml_value(value: str) -> object:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
