from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from . import __version__
from .backends import DEFAULT_BACKEND
from .backends import SUPPORTED_DIAGNOSTIC_BACKENDS
from .backends import SUPPORTED_WORKER_BACKENDS
from .backends import check_backend
from .backends import normalize_backend
from .models import Decision, GpuProcess, GpuSnapshot
from .policy import decide
from .policy import make_hold
from .policy import make_release
from .policy import parse_bytes
from .policy import process_signature
from .policy import resolve_memory_ratio
from .policy import resolve_memory_spec
from .worker import WorkerProcess


STATE_DIR = Path.home() / ".gpu-holder"
BASE_PROGRAMS = {"matmul", "conv", "fft", "elementwise", "mixed", "random"}

__all__ = [
    "Config",
    "Decision",
    "GpuProcess",
    "GpuSnapshot",
    "decide",
    "make_hold",
    "make_release",
    "parse_bytes",
    "process_signature",
    "resolve_memory_ratio",
    "resolve_memory_spec",
]


@dataclass(frozen=True)
class Config:
    gpus: tuple[int, ...] | str = "all"
    target_util: float = 0.75
    risk_util: float = 0.5
    mem: float = 0.2
    reserve: str = "2GiB"
    busy_process_mem_threshold: str = "10GiB"
    assist_mem: str = "512MiB"
    sample_interval: float = 2.0
    backend: str = DEFAULT_BACKEND
    program: str = "matmul"
    min_duty_cycle: float = 0.0
    max_duty_cycle: float = 1.0
    compute_burst_seconds: float = 0.2
    compute_burst_jitter: float = 0.0
    process_grace_window: float = 120.0
    state_dir: Path = STATE_DIR
    log_interval: float = 10.0
    dry_run: bool = False
    once: bool = False

    @property
    def pid_file(self) -> Path:
        return self.state_dir / "gpu-holder.pid"

    @property
    def status_file(self) -> Path:
        return self.state_dir / "status.json"

    @property
    def log_file(self) -> Path:
        return self.state_dir / "gpu-holder.log"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "guard":
        return cmd_guard(args)
    if args.command == "start":
        return cmd_start(args)
    if args.command == "stop":
        return cmd_stop(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "dashboard":
        return cmd_dashboard(args)
    if args.command == "doctor":
        return cmd_doctor(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-holder",
        description="Small CUDA GPU holder that keeps utilization high and yields to real jobs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")
    add_run_args(sub.add_parser("guard", help="run in foreground"), include_controls=True)
    add_run_args(sub.add_parser("start", help="start background guard"))
    stop = sub.add_parser("stop", help="stop background guard")
    stop.add_argument("--state-dir", default=str(STATE_DIR))
    status = sub.add_parser("status", help="print latest status")
    status.add_argument("--state-dir", default=str(STATE_DIR))
    status.add_argument("--json", action="store_true")
    status.add_argument("--check", action="store_true")
    dashboard = sub.add_parser("dashboard", help="print compact status snapshot")
    dashboard.add_argument("--state-dir", default=str(STATE_DIR))
    doctor = sub.add_parser("doctor", help="check nvidia-smi and worker backend")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--backend", default=DEFAULT_BACKEND, choices=SUPPORTED_DIAGNOSTIC_BACKENDS)
    return parser


def add_run_args(parser: argparse.ArgumentParser, *, include_controls: bool = False) -> None:
    parser.add_argument("--gpus", default="all", help="all, comma list, or ranges like 0-7")
    parser.add_argument(
        "--target-util",
        type=parse_ratio,
        default=0.75,
        help="Target holder utilization, from 0 to 1. Legacy percent like 75 or 75%% is accepted.",
    )
    parser.add_argument(
        "--risk-util",
        type=parse_ratio,
        default=0.5,
        help="Per-GPU utilization risk threshold for starting a holder, from 0 to 1.",
    )
    parser.add_argument(
        "--idle-util",
        dest="risk_util",
        type=parse_ratio,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mem",
        type=parse_mem_ratio,
        default=0.2,
        help="GPU memory fraction to hold, from 0 to 1. Legacy percent like 20%% is accepted.",
    )
    parser.add_argument("--reserve", default="2GiB")
    parser.add_argument("--busy-process-mem-threshold", default="10GiB")
    parser.add_argument("--assist-mem", default="512MiB")
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=SUPPORTED_WORKER_BACKENDS,
        help="worker backend to use",
    )
    parser.add_argument("--program", default="matmul")
    parser.add_argument("--min-duty-cycle", type=float, default=0.0)
    parser.add_argument("--max-duty-cycle", type=float, default=1.0)
    parser.add_argument("--compute-burst-seconds", type=float, default=0.2)
    parser.add_argument("--compute-burst-jitter", type=float, default=0.0)
    parser.add_argument("--process-grace-window", type=float, default=120.0)
    parser.add_argument("--state-dir", default=str(STATE_DIR))
    parser.add_argument("--log-interval", type=float, default=10.0)
    if include_controls:
        parser.add_argument("--dry-run", action="store_true", help="print one decision snapshot only")
        parser.add_argument("--once", action="store_true", help="run one guard iteration and exit")


def cmd_guard(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    error = validate_config(config)
    if error:
        print(error, file=sys.stderr)
        return 2
    guard = Guard(config)
    try:
        return guard.run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def cmd_start(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    error = validate_config(config)
    if error:
        print(error, file=sys.stderr)
        return 2
    config.state_dir.mkdir(parents=True, exist_ok=True)
    pid = read_pid(config.pid_file)
    if pid and is_gpu_holder_process(pid):
        print(f"gpu-holder already running pid={pid}", file=sys.stderr)
        return 1
    if pid and process_exists(pid):
        print(f"refusing pid file owned by non-holder pid={pid}", file=sys.stderr)
        return 2
    config.pid_file.unlink(missing_ok=True)
    command = [sys.executable, "-m", "gpu_holder.cli", "guard", *child_args(config)]
    log = config.log_file.open("ab")
    proc = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
    config.pid_file.write_text(str(proc.pid), encoding="utf-8")
    print(f"started gpu-holder pid={proc.pid} log={config.log_file}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).expanduser()
    pid_file = state_dir / "gpu-holder.pid"
    pid = read_pid(pid_file)
    if not pid:
        print("gpu-holder is not running")
        return 0
    if not process_exists(pid):
        pid_file.unlink(missing_ok=True)
        print("gpu-holder is not running")
        return 0
    if not is_gpu_holder_process(pid):
        print(f"refusing to stop non-holder pid={pid}", file=sys.stderr)
        return 2
    kill_target = ProcessKillTarget(pid)
    kill_target.terminate()
    deadline = time.monotonic() + 10.0
    while process_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if process_exists(pid):
        kill_target.kill()
        print(f"sent SIGKILL to slow holder pid={pid}")
    else:
        print(f"stopped gpu-holder pid={pid}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    status_file = Path(args.state_dir).expanduser() / "status.json"
    try:
        payload = json.loads(status_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print("no status file")
        return 1 if args.check else 0
    except json.JSONDecodeError as exc:
        print(f"invalid status file: {exc}")
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_status(payload))
    if args.check:
        return 0 if time.time() - float(payload.get("timestamp", 0)) < 30 else 1
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    status_file = Path(args.state_dir).expanduser() / "status.json"
    try:
        payload = json.loads(status_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print("no status file")
        return 1
    print(format_status(payload))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    checks = [
        {"name": "python", "ok": sys.version_info[:2] == (3, 10), "detail": sys.version.split()[0]},
        check_nvidia_smi(),
        check_backend(args.backend).as_payload(),
    ]
    payload = {"ok": all(item["ok"] for item in checks), "checks": checks}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for item in checks:
            status = "ok" if item["ok"] else "fail"
            print(f"{item['name']}: {status} {item['detail']}")
    return 0 if payload["ok"] else 1


class Guard:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.workers: dict[int, WorkerProcess] = {}
        self.external_process_first_seen: dict[int, dict[tuple[int, ...], float]] = {}
        self.stop = False
        self.last_log = 0.0

    def run(self) -> int:
        if self.config.dry_run:
            return self.run_once(apply=False, write_status=False, log_status=False, print_status=True)
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        existing = read_pid(self.config.pid_file)
        if existing and existing != os.getpid() and process_exists(existing):
            print(f"gpu-holder already running pid={existing}", file=sys.stderr)
            return 1
        self.config.pid_file.write_text(str(os.getpid()), encoding="utf-8")
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        log(self.config, f"started pid={os.getpid()} gpus={format_gpus(self.config.gpus)}")
        try:
            while not self.stop:
                self.run_once(apply=True, write_status=True, log_status=True, print_status=False)
                if self.config.once:
                    break
                time.sleep(max(0.2, self.config.sample_interval))
        finally:
            self.stop_all_workers()
            if not self.config.once:
                self.config.status_file.unlink(missing_ok=True)
            self.config.pid_file.unlink(missing_ok=True)
            log(self.config, "stopped")
        return 0

    def run_once(
        self,
        *,
        apply: bool,
        write_status: bool,
        log_status: bool,
        print_status: bool,
    ) -> int:
        snapshots = read_snapshots(self.config, self.workers)
        selected = select_gpus(snapshots, self.config.gpus)
        now = time.monotonic()
        if apply:
            self.update_external_process_first_seen(selected, now=now)
        decisions = decide(
            selected,
            self.config,
            external_process_first_seen=self.external_process_first_seen,
            now=now,
        )
        if apply:
            self.apply(decisions)
        payload = status_payload(selected, decisions, self.workers, self.config)
        if write_status:
            write_json(self.config.status_file, payload)
        if log_status:
            self.log_summary(payload)
        if print_status:
            print(format_status(payload))
        return 0

    def apply(self, decisions: list[Decision]) -> None:
        for decision in decisions:
            if decision.action == "hold":
                worker = self.workers.get(decision.gpu_index)
                if worker and worker.is_alive():
                    continue
                self.start_worker(decision)
            else:
                self.stop_worker(decision.gpu_index)

    def update_external_process_first_seen(
        self,
        snapshots: list[GpuSnapshot],
        *,
        now: float,
    ) -> None:
        visible_gpu_indices = {gpu.index for gpu in snapshots}
        for gpu in snapshots:
            external = [process for process in gpu.processes if not process.is_holder]
            if not external:
                self.external_process_first_seen.pop(gpu.index, None)
                continue
            signature = process_signature(external)
            gpu_seen = self.external_process_first_seen.setdefault(gpu.index, {})
            if signature not in gpu_seen:
                gpu_seen.clear()
                gpu_seen[signature] = now
        for gpu_index in list(self.external_process_first_seen):
            if gpu_index not in visible_gpu_indices:
                self.external_process_first_seen.pop(gpu_index, None)

    def start_worker(self, decision: Decision) -> None:
        worker = WorkerProcess(
            gpu_index=decision.gpu_index,
            memory_bytes=decision.memory_bytes,
            duty_cycle=decision.duty_cycle,
            program=self.config.program,
            hold_mode=decision.hold_mode,
            backend=self.config.backend,
            burst_seconds=self.config.compute_burst_seconds,
            burst_jitter=self.config.compute_burst_jitter,
        )
        try:
            worker.start(timeout=10)
        except Exception as exc:
            log(self.config, f"worker_start_failed gpu={decision.gpu_index} error={exc}")
            return
        self.workers[decision.gpu_index] = worker
        log(self.config, f"worker_start gpu={decision.gpu_index} pid={worker.pid}")

    def stop_worker(self, gpu_index: int) -> None:
        worker = self.workers.pop(gpu_index, None)
        if not worker:
            return
        log(self.config, f"worker_stop gpu={gpu_index} pid={worker.pid}")
        worker.stop(timeout=0.2)

    def stop_all_workers(self) -> None:
        workers = list(self.workers.items())
        self.workers.clear()
        threads: list[threading.Thread] = []
        for gpu_index, worker in workers:
            log(self.config, f"worker_stop gpu={gpu_index} pid={worker.pid}")
            thread = threading.Thread(target=worker.stop, kwargs={"timeout": 0.2}, daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join(timeout=1.5)

    def log_summary(self, payload: dict[str, object]) -> None:
        now = time.monotonic()
        if now - self.last_log < self.config.log_interval:
            return
        self.last_log = now
        log(self.config, compact_status(payload))

    def request_stop(self, signum: int, frame: object) -> None:
        del signum, frame
        self.stop = True


def read_snapshots(config: Config, workers: dict[int, WorkerProcess]) -> list[GpuSnapshot]:
    rows = run_csv(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.total,memory.used,memory.free,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[GpuSnapshot] = []
    for row in rows:
        if len(row) < 8:
            continue
        gpus.append(
            GpuSnapshot(
                index=int(row[0]),
                uuid=row[1],
                name=row[2],
                utilization=int(float(row[3] or 0)),
                memory_total=mib(row[4]),
                memory_used=mib(row[5]),
                memory_free=mib(row[6]),
                temperature=None if row[7] in {"", "[N/A]"} else int(float(row[7])),
                processes=[],
            )
        )
    by_uuid = {gpu.uuid: gpu for gpu in gpus}
    worker_pids = {worker.pid for worker in workers.values() if worker.pid}
    worker_mem = {gpu: worker.memory_bytes for gpu, worker in workers.items()}
    for row in run_process_rows():
        if len(row) < 4 or row[0] not in by_uuid:
            continue
        gpu = by_uuid[row[0]]
        pid = int(float(row[1]))
        used = mib(row[2])
        is_holder = pid in worker_pids or abs(used - worker_mem.get(gpu.index, -1)) < 1024**3
        gpu.processes.append(GpuProcess(pid=pid, used_memory=used, name=row[3], is_holder=is_holder))
    return gpus


def run_process_rows() -> list[list[str]]:
    try:
        return run_csv(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ]
        )
    except RuntimeError:
        return []


def run_csv(command: list[str]) -> list[list[str]]:
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"failed to run {' '.join(command)}: {exc}") from exc
    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            rows.append([part.strip() for part in line.split(",")])
    return rows


def select_gpus(snapshots: list[GpuSnapshot], gpus: tuple[int, ...] | str) -> list[GpuSnapshot]:
    if gpus == "all":
        return snapshots
    wanted = set(gpus)
    return [gpu for gpu in snapshots if gpu.index in wanted]


def status_payload(
    snapshots: list[GpuSnapshot],
    decisions: list[Decision],
    workers: dict[int, WorkerProcess],
    config: Config,
) -> dict[str, object]:
    decision_by_gpu = {decision.gpu_index: decision for decision in decisions}
    return {
        "timestamp": time.time(),
        "config": {
            "gpus": list(config.gpus) if config.gpus != "all" else "all",
            "target_util": config.target_util,
            "risk_util": config.risk_util,
            "mem": config.mem,
            "backend": config.backend,
            "program": config.program,
            "sample_interval": config.sample_interval,
            "process_grace_window": config.process_grace_window,
        },
        "machine": {
            "gpu_count": len(snapshots),
            "average_utilization": round(sum(g.utilization for g in snapshots) / max(1, len(snapshots)), 2),
            "worker_count": sum(1 for worker in workers.values() if worker.is_alive()),
        },
        "gpus": [
            {
                "index": gpu.index,
                "name": gpu.name,
                "utilization": gpu.utilization,
                "temperature": gpu.temperature,
                "memory_total": gpu.memory_total,
                "memory_used": gpu.memory_used,
                "memory_total_human": human_bytes(gpu.memory_total),
                "memory_used_human": human_bytes(gpu.memory_used),
                "processes": [asdict(process) for process in gpu.processes],
                "decision": asdict(decision_by_gpu[gpu.index]),
                "worker": worker_payload(workers.get(gpu.index)),
            }
            for gpu in snapshots
        ],
    }


def worker_payload(worker: WorkerProcess | None) -> dict[str, object] | None:
    if worker is None:
        return None
    return {
        "pid": worker.pid,
        "alive": worker.is_alive(),
        "memory_bytes": worker.memory_bytes,
        "memory_human": human_bytes(worker.memory_bytes),
        "program": worker.program,
        "backend": worker.backend,
        "duty_cycle": worker.duty_cycle,
    }


def format_status(payload: dict[str, object]) -> str:
    lines = [compact_status(payload)]
    for gpu in payload.get("gpus", []):
        if not isinstance(gpu, dict):
            continue
        decision = gpu.get("decision") if isinstance(gpu.get("decision"), dict) else {}
        worker = gpu.get("worker") if isinstance(gpu.get("worker"), dict) else None
        worker_text = "-" if not worker else f"pid={worker.get('pid')} {worker.get('memory_human')}"
        lines.append(
            f"gpu={gpu.get('index')} util={gpu.get('utilization')}% "
            f"mem={gpu.get('memory_used_human')}/{gpu.get('memory_total_human')} "
            f"action={decision.get('action')} reason={decision.get('reason')} worker={worker_text}"
        )
    return "\n".join(lines)


def compact_status(payload: dict[str, object]) -> str:
    machine = payload.get("machine") if isinstance(payload.get("machine"), dict) else {}
    return (
        f"avg={machine.get('average_utilization')}% "
        f"workers={machine.get('worker_count')} "
        f"age={time.time() - float(payload.get('timestamp', time.time())):.0f}s"
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def log(config: Config, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    if stdout_points_to(config.log_file):
        return
    config.state_dir.mkdir(parents=True, exist_ok=True)
    with config.log_file.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def stdout_points_to(path: Path) -> bool:
    try:
        return os.fstat(sys.stdout.fileno()).st_ino == path.stat().st_ino
    except OSError:
        return False


def check_nvidia_smi() -> dict[str, object]:
    try:
        count = len(run_csv(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]))
    except RuntimeError as exc:
        return {"name": "nvidia-smi", "ok": False, "detail": str(exc)}
    return {"name": "nvidia-smi", "ok": count > 0, "detail": f"gpu_count={count}"}


def validate_config(config: Config) -> str | None:
    if config.sample_interval <= 0:
        return "--sample-interval must be positive"
    if config.log_interval < 0:
        return "--log-interval must be non-negative"
    if config.process_grace_window < 0:
        return "--process-grace-window must be non-negative"
    if config.compute_burst_seconds <= 0:
        return "--compute-burst-seconds must be positive"
    try:
        parse_gpus(format_gpus(config.gpus))
        parse_ratio(config.target_util)
        parse_ratio(config.risk_util)
        parse_ratio(config.mem)
        parse_bytes(config.reserve)
        parse_bytes(config.busy_process_mem_threshold)
        parse_bytes(config.assist_mem)
        normalize_backend(config.backend)
    except ValueError as exc:
        return str(exc)
    programs = [part.strip().lower() for part in config.program.split(",")]
    if any(program not in BASE_PROGRAMS for program in programs):
        return f"unsupported --program: {config.program}"
    return None


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        gpus=parse_gpus(args.gpus),
        target_util=args.target_util,
        risk_util=args.risk_util,
        mem=args.mem,
        reserve=args.reserve,
        busy_process_mem_threshold=args.busy_process_mem_threshold,
        assist_mem=args.assist_mem,
        sample_interval=args.sample_interval,
        backend=args.backend,
        program=args.program,
        min_duty_cycle=args.min_duty_cycle,
        max_duty_cycle=args.max_duty_cycle,
        compute_burst_seconds=args.compute_burst_seconds,
        compute_burst_jitter=args.compute_burst_jitter,
        process_grace_window=args.process_grace_window,
        state_dir=Path(args.state_dir).expanduser(),
        log_interval=args.log_interval,
        dry_run=getattr(args, "dry_run", False),
        once=getattr(args, "once", False),
    )


def child_args(config: Config) -> list[str]:
    return [
        "--gpus", format_gpus(config.gpus),
        "--target-util", format_ratio(config.target_util),
        "--risk-util", format_ratio(config.risk_util),
        "--mem", format_ratio(config.mem),
        "--reserve", config.reserve,
        "--busy-process-mem-threshold", config.busy_process_mem_threshold,
        "--assist-mem", config.assist_mem,
        "--sample-interval", str(config.sample_interval),
        "--backend", config.backend,
        "--program", config.program,
        "--min-duty-cycle", str(config.min_duty_cycle),
        "--max-duty-cycle", str(config.max_duty_cycle),
        "--compute-burst-seconds", str(config.compute_burst_seconds),
        "--compute-burst-jitter", str(config.compute_burst_jitter),
        "--process-grace-window", str(config.process_grace_window),
        "--state-dir", str(config.state_dir),
        "--log-interval", str(config.log_interval),
    ]


def parse_gpus(raw: str) -> tuple[int, ...] | str:
    text = str(raw).strip().lower()
    if text == "all":
        return "all"
    values: list[int] = []
    try:
        for part in text.split(","):
            token = part.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", maxsplit=1)
                start = int(start_text.strip())
                end = int(end_text.strip())
                if start > end:
                    raise ValueError
                values.extend(range(start, end + 1))
            else:
                values.append(int(token))
    except ValueError as exc:
        raise ValueError("--gpus must be 'all', comma-separated integers, or ranges like 0-7") from exc
    if not values:
        raise ValueError("--gpus must not be empty")
    if any(value < 0 for value in values):
        raise ValueError("--gpus must not contain negative indices")
    return tuple(dict.fromkeys(values))


def format_gpus(gpus: tuple[int, ...] | str) -> str:
    if gpus == "all":
        return "all"
    if not gpus:
        return ""
    parts: list[str] = []
    start = previous = gpus[0]
    for gpu in gpus[1:]:
        if gpu == previous + 1:
            previous = gpu
            continue
        parts.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = gpu
    parts.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(parts)


def parse_ratio(raw: object) -> float:
    if isinstance(raw, (float, int)):
        value = float(raw)
    else:
        text = str(raw).strip()
        if text.endswith("%"):
            value = float(text[:-1]) / 100
        else:
            value = float(text)
    if 1 < value <= 100:
        value = value / 100
    if not 0 <= value <= 1:
        raise ValueError("ratio must be a float from 0 to 1")
    return value


parse_mem_ratio = parse_ratio


def format_ratio(value: float) -> str:
    return f"{float(value):g}"


def mib(raw: str) -> int:
    if raw in {"", "[N/A]"}:
        return 0
    return int(float(raw)) * 1024**2


def human_bytes(value: int) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.1f}GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.0f}MiB"
    return f"{value}B"


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def process_exists(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return False
    try:
        fields = stat.read_text(encoding="utf-8").split()
    except OSError:
        return False
    return len(fields) < 3 or fields[2] != "Z"


class ProcessKillTarget:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.use_group = False
        try:
            self.use_group = os.getsid(pid) == pid
        except OSError:
            pass

    def terminate(self) -> None:
        self._send(signal.SIGTERM)

    def kill(self) -> None:
        self._send(signal.SIGKILL)

    def _send(self, signum: int) -> None:
        if self.use_group:
            os.killpg(self.pid, signum)
        else:
            os.kill(self.pid, signum)


def is_gpu_holder_process(pid: int) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False
    return "guard" in command and ("gpu-holder" in command or "gpu_holder.cli" in command)


if __name__ == "__main__":
    raise SystemExit(main())
