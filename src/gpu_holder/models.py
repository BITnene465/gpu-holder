from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DecisionAction(str, Enum):
    HOLD = "hold"
    ASSIST = "assist"
    RELEASE = "release"
    STANDBY = "standby"


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    used_memory_bytes: int
    name: str = ""
    is_holder: bool = False


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    name: str
    memory_total_bytes: int
    memory_used_bytes: int
    utilization: int
    temperature_celsius: int | None = None
    processes: tuple[GpuProcess, ...] = ()
    timestamp: float = 0.0

    def non_holder_processes(self) -> tuple[GpuProcess, ...]:
        return tuple(process for process in self.processes if not process.is_holder)


@dataclass(frozen=True)
class HolderDecision:
    gpu_index: int
    action: DecisionAction
    reason: str
    memory_bytes: int = 0
    duty_cycle: float = 0.0
    program: str = "mixed"
    hold_mode: str = "balanced"
    machine_average: float = 0.0
    machine_policy_average: float = 0.0
    history_average: float = 0.0


@dataclass
class GpuHistory:
    window_seconds: float
    samples: list[tuple[float, int]] = field(default_factory=list)

    def add(self, timestamp: float, utilization: int) -> None:
        self.samples.append((float(timestamp), int(utilization)))
        cutoff = float(timestamp) - float(self.window_seconds)
        self.samples = [(ts, util) for ts, util in self.samples if ts >= cutoff]

    def full_window_below(self, *, now: float, threshold: int) -> bool:
        if not self.samples:
            return False
        oldest = self.samples[0][0]
        if float(now) - oldest < float(self.window_seconds):
            return False
        return all(util < int(threshold) for _, util in self.samples)

    def average(self) -> float:
        if not self.samples:
            return 0.0
        return sum(util for _, util in self.samples) / len(self.samples)

    def sample_count(self) -> int:
        return len(self.samples)

    def coverage_seconds(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return max(0.0, self.samples[-1][0] - self.samples[0][0])


@dataclass
class MachineHistory:
    window_seconds: float
    samples: list[tuple[float, float]] = field(default_factory=list)

    def add(self, timestamp: float, utilization: float) -> None:
        self.samples.append((float(timestamp), float(utilization)))
        cutoff = float(timestamp) - float(self.window_seconds)
        self.samples = [(ts, util) for ts, util in self.samples if ts >= cutoff]

    def average(self) -> float:
        if not self.samples:
            return 0.0
        return sum(util for _, util in self.samples) / len(self.samples)

    def sample_count(self) -> int:
        return len(self.samples)

    def coverage_seconds(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return max(0.0, self.samples[-1][0] - self.samples[0][0])
