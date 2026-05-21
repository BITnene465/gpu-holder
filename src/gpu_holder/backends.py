from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_BACKEND = "torch"
SUPPORTED_BACKENDS = (DEFAULT_BACKEND,)


@dataclass(frozen=True)
class BackendCheck:
    name: str
    ok: bool
    detail: str

    def as_payload(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def normalize_backend(raw: str) -> str:
    backend = str(raw).strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        supported = ", ".join(SUPPORTED_BACKENDS)
        raise ValueError(f"unsupported backend: {raw!r}; supported backends: {supported}")
    return backend


def check_backend(backend: str = DEFAULT_BACKEND) -> BackendCheck:
    normalized = normalize_backend(backend)
    if normalized == "torch":
        return check_torch_cuda()
    raise AssertionError(f"unhandled backend: {normalized}")


def check_torch_cuda() -> BackendCheck:
    try:
        import torch
    except Exception as exc:
        return BackendCheck(name="torch_cuda", ok=False, detail=f"{type(exc).__name__}: {exc}")
    ok = bool(torch.cuda.is_available())
    detail = (
        f"torch={torch.__version__} cuda_available={ok} "
        f"device_count={torch.cuda.device_count() if ok else 0}"
    )
    return BackendCheck(name="torch_cuda", ok=ok, detail=detail)


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "The torch backend requires a CUDA-enabled PyTorch build. "
            "Install one for this machine or use: pip install 'gpu-holder[torch]'"
        ) from exc
    return torch

