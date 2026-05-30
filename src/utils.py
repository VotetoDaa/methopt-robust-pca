from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import psutil
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested_device: str | torch.device = "auto") -> torch.device:
    requested = str(requested_device).lower()
    if requested in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device(requested)
    raise ValueError("Device must be 'auto', 'cpu', 'cuda', or 'cuda:N'.")


@contextmanager
def measure_runtime(device: str | torch.device = "auto") -> Iterator[dict[str, float]]:
    resolved_device = get_device(device)
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    start = time.perf_counter()
    result: dict[str, float] = {}
    try:
        yield result
    finally:
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        result["runtime_sec"] = time.perf_counter() - start


def get_memory_usage() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024**2)


def reset_peak_memory(device: str | torch.device = "auto") -> None:
    resolved_device = get_device(device)
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(resolved_device)


def get_peak_memory_mb(device: str | torch.device = "auto") -> float:
    resolved_device = get_device(device)
    if resolved_device.type == "cuda":
        return torch.cuda.max_memory_allocated(resolved_device) / (1024**2)
    return get_memory_usage()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=_json_default)
