"""Shared output helpers for example scripts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def create_run_dir(output_root: str, experiment_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / experiment_name / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)
