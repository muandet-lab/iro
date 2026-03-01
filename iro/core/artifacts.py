"""Artifact runtime utilities for train runs."""

from __future__ import annotations

import json
import sys
import traceback
from contextlib import AbstractContextManager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Any

import torch

from iro.utility.misc import Tee


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def _remove_training_seed(cfg_payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg_payload)
    out.pop("master_seed", None)
    training = dict(out.get("training", {}))
    training.pop("seed", None)
    out["training"] = training
    return out


def _compute_args_id(cfg_payload: dict[str, Any]) -> str:
    normalized = _remove_training_seed(cfg_payload)
    blob = json.dumps(_to_jsonable(normalized), sort_keys=True, separators=(",", ":"))
    return md5(blob.encode("utf-8")).hexdigest()


class ArtifactContext(AbstractContextManager["ArtifactContext"]):
    """Context manager for logs/results/checkpoints for one run."""

    def __init__(self, cfg, *, experiment: str):
        self.cfg = cfg
        self.experiment = experiment
        self.enabled = bool(getattr(cfg.training, "write_artifacts", True))
        self.capture_logs = bool(getattr(cfg.training, "capture_logs", True))
        self.save_ckpts = bool(getattr(cfg.training, "save_ckpts", True))

        self.output_root = Path(str(getattr(cfg.training, "output_root", "./iro_exp")))
        self.exp_name = str(getattr(cfg.training, "exp_name", "reproduce"))
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        self.logs_dir = self.output_root / "logs" / self.exp_name
        self.results_dir = self.output_root / "results" / self.exp_name
        self.ckpt_dir = self.output_root / "ckpts"

        self.out_file = self.logs_dir / "out.txt"
        self.err_file = self.logs_dir / "err.txt"
        self.results_file = self.results_dir / f"{self.run_id}.jsonl"
        self.ckpt_final_file = self.ckpt_dir / f"{self.run_id}_final.pkl"
        self.ckpt_best_file = self.ckpt_dir / f"{self.run_id}_best.pkl"

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._tee_out: Tee | None = None
        self._tee_err: Tee | None = None

    def __enter__(self) -> "ArtifactContext":
        if not self.enabled:
            return self
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.capture_logs:
            self._tee_out = Tee(str(self.out_file), mode="a", stream=self._orig_stdout)
            self._tee_err = Tee(str(self.err_file), mode="a", stream=self._orig_stderr)
            sys.stdout = self._tee_out
            sys.stderr = self._tee_err
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.capture_logs:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
            if self._tee_out is not None:
                self._tee_out.close()
            if self._tee_err is not None:
                self._tee_err.close()
        return False

    def build_record(self, *, status: str, result: dict[str, Any] | None, error: BaseException | None) -> dict[str, Any]:
        cfg_payload = _to_jsonable(asdict(self.cfg))
        record: dict[str, Any] = {
            "status": status,
            "run_id": self.run_id,
            "experiment": self.experiment,
            "source": str(self.cfg.data.source),
            "dataset_name": str(self.cfg.data.dataset_name),
            "exp_name": self.exp_name,
            "output_root": str(self.output_root),
            "args_id": _compute_args_id(cfg_payload),
            "algorithm": str(getattr(self.cfg.iro, "algorithm", getattr(self.cfg.iro, "risk_measure", "unknown"))),
            "seed": int(self.cfg.training.seed),
            "args": _remove_training_seed(cfg_payload),
            "config": cfg_payload,
            "result": _to_jsonable(result or {}),
        }
        if error is not None:
            record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        # Surface flat CMNIST metrics for legacy compatibility when present.
        if result:
            for key, value in result.items():
                if isinstance(key, str) and (
                    key.endswith("_acc_final")
                    or key.endswith("_loss_final")
                    or key.endswith("_acc_best")
                    or key.endswith("_loss_best")
                ):
                    record[key] = _to_jsonable(value)
            if "algorithm_name" in result:
                record["algorithm"] = str(result["algorithm_name"])
        return record

    def write_jsonl_record(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with open(self.results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(_to_jsonable(record), sort_keys=True) + "\n")

    def write_success(self, result: dict[str, Any]) -> None:
        self.write_jsonl_record(self.build_record(status="ok", result=result, error=None))

    def write_failure(self, exc: BaseException) -> None:
        traceback.print_exc()
        self.write_jsonl_record(self.build_record(status="failed", result=None, error=exc))

    def save_checkpoints(self, result: dict[str, Any]) -> dict[str, str]:
        if not (self.enabled and self.save_ckpts):
            return {}

        final_payload: dict[str, Any] | None = None
        best_payload: dict[str, Any] | None = None

        if "final_checkpoint" in result and isinstance(result["final_checkpoint"], dict):
            final_payload = result["final_checkpoint"]
        if "best_checkpoint" in result and isinstance(result["best_checkpoint"], dict):
            best_payload = result["best_checkpoint"]

        if "final_state_dict" in result:
            final_payload = {"state_dict": result["final_state_dict"], "experiment": self.experiment}
        if "best_state_dict" in result:
            best_payload = {"state_dict": result["best_state_dict"], "experiment": self.experiment}

        if final_payload is None and "model" in result and hasattr(result["model"], "state_dict"):
            final_payload = {"state_dict": result["model"].state_dict(), "experiment": self.experiment}

        if final_payload is None and {"encoder", "head"}.issubset(result.keys()):
            encoder = result["encoder"]
            head = result["head"]
            if hasattr(encoder, "state_dict") and hasattr(head, "state_dict"):
                final_payload = {
                    "encoder_state_dict": encoder.state_dict(),
                    "head_state_dict": head.state_dict(),
                    "experiment": self.experiment,
                }

        if final_payload is None and "algorithm" in result and hasattr(result["algorithm"], "state_dict"):
            final_payload = {"state_dict": result["algorithm"].state_dict(), "experiment": self.experiment}

        if final_payload is None:
            return {}
        if best_payload is None:
            best_payload = final_payload

        torch.save(final_payload, self.ckpt_final_file)
        torch.save(best_payload, self.ckpt_best_file)
        return {
            "ckpt_final": str(self.ckpt_final_file),
            "ckpt_best": str(self.ckpt_best_file),
        }

    def as_metadata(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "results_file": str(self.results_file),
            "out_file": str(self.out_file),
            "err_file": str(self.err_file),
        }
