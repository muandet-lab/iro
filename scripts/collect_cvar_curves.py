#!/usr/bin/env python3
"""Collect CVaR-vs-alpha curves in an experiment/model-agnostic workflow.

Supported experiments:
- iwildcam_iro
- cmnist_iro

The script scans JSONL run records, selects one checkpoint per (algorithm, seed),
evaluates group/environment risks on the requested split, computes CVaR across
group risks for a grid of alpha_op values, and writes CSV + figure outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iro.core.config import load_experiment_config
from iro.data.cmnist_dataset import get_cmnist_datasets
from iro.data.iwildcam import build_iwildcam_data_bundle, build_iwildcam_eval_loader
from iro.training.train_cmnist import _build_algorithm as _build_cmnist_algorithm
from iro.training.train_cmnist import _loss_setup as _cmnist_loss_setup
from iro.training.train_cmnist import _parse_env_spec as _cmnist_parse_env_spec
from iro.training.train_iwildcam import _build_algorithm as _build_iwildcam_algorithm
from iro.utility.fast_data_loader import FastDataLoader
from iro.visualization.cvar_curves import plot_cvar_alpha_curves

SUPPORTED_EXPERIMENTS = ("iwildcam_iro", "cmnist_iro")
TARGET_ALGORITHMS = ("erm", "groupdro", "iro")
DEFAULT_ALPHA_GRID = tuple(np.round(np.linspace(0.0, 1.0, 11), 2).tolist())


@dataclass(frozen=True)
class SelectedRun:
    algorithm: str
    seed: int
    run_id: str
    checkpoint_path: Path
    selection_score: float | None
    record: dict[str, Any]


def parse_alpha_grid_spec(spec: str | None) -> tuple[float, ...]:
    if spec is None or not str(spec).strip():
        return DEFAULT_ALPHA_GRID
    out: list[float] = []
    for part in str(spec).split(","):
        text = part.strip()
        if not text:
            continue
        value = float(text)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Alpha values must be in [0,1], got {value}.")
        out.append(round(value, 6))
    if not out:
        return DEFAULT_ALPHA_GRID
    return tuple(sorted(set(out)))


def cvar_from_risks(risks: np.ndarray, alpha_op: float) -> float:
    values = np.asarray(risks, dtype=float)
    if values.size == 0:
        return float("nan")
    q = float(np.quantile(values, float(alpha_op)))
    tail = values[values >= q]
    return float(np.mean(tail)) if tail.size else q


def _iter_json_records(results_root: Path):
    for path in sorted(results_root.rglob("*.jsonl")):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record["_jsonl_path"] = str(path)
                yield record


def _normalize_algorithm(record: dict[str, Any]) -> str:
    return str(record.get("algorithm", "")).strip().lower()


def _record_matches_experiment(record: dict[str, Any], experiment: str) -> bool:
    if str(record.get("status", "")).lower() != "ok":
        return False
    source = str(record.get("source", "")).lower()
    dataset = str(record.get("dataset_name", "")).lower()
    if experiment == "iwildcam_iro":
        return source == "iwildcam" or dataset == "iwildcam"
    if experiment == "cmnist_iro":
        return source == "cmnist" or dataset == "cmnist"
    return False


def _to_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def _selection_score(record: dict[str, Any], metric_key: str) -> float | None:
    if metric_key in record:
        return _to_float_or_none(record.get(metric_key))
    result = record.get("result")
    if isinstance(result, dict) and metric_key in result:
        return _to_float_or_none(result.get(metric_key))
    return None


def _checkpoint_path_for_record(record: dict[str, Any], *, ckpt_kind: str) -> Path:
    run_id = str(record["run_id"])
    output_root = str(record.get("output_root", "")).strip()
    if not output_root:
        cfg = record.get("config")
        if isinstance(cfg, dict):
            training = cfg.get("training")
            if isinstance(training, dict):
                output_root = str(training.get("output_root", "")).strip()
    if not output_root:
        raise ValueError(f"Record {run_id} has no output_root; cannot locate checkpoint.")
    suffix = "best" if ckpt_kind == "best" else "final"
    return Path(output_root) / "ckpts" / f"{run_id}_{suffix}.pkl"


def select_best_run_records(
    records: list[dict[str, Any]],
    *,
    experiment: str,
    algorithms: tuple[str, ...],
    selection_metric: str,
    ckpt_kind: str,
) -> list[SelectedRun]:
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    allowed = {a.lower() for a in algorithms}
    for record in records:
        if not _record_matches_experiment(record, experiment):
            continue
        alg = _normalize_algorithm(record)
        if alg not in allowed:
            continue
        try:
            seed = int(record.get("seed", 0))
        except Exception:
            seed = 0
        by_key.setdefault((alg, seed), []).append(record)

    selected: list[SelectedRun] = []
    for (alg, seed), candidates in sorted(by_key.items()):
        scored: list[tuple[float, str, dict[str, Any]]] = []
        fallback: list[tuple[str, dict[str, Any]]] = []
        for record in candidates:
            run_id = str(record.get("run_id", ""))
            score = _selection_score(record, selection_metric)
            if score is None:
                fallback.append((run_id, record))
            else:
                scored.append((score, run_id, record))
        if scored:
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            chosen_score, _run_id, chosen = scored[0]
        else:
            fallback.sort(key=lambda t: t[0], reverse=True)
            if not fallback:
                continue
            _run_id, chosen = fallback[0]
            chosen_score = None

        ckpt_path = _checkpoint_path_for_record(chosen, ckpt_kind=ckpt_kind)
        selected.append(
            SelectedRun(
                algorithm=alg,
                seed=seed,
                run_id=str(chosen["run_id"]),
                checkpoint_path=ckpt_path,
                selection_score=chosen_score,
                record=chosen,
            )
        )
    return selected


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _checkpoint_state_dict(path: Path, device: str):
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported checkpoint payload in {path}: {type(payload)}")


def _record_model_overrides(record: dict[str, Any], *, experiment: str) -> list[str]:
    cfg = record.get("config")
    if not isinstance(cfg, dict):
        return []
    out: list[str] = []
    model = cfg.get("model")
    if isinstance(model, dict):
        if "name" in model:
            out.append(f"model.name={model['name']}")
        if "pretrained" in model:
            out.append(f"model.pretrained={str(model['pretrained']).lower()}")
        if "hidden_sizes" in model and isinstance(model["hidden_sizes"], list):
            hs = ",".join(str(v) for v in model["hidden_sizes"])
            out.append(f"model.hidden_sizes=[{hs}]")
        if "dropout" in model:
            out.append(f"model.dropout={float(model['dropout'])}")
    if experiment == "iwildcam_iro":
        data = cfg.get("data")
        if isinstance(data, dict):
            if "iwildcam_image_size" in data:
                out.append(f"data.iwildcam_image_size={int(data['iwildcam_image_size'])}")
            if "iwildcam_eval_resize" in data:
                out.append(f"data.iwildcam_eval_resize={int(data['iwildcam_eval_resize'])}")
    return out


def _iwildcam_group_risks_for_alpha(algorithm, loader, *, grouper, device: str, eval_alpha: float | None) -> np.ndarray:
    group_loss_sum: dict[int, float] = {}
    group_count: dict[int, int] = {}

    with torch.no_grad():
        for x, y, metadata in loader:
            x = x.to(device)
            y = y.to(device).view(-1).long()
            if eval_alpha is None:
                logits = algorithm.predict(x)
            else:
                alpha_t = torch.tensor(float(eval_alpha), device=x.device, dtype=x.dtype)
                logits = algorithm.predict(x, alpha_t)
            losses = F.cross_entropy(logits, y, reduction="none").detach().cpu()
            group_ids = grouper.metadata_to_group(metadata.detach().cpu())
            for gid in torch.unique(group_ids):
                g = int(gid.item())
                mask = group_ids == gid
                group_loss_sum[g] = group_loss_sum.get(g, 0.0) + float(losses[mask].sum().item())
                group_count[g] = group_count.get(g, 0) + int(mask.sum().item())

    if not group_loss_sum:
        return np.array([], dtype=float)
    keys = sorted(group_loss_sum.keys())
    return np.asarray([group_loss_sum[k] / max(group_count[k], 1) for k in keys], dtype=float)


def _cmnist_eval_env_setup(cfg, split: str) -> tuple[tuple[float, ...], bool]:
    normalized = split.lower().strip()
    if normalized == "all":
        return tuple(i / 10.0 for i in range(11)), True
    if normalized == "test":
        envs = _cmnist_parse_env_spec(getattr(cfg.data, "cmnist_test_envs", [0.9]), field_name="data.cmnist_test_envs")
        use_test_set = bool(getattr(cfg.data, "cmnist_use_test_set", False))
        return envs, use_test_set
    raise ValueError("CMNIST split must be one of: test | all.")


def _cmnist_env_risks_for_alpha(
    algorithm,
    loaders: list[FastDataLoader],
    *,
    device: str,
    loss_name: str,
    eval_alpha: float | None,
) -> np.ndarray:
    env_risks: list[float] = []
    with torch.no_grad():
        for loader in loaders:
            loss_sum = 0.0
            count = 0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                if eval_alpha is None:
                    logits = algorithm.predict(x)
                else:
                    alpha_t = torch.tensor(float(eval_alpha), device=x.device, dtype=x.dtype)
                    logits = algorithm.predict(x, alpha_t)
                if loss_name == "nll":
                    losses = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none").view(-1)
                else:
                    losses = F.cross_entropy(logits, y.view(-1).long(), reduction="none")
                loss_sum += float(losses.sum().item())
                count += int(losses.numel())
            env_risks.append(loss_sum / max(count, 1))
    return np.asarray(env_risks, dtype=float)


def _evaluate_selected_iwildcam(
    selected_runs: list[SelectedRun],
    *,
    config_root: Path,
    data_root: Path,
    split: str,
    alpha_grid: tuple[float, ...],
    eval_batch_size: int,
    num_workers: int,
    device: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sel in selected_runs:
        if not sel.checkpoint_path.exists():
            print(f"[skip] missing checkpoint: {sel.checkpoint_path}")
            continue
        overrides = [
            f"data.root={data_root}",
            f"data.root_dir={data_root}",
            f"data.data_dir={data_root}",
            "data.download=false",
            f"data.iwildcam_eval_split={split}",
            f"data.batch_size={int(eval_batch_size)}",
            f"data.num_workers={int(num_workers)}",
            f"iro.algorithm={sel.algorithm}",
            f"training.device={device}",
            "training.steps=1",
        ]
        overrides.extend(_record_model_overrides(sel.record, experiment="iwildcam_iro"))
        cfg = load_experiment_config(experiment="iwildcam_iro", config_root=str(config_root), overrides=overrides)

        bundle = build_iwildcam_data_bundle(cfg)
        if split not in bundle.eval_data:
            raise ValueError(f"Split '{split}' not available. Got {bundle.eval_splits}.")
        loader = build_iwildcam_eval_loader(cfg, bundle.eval_data[split])

        algorithm, _hparams = _build_iwildcam_algorithm(cfg, n_classes=int(bundle.dataset.n_classes), steps=1)
        state = _checkpoint_state_dict(sel.checkpoint_path, device=device)
        algorithm.load_state_dict(state, strict=False)
        algorithm.to(device)
        algorithm.eval()

        is_conditional = sel.algorithm in {"iro", "inftask"}
        cached_risks = None
        if not is_conditional:
            cached_risks = _iwildcam_group_risks_for_alpha(
                algorithm, loader, grouper=bundle.grouper, device=device, eval_alpha=None
            )

        for alpha_op in alpha_grid:
            risks = cached_risks
            if is_conditional:
                risks = _iwildcam_group_risks_for_alpha(
                    algorithm, loader, grouper=bundle.grouper, device=device, eval_alpha=float(alpha_op)
                )
            risks = np.asarray(risks if risks is not None else [], dtype=float)
            rows.append(
                {
                    "experiment": "iwildcam_iro",
                    "algorithm": sel.algorithm,
                    "seed": sel.seed,
                    "run_id": sel.run_id,
                    "split": split,
                    "alpha_op": float(alpha_op),
                    "eval_alpha": float(alpha_op) if is_conditional else np.nan,
                    "n_groups": int(risks.size),
                    "risk_mean": float(np.mean(risks)) if risks.size else np.nan,
                    "risk_std": float(np.std(risks, ddof=1)) if risks.size > 1 else 0.0,
                    "cvar": cvar_from_risks(risks, float(alpha_op)),
                    "checkpoint_path": str(sel.checkpoint_path),
                    "selection_score": sel.selection_score,
                }
            )
    return pd.DataFrame(rows)


def _evaluate_selected_cmnist(
    selected_runs: list[SelectedRun],
    *,
    config_root: Path,
    data_root: Path,
    split: str,
    alpha_grid: tuple[float, ...],
    eval_batch_size: int,
    num_workers: int,
    device: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sel in selected_runs:
        if not sel.checkpoint_path.exists():
            print(f"[skip] missing checkpoint: {sel.checkpoint_path}")
            continue
        overrides = [
            f"data.root={data_root}",
            "data.download=false",
            f"training.device={device}",
            "training.steps=1",
            f"iro.algorithm={sel.algorithm}",
            f"data.num_workers={int(num_workers)}",
        ]
        overrides.extend(_record_model_overrides(sel.record, experiment="cmnist_iro"))
        cfg = load_experiment_config(experiment="cmnist_iro", config_root=str(config_root), overrides=overrides)

        eval_env_ps, use_test_set = _cmnist_eval_env_setup(cfg, split=split)
        loss_name = str(getattr(cfg.training, "loss_fn", "nll")).lower()
        n_targets, loss_fn, int_target = _cmnist_loss_setup(loss_name)
        eval_envs = get_cmnist_datasets(
            str(data_root),
            train_envs=(),
            test_envs=eval_env_ps,
            label_noise_rate=float(getattr(cfg.data, "cmnist_label_noise_rate", 0.25)),
            cuda=False,
            int_target=int_target,
            subsample=bool(getattr(cfg.data, "cmnist_subsample", True)),
            use_test_set=use_test_set,
            download=False,
        )
        if not eval_envs:
            raise ValueError("No CMNIST evaluation environments were constructed.")
        eval_loaders = [FastDataLoader(dataset=env, batch_size=eval_batch_size, num_workers=num_workers) for env in eval_envs]
        input_shape = tuple(eval_envs[0].tensors[0].size()[1:])

        algorithm, _hparams = _build_cmnist_algorithm(
            cfg,
            input_shape=input_shape,
            n_targets=n_targets,
            loss_fn=loss_fn,
            steps=1,
        )
        state = _checkpoint_state_dict(sel.checkpoint_path, device=device)
        algorithm.load_state_dict(state, strict=False)
        algorithm.to(device)
        algorithm.eval()

        is_conditional = sel.algorithm in {"iro", "inftask"}
        cached_risks = None
        if not is_conditional:
            cached_risks = _cmnist_env_risks_for_alpha(
                algorithm, eval_loaders, device=device, loss_name=loss_name, eval_alpha=None
            )

        for alpha_op in alpha_grid:
            risks = cached_risks
            if is_conditional:
                risks = _cmnist_env_risks_for_alpha(
                    algorithm,
                    eval_loaders,
                    device=device,
                    loss_name=loss_name,
                    eval_alpha=float(alpha_op),
                )
            risks = np.asarray(risks if risks is not None else [], dtype=float)
            rows.append(
                {
                    "experiment": "cmnist_iro",
                    "algorithm": sel.algorithm,
                    "seed": sel.seed,
                    "run_id": sel.run_id,
                    "split": split,
                    "alpha_op": float(alpha_op),
                    "eval_alpha": float(alpha_op) if is_conditional else np.nan,
                    "n_groups": int(risks.size),
                    "risk_mean": float(np.mean(risks)) if risks.size else np.nan,
                    "risk_std": float(np.std(risks, ddof=1)) if risks.size > 1 else 0.0,
                    "cvar": cvar_from_risks(risks, float(alpha_op)),
                    "checkpoint_path": str(sel.checkpoint_path),
                    "selection_score": sel.selection_score,
                }
            )
    return pd.DataFrame(rows)


def evaluate_selected_runs(
    selected_runs: list[SelectedRun],
    *,
    experiment: str,
    config_root: Path,
    data_root: Path,
    split: str,
    alpha_grid: tuple[float, ...],
    eval_batch_size: int,
    num_workers: int,
    device: str,
) -> pd.DataFrame:
    if experiment == "iwildcam_iro":
        return _evaluate_selected_iwildcam(
            selected_runs,
            config_root=config_root,
            data_root=data_root,
            split=split,
            alpha_grid=alpha_grid,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            device=device,
        )
    if experiment == "cmnist_iro":
        return _evaluate_selected_cmnist(
            selected_runs,
            config_root=config_root,
            data_root=data_root,
            split=split,
            alpha_grid=alpha_grid,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            device=device,
        )
    raise ValueError(f"Unsupported experiment '{experiment}'.")


def _default_selection_metric(experiment: str) -> str:
    if experiment == "iwildcam_iro":
        return "val_acc_best"
    if experiment == "cmnist_iro":
        return "0.9_acc_best"
    return "val_acc_best"


def _validate_split(experiment: str, split: str) -> str:
    value = split.strip().lower()
    if experiment == "iwildcam_iro":
        valid = {"val", "test", "id_val", "id_test"}
    else:
        valid = {"test", "all"}
    if value not in valid:
        raise ValueError(f"Invalid split '{split}' for {experiment}. Expected one of: {sorted(valid)}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CVaR curves for IRO experiments.")
    parser.add_argument("results_root", type=Path, help="Root directory containing JSONL run records.")
    parser.add_argument("--experiment", type=str, default="iwildcam_iro", choices=SUPPORTED_EXPERIMENTS)
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("collected_results/cvar_curves"))
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--split", type=str, default="val", help="Eval split (experiment-dependent).")
    parser.add_argument("--algorithms", type=str, default="erm,groupdro,iro")
    parser.add_argument("--selection-metric", type=str, default="")
    parser.add_argument("--ckpt-kind", type=str, default="best", choices=["best", "final"])
    parser.add_argument("--alpha-grid", type=str, default=",".join(str(x) for x in DEFAULT_ALPHA_GRID))
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--smooth", action="store_true", default=True, help="Use spline smoothing for curves.")
    parser.add_argument("--no-smooth", dest="smooth", action="store_false")
    args = parser.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"results_root not found: {args.results_root}")
    if not args.data_root.exists():
        raise SystemExit(f"data_root not found: {args.data_root}")
    if not args.config_root.exists():
        raise SystemExit(f"config_root not found: {args.config_root}")

    experiment = str(args.experiment)
    split = _validate_split(experiment, str(args.split))
    alpha_grid = parse_alpha_grid_spec(args.alpha_grid)
    algorithms = tuple(a.strip().lower() for a in str(args.algorithms).split(",") if a.strip())
    if not algorithms:
        algorithms = TARGET_ALGORITHMS
    selection_metric = str(args.selection_metric).strip() or _default_selection_metric(experiment)

    records = list(_iter_json_records(args.results_root))
    selected = select_best_run_records(
        records,
        experiment=experiment,
        algorithms=algorithms,
        selection_metric=selection_metric,
        ckpt_kind=str(args.ckpt_kind),
    )
    if not selected:
        raise SystemExit("No checkpoints selected from JSONL records for requested filters.")

    selected_df = pd.DataFrame(
        [
            {
                "experiment": experiment,
                "algorithm": s.algorithm,
                "seed": s.seed,
                "run_id": s.run_id,
                "checkpoint_path": str(s.checkpoint_path),
                "selection_score": s.selection_score,
            }
            for s in selected
        ]
    )
    print("Selected runs:")
    print(selected_df.sort_values(["algorithm", "seed"]).to_string(index=False))

    device = _resolve_device(str(args.device))
    rows_df = evaluate_selected_runs(
        selected,
        experiment=experiment,
        config_root=args.config_root,
        data_root=args.data_root,
        split=split,
        alpha_grid=alpha_grid,
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )
    if rows_df.empty:
        raise SystemExit("No CVaR rows were produced; check checkpoint paths and dataset availability.")

    summary = (
        rows_df.groupby(["algorithm", "alpha_op"], as_index=False)
        .agg(
            cvar_mean=("cvar", "mean"),
            cvar_std=("cvar", "std"),
            risk_mean_mean=("risk_mean", "mean"),
            risk_mean_std=("risk_mean", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["algorithm", "alpha_op"])
    )
    summary[["cvar_std", "risk_mean_std"]] = summary[["cvar_std", "risk_mean_std"]].fillna(0.0)

    output_dir = args.output_dir / experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / f"{experiment}_cvar_rows.csv"
    summary_csv = output_dir / f"{experiment}_cvar_summary.csv"
    selected_csv = output_dir / f"{experiment}_selected_runs.csv"
    fig_png = output_dir / f"{experiment}_cvar_curve.png"
    fig_pdf = output_dir / f"{experiment}_cvar_curve.pdf"

    rows_df.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    selected_df.to_csv(selected_csv, index=False)
    fig = plot_cvar_alpha_curves(
        summary,
        output_png=fig_png,
        output_pdf=fig_pdf,
        split=split,
        smooth=bool(args.smooth),
        title=f"{experiment} CVaR curve ({split} split)",
    )
    fig.clf()

    print(f"Saved rows: {rows_csv}")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved run selection: {selected_csv}")
    print(f"Saved figure: {fig_png}")


if __name__ == "__main__":
    main()

