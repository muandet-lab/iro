"""WILDS iWildCam training/evaluation loops for IRO algorithms."""

from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from iro.data.iwildcam import (
    IWILDCAM_EVAL_SPLITS,
    build_iwildcam_data_bundle,
    build_iwildcam_eval_loader,
    build_iwildcam_train_loader,
    parse_iwildcam_eval_splits,
    split_group_batch_to_minibatches,
)
from iro.utility import algorithms, misc, networks


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _seed_all(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _cfg_get(obj, key: str, default):
    return getattr(obj, key, default)


def _algorithm_hparams(cfg, *, steps: int) -> dict:
    return {
        "algorithm": str(_cfg_get(cfg.iro, "algorithm", "iro")).lower(),
        "penalty_weight": float(_cfg_get(cfg.iro, "penalty_weight", 1000.0)),
        "alpha": float(_cfg_get(cfg.iro, "alpha", 0.8)),
        "alpha_samples": int(_cfg_get(cfg.iro, "alpha_samples", 10)),
        "pareto_num_samples": int(_cfg_get(cfg.iro, "pareto_num_samples", 5)),
        "groupdro_eta": float(_cfg_get(cfg.iro, "groupdro_eta", 1.0)),
        "lr_factor_reduction": float(_cfg_get(cfg.training, "lr_factor_reduction", 1.0)),
        "lr_cos_sched": bool(_cfg_get(cfg.training, "lr_cos_sched", False)),
        "steps": int(steps),
        "batch_size": int(cfg.data.batch_size),
        "lr": float(cfg.training.lr),
        "weight_decay": float(cfg.training.weight_decay),
        "dropout_p": float(_cfg_get(cfg.model, "dropout", 0.0)),
        "erm_pretrain_iters": int(_cfg_get(cfg.training, "erm_pretrain_iters", 0)),
        "eval_freq": int(_cfg_get(cfg.training, "eval_freq", 100)),
    }


def _build_network(cfg, *, n_classes: int):
    net_name = str(_cfg_get(cfg.model, "name", "film_resnet50")).lower()
    hidden_sizes = list(_cfg_get(cfg.model, "hidden_sizes", [256]) or [256])
    pretrained = bool(_cfg_get(cfg.model, "pretrained", False))

    if net_name in {"film_resnet18", "filmedresnet", "filmed_resnet", "resnet18_film", "film_resnet"}:
        return networks.FiLMedResNetClassifier(
            num_classes=n_classes,
            pretrained=pretrained,
            film_hidden_sizes=hidden_sizes,
            backbone_name="resnet18",
        )

    if net_name in {"film_resnet50", "resnet50_film", "film_resnet_50"}:
        return networks.FiLMedResNetClassifier(
            num_classes=n_classes,
            pretrained=pretrained,
            film_hidden_sizes=hidden_sizes,
            backbone_name="resnet50",
        )

    if net_name in {"resnet18", "resnet"}:
        return networks.FiLMedResNetClassifier(
            num_classes=n_classes,
            pretrained=pretrained,
            film_hidden_sizes=hidden_sizes,
            backbone_name="resnet18",
        )

    if net_name in {"resnet50"}:
        return networks.FiLMedResNetClassifier(
            num_classes=n_classes,
            pretrained=pretrained,
            film_hidden_sizes=hidden_sizes,
            backbone_name="resnet50",
        )

    raise NotImplementedError(f"Unsupported iWildCam model '{cfg.model.name}'.")


def _state_dict_copy(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _adjust_learning_rate(optimizer, current_step: int, lr: float, total_steps: int) -> None:
    lr_adj = float(lr)
    lr_adj *= 0.5 * (1.0 + math.cos(math.pi * float(current_step) / float(total_steps)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_adj


def _loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y.view(-1).long())


def _build_algorithm(cfg, *, n_classes: int, steps: int):
    hparams = _algorithm_hparams(cfg, steps=steps)
    net = _build_network(cfg, n_classes=n_classes)
    algorithm_class = algorithms.get_algorithm_class(hparams["algorithm"])
    algorithm = algorithm_class(net, hparams, _loss_fn)
    return algorithm, hparams


def _to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # pragma: no cover - defensive conversion
            return float(value)
    return float(value)


def _sanitize_metric_name(name: str) -> str:
    return str(name).replace("-", "_").replace("/", "_").replace(" ", "_")


def _extract_primary_metrics(raw: dict[str, float]) -> tuple[float, float, float]:
    acc = raw.get("acc_avg", raw.get("acc", 0.0))
    macro_recall = raw.get("recall-macro_all", raw.get("macro_recall", raw.get("recall", 0.0)))
    macro_f1 = raw.get("F1-macro_all", raw.get("macro_f1", raw.get("f1", 0.0)))
    return float(acc), float(macro_recall), float(macro_f1)


def _predict_for_eval(algorithm, x: torch.Tensor, *, use_alpha: bool, eval_alpha: float) -> torch.Tensor:
    if not use_alpha:
        return algorithm.predict(x)
    alpha = torch.tensor(float(eval_alpha), device=x.device, dtype=x.dtype)
    return algorithm.predict(x, alpha)


def _evaluate_iwildcam_split(
    algorithm,
    *,
    dataset,
    split: str,
    loader,
    device: str,
    use_alpha: bool,
    eval_alpha: float,
) -> dict[str, float | str | None | dict[str, float]]:
    algorithm.eval()
    y_pred_parts: list[torch.Tensor] = []
    y_true_parts: list[torch.Tensor] = []
    metadata_parts: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y, metadata in loader:
            x = x.to(device)
            y = y.to(device).view(-1).long()
            metadata = metadata.to(device)
            logits = _predict_for_eval(algorithm, x, use_alpha=use_alpha, eval_alpha=eval_alpha)
            y_pred_parts.append(logits.detach().cpu())
            y_true_parts.append(y.detach().cpu())
            metadata_parts.append(metadata.detach().cpu())

    if not y_pred_parts:
        raise ValueError(f"No batches were produced while evaluating split '{split}'.")

    y_pred = torch.cat(y_pred_parts)
    y_true = torch.cat(y_true_parts)
    metadata = torch.cat(metadata_parts)

    raw_metrics, _results_str = dataset.eval(y_pred, y_true, metadata, prediction_fn=lambda t: t.argmax(dim=1))
    flattened_raw = {str(k): _to_float(v) for k, v in raw_metrics.items()}
    acc, macro_recall, macro_f1 = _extract_primary_metrics(flattened_raw)

    return {
        "split": split,
        "alpha": float(eval_alpha) if use_alpha else None,
        "accuracy": acc,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "wilds_metrics": flattened_raw,
    }


def _evaluate_iwildcam_splits(
    algorithm,
    *,
    dataset,
    eval_loaders: dict[str, Any],
    eval_splits: tuple[str, ...],
    device: str,
    use_alpha: bool,
    eval_alpha: float,
) -> dict[str, dict[str, float | str | None | dict[str, float]]]:
    out: dict[str, dict[str, float | str | None | dict[str, float]]] = {}
    for split in eval_splits:
        out[split] = _evaluate_iwildcam_split(
            algorithm,
            dataset=dataset,
            split=split,
            loader=eval_loaders[split],
            device=device,
            use_alpha=use_alpha,
            eval_alpha=eval_alpha,
        )
    return out


def _selection_split(eval_splits: tuple[str, ...]) -> str:
    if "val" in eval_splits:
        return "val"
    if not eval_splits:
        raise ValueError("At least one iWildCam evaluation split must be configured.")
    return eval_splits[0]


def _prepare_minibatches(batch, *, group_loader: bool, grouper, device: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    x, y, metadata = batch
    x = x.to(device)
    y = y.to(device).view(-1).long()
    metadata = metadata.to(device)

    if not group_loader:
        return [(x, y)]

    return split_group_batch_to_minibatches(x, y, metadata, grouper=grouper)


def _flatten_split_metrics(prefix: str, metrics: dict[str, dict[str, float | str | None | dict[str, float]]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for split, metric in metrics.items():
        out[f"{split}_acc_{prefix}"] = float(metric["accuracy"])
        out[f"{split}_macro_recall_{prefix}"] = float(metric["macro_recall"])
        out[f"{split}_macro_f1_{prefix}"] = float(metric["macro_f1"])
        wilds_metrics = metric.get("wilds_metrics", {})
        if isinstance(wilds_metrics, dict):
            for key, value in wilds_metrics.items():
                out[f"{split}_{_sanitize_metric_name(key)}_{prefix}"] = float(value)
    return out


def _load_checkpoint_state_dict(checkpoint_path: str, device: str):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(
        f"Unsupported checkpoint payload type {type(payload)} in {ckpt_path}. "
        "Expected a state dict or a dict with a 'state_dict' entry."
    )


def train_iwildcam_iro(cfg):
    """Run iWildCam training through WILDS data/subset/loader conventions."""

    _seed_all(int(cfg.training.seed), deterministic=bool(_cfg_get(cfg.training, "deterministic", False)))
    device = _resolve_device(str(cfg.training.device))

    bundle = build_iwildcam_data_bundle(cfg)
    configured_steps = _cfg_get(cfg.training, "steps", None)
    if configured_steps is None:
        steps = max(1, int(cfg.training.epochs) * 100)
    else:
        steps = int(configured_steps)
        if steps <= 0:
            steps = max(1, int(cfg.training.epochs) * 100)

    algorithm, hparams = _build_algorithm(cfg, n_classes=int(bundle.dataset.n_classes), steps=steps)
    algorithm.to(device)

    algo_name = str(hparams["algorithm"]).lower()
    group_loader = algo_name != "erm"
    train_loader = build_iwildcam_train_loader(cfg, bundle, algorithm=algo_name)
    eval_splits = bundle.eval_splits
    eval_loaders = {split: build_iwildcam_eval_loader(cfg, subset) for split, subset in bundle.eval_data.items()}

    use_alpha = algo_name in {"iro", "inftask"}
    eval_alpha = float(max(0.0, min(1.0, float(_cfg_get(cfg.eval, "alpha", 0.8)))))

    history = []
    best_acc = float("-inf")
    best_weights = _state_dict_copy(algorithm)
    eval_freq = max(1, int(hparams["eval_freq"]))
    start_time = time.time()
    step_since_eval = 0

    train_iter = iter(train_loader)
    selection_split = _selection_split(eval_splits)

    for step in range(1, steps + 1):
        if hparams["lr_cos_sched"] and hparams["algorithm"] != "erm":
            if not hasattr(algorithm, "optimizer"):
                raise AttributeError("iWildCam LR cosine schedule requires algorithm.optimizer.")
            if hparams["erm_pretrain_iters"] == 0:
                _adjust_learning_rate(algorithm.optimizer, step, hparams["lr"], steps)
            elif step > hparams["erm_pretrain_iters"] > 0:
                lr_ = hparams["lr"] / hparams["lr_factor_reduction"]
                steps_ = steps - hparams["erm_pretrain_iters"]
                step_ = step - hparams["erm_pretrain_iters"]
                _adjust_learning_rate(algorithm.optimizer, step_, lr_, steps_)

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        minibatches = _prepare_minibatches(
            batch,
            group_loader=group_loader,
            grouper=bundle.grouper,
            device=device,
        )
        step_values = algorithm.update(minibatches)

        if step % eval_freq == 0 or step == steps:
            eval_metrics = _evaluate_iwildcam_splits(
                algorithm,
                dataset=bundle.dataset,
                eval_loaders=eval_loaders,
                eval_splits=eval_splits,
                device=device,
                use_alpha=use_alpha,
                eval_alpha=eval_alpha,
            )
            selection_acc = float(eval_metrics[selection_split]["accuracy"])
            if selection_acc > best_acc:
                best_acc = selection_acc
                best_weights = _state_dict_copy(algorithm)

            results = {
                "step": int(step),
                "avg_step_time": (time.time() - start_time) / max(step - step_since_eval, 1),
            }
            for key, val in step_values.items():
                results[key] = float(val)
            for split, metric in eval_metrics.items():
                results[f"{split}_acc"] = float(metric["accuracy"])
                results[f"{split}_macro_recall"] = float(metric["macro_recall"])
                results[f"{split}_macro_f1"] = float(metric["macro_f1"])
                wilds_metrics = metric.get("wilds_metrics", {})
                if isinstance(wilds_metrics, dict):
                    for key, val in wilds_metrics.items():
                        results[f"{split}_{_sanitize_metric_name(key)}"] = float(val)
            if torch.cuda.is_available():
                results["mem_gb"] = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0 * 1024.0))
            else:
                results["mem_gb"] = 0.0

            result_keys = sorted(results.keys())
            misc.print_row(result_keys, colwidth=12)
            misc.print_row([results[key] for key in result_keys], colwidth=12)

            history.append(
                {
                    "step": int(step),
                    "avg_step_time": float(results["avg_step_time"]),
                    "train": {k: float(v) for k, v in step_values.items()},
                    "eval": copy.deepcopy(eval_metrics),
                    "selection_split": selection_split,
                    "selection_acc": float(selection_acc),
                }
            )
            start_time = time.time()
            step_since_eval = 0

    final_state_dict = _state_dict_copy(algorithm)

    algorithm.load_state_dict(final_state_dict)
    final_metrics = _evaluate_iwildcam_splits(
        algorithm,
        dataset=bundle.dataset,
        eval_loaders=eval_loaders,
        eval_splits=eval_splits,
        device=device,
        use_alpha=use_alpha,
        eval_alpha=eval_alpha,
    )

    algorithm.load_state_dict(best_weights)
    best_metrics = _evaluate_iwildcam_splits(
        algorithm,
        dataset=bundle.dataset,
        eval_loaders=eval_loaders,
        eval_splits=eval_splits,
        device=device,
        use_alpha=use_alpha,
        eval_alpha=eval_alpha,
    )

    algorithm.load_state_dict(best_weights)

    result = {
        "algorithm": algorithm,
        "history": history,
        "test_metrics": [final_metrics[split] for split in eval_splits],
        "best_test_metrics": [best_metrics[split] for split in eval_splits],
        "iwildcam_metrics_final": final_metrics,
        "iwildcam_metrics_best": best_metrics,
        "device": device,
        "dataset": "iwildcam",
        "steps": steps,
        "algorithm_name": algo_name,
        "final_state_dict": final_state_dict,
        "best_state_dict": best_weights,
        "selection_split": selection_split,
    }
    result.update(_flatten_split_metrics("final", final_metrics))
    result.update(_flatten_split_metrics("best", best_metrics))
    return result


def _eval_split_spec(cfg) -> tuple[str, ...]:
    raw = str(_cfg_get(cfg.eval, "split", "all")).strip()
    if raw.lower() in {"all", "val", "test", "id_val", "id_test"} or "," in raw:
        return parse_iwildcam_eval_splits(raw)
    return IWILDCAM_EVAL_SPLITS


def eval_iwildcam_iro(cfg):
    """Evaluate an iWildCam checkpoint using WILDS metrics."""

    checkpoint_path = str(_cfg_get(cfg.eval, "checkpoint_path", "")).strip()
    if not checkpoint_path:
        raise ValueError("eval.checkpoint_path is required for `iro eval`.")

    _seed_all(int(cfg.training.seed), deterministic=bool(_cfg_get(cfg.training, "deterministic", False)))
    device = _resolve_device(str(cfg.training.device))

    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.data.iwildcam_eval_split = ",".join(_eval_split_spec(cfg))
    bundle = build_iwildcam_data_bundle(cfg_copy)
    eval_splits = bundle.eval_splits
    eval_loaders = {split: build_iwildcam_eval_loader(cfg, subset) for split, subset in bundle.eval_data.items()}

    configured_steps = _cfg_get(cfg.training, "steps", None)
    if configured_steps is None:
        steps = max(1, int(cfg.training.epochs) * 100)
    else:
        steps = max(1, int(configured_steps))

    algorithm, hparams = _build_algorithm(cfg, n_classes=int(bundle.dataset.n_classes), steps=steps)
    algorithm.to(device)
    algorithm.load_state_dict(_load_checkpoint_state_dict(checkpoint_path, device), strict=False)

    algo_name = str(hparams["algorithm"]).lower()
    use_alpha = algo_name in {"iro", "inftask"}
    eval_alpha = float(max(0.0, min(1.0, float(_cfg_get(cfg.eval, "alpha", 0.8)))))

    metrics = _evaluate_iwildcam_splits(
        algorithm,
        dataset=bundle.dataset,
        eval_loaders=eval_loaders,
        eval_splits=eval_splits,
        device=device,
        use_alpha=use_alpha,
        eval_alpha=eval_alpha,
    )

    result = {
        "dataset": "iwildcam",
        "device": device,
        "split": ",".join(eval_splits),
        "algorithm_name": algo_name,
        "eval_alpha": eval_alpha if use_alpha else None,
        "metrics": [metrics[split] for split in eval_splits],
        "metrics_by_split": metrics,
        "checkpoint_path": checkpoint_path,
    }
    for split, metric in metrics.items():
        result[f"{split}_acc_eval"] = float(metric["accuracy"])
        result[f"{split}_macro_recall_eval"] = float(metric["macro_recall"])
        result[f"{split}_macro_f1_eval"] = float(metric["macro_f1"])
    return result


__all__ = ["train_iwildcam_iro", "eval_iwildcam_iro"]
