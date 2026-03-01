"""CMNIST training loop for IRO experiments.

This module exposes `train_cmnist_iro(cfg)` so the package dispatcher and
examples can call CMNIST training directly from YAML-composed configuration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import random
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from iro.data.cmnist_dataset import get_cmnist_datasets
from iro.utility import algorithms, misc, networks
from iro.utility.fast_data_loader import FastDataLoader


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


def _parse_env_spec(value: Iterable[float] | str, *, field_name: str) -> tuple[float, ...]:
    if isinstance(value, str):
        raw = value.strip()
        lowered = raw.lower()
        if lowered == "default":
            parsed = (0.1, 0.2)
        elif lowered == "gray":
            parsed = (0.5, 0.5)
        else:
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            if not parts:
                raise ValueError(f"{field_name} must contain at least one environment value.")
            parsed = tuple(float(part) for part in parts)
    else:
        parsed = tuple(float(env) for env in value)

    if not parsed:
        raise ValueError(f"{field_name} must contain at least one environment value.")
    return parsed


def _envs_to_legacy_string(value: Iterable[float] | str, *, spaced: bool) -> str:
    if isinstance(value, str):
        return value.strip()

    sep = ", " if spaced else ","
    return sep.join(f"{float(env):g}" for env in value)


def _loss_setup(loss_name: str):
    normalized = str(loss_name).strip().lower()
    if normalized == "nll":
        return 1, F.binary_cross_entropy_with_logits, False
    if normalized == "cross_ent":
        return 2, F.cross_entropy, True
    raise ValueError("training.loss_fn must be one of: 'nll', 'cross_ent'.")


def _algorithm_hparams(cfg, *, steps: int) -> dict:
    return {
        "algorithm": str(_cfg_get(cfg.iro, "algorithm", "iro")).lower(),
        "penalty_weight": float(_cfg_get(cfg.iro, "penalty_weight", 1000.0)),
        "alpha": float(_cfg_get(cfg.iro, "alpha", 0.8)),
        "groupdro_eta": float(_cfg_get(cfg.iro, "groupdro_eta", 1.0)),
        "lr_factor_reduction": float(_cfg_get(cfg.training, "lr_factor_reduction", 1.0)),
        "lr_cos_sched": bool(_cfg_get(cfg.training, "lr_cos_sched", False)),
        "steps": int(steps),
        "batch_size": int(cfg.data.batch_size),
        "lr": float(cfg.training.lr),
        "weight_decay": float(cfg.training.weight_decay),
        "dropout_p": float(_cfg_get(cfg.model, "dropout", 0.2)),
        "erm_pretrain_iters": int(_cfg_get(cfg.training, "erm_pretrain_iters", 0)),
        "eval_freq": int(_cfg_get(cfg.training, "eval_freq", 50)),
    }


def _build_network(cfg, *, input_shape: tuple[int, ...], n_targets: int):
    net_name = str(_cfg_get(cfg.model, "name", "filmedmlp")).lower()
    hidden_dim = int((_cfg_get(cfg.model, "hidden_sizes", [390]) or [390])[0])
    dropout = float(_cfg_get(cfg.model, "dropout", 0.2))
    input_dim = int(np.prod(input_shape))

    if net_name in {"filmedmlp", "film_mlp", "film_cnn", "film"}:
        return networks.FiLMedMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_classes=n_targets,
            dropout=dropout,
            film_dim=1,
        )
    if net_name == "mlp":
        return networks.MLP(input_dim=input_dim, hidden_dim=hidden_dim, n_classes=n_targets, dropout=dropout)
    if net_name == "cnn":
        return networks.CNN(input_shape=input_shape, n_classes=n_targets)
    raise NotImplementedError(f"Unsupported CMNIST model '{cfg.model.name}'.")


def _evaluate_test_envs(algorithm, loaders, env_names, loss_fn, device: str, use_alpha: bool, h_alphas: list[float]):
    metrics = []
    for i, (env_name, env_loader) in enumerate(zip(env_names, loaders)):
        alpha = float(h_alphas[i]) if use_alpha and i < len(h_alphas) else None
        acc = misc.accuracy(algorithm, env_loader, device, alpha=alpha)
        loss = misc.loss(algorithm, env_loader, loss_fn, device, alpha=alpha)
        metrics.append(
            {
                "env": env_name,
                "alpha": alpha,
                "acc": float(acc),
                "loss": float(loss),
            }
        )
    return metrics


def _env_name(value: float) -> str:
    return f"{float(value):g}"


def _state_dict_copy(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _adjust_learning_rate(optimizer, current_step: int, lr: float, total_steps: int) -> None:
    lr_adj = float(lr)
    lr_adj *= 0.5 * (1.0 + math.cos(math.pi * float(current_step) / float(total_steps)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_adj


def _legacy_network_name(model_name: str) -> str:
    lowered = model_name.lower()
    if lowered in {"filmedmlp", "film_mlp", "film_cnn", "film"}:
        return "FiLMedMLP"
    if lowered == "mlp":
        return "MLP"
    if lowered == "cnn":
        return "CNN"
    return model_name


def _legacy_namespace(cfg, *, steps: int, train_envs_raw: Iterable[float] | str, test_envs_raw: Iterable[float] | str):
    hidden_sizes = _cfg_get(cfg.model, "hidden_sizes", [390]) or [390]
    ns = argparse.Namespace()
    ns.train_envs = _envs_to_legacy_string(train_envs_raw, spaced=True)
    ns.test_envs = _envs_to_legacy_string(test_envs_raw, spaced=False)
    ns.test_env_ms = str(_cfg_get(cfg.data, "cmnist_test_env_ms", 0.9))
    ns.full_resolution = not bool(_cfg_get(cfg.data, "cmnist_subsample", True))
    ns.network = _legacy_network_name(str(_cfg_get(cfg.model, "name", "filmedmlp")))
    ns.mlp_hidden_dim = int(hidden_sizes[0])
    ns.algorithm = str(_cfg_get(cfg.iro, "algorithm", "iro")).lower()
    ns.penalty_weight = float(_cfg_get(cfg.iro, "penalty_weight", 1000.0))
    ns.alpha = float(_cfg_get(cfg.iro, "alpha", 0.8))
    ns.groupdro_eta = float(_cfg_get(cfg.iro, "groupdro_eta", 1.0))
    ns.steps = int(steps)
    ns.batch_size = int(cfg.data.batch_size)
    ns.loss_fn = str(_cfg_get(cfg.training, "loss_fn", "nll")).lower()
    ns.lr = float(cfg.training.lr)
    ns.lr_factor_reduction = float(_cfg_get(cfg.training, "lr_factor_reduction", 1.0))
    ns.lr_cos_sched = bool(_cfg_get(cfg.training, "lr_cos_sched", False))
    ns.weight_decay = float(cfg.training.weight_decay)
    ns.dropout_p = float(_cfg_get(cfg.model, "dropout", 0.2))
    ns.erm_pretrain_iters = int(_cfg_get(cfg.training, "erm_pretrain_iters", 0))
    ns.eval_freq = int(_cfg_get(cfg.training, "eval_freq", 50))
    ns.data_dir = str(cfg.data.root)
    ns.output_dir = str(_cfg_get(cfg.training, "output_root", "./iro_exp"))
    ns.exp_name = str(_cfg_get(cfg.training, "exp_name", "reproduce"))
    ns.save_ckpts = bool(_cfg_get(cfg.training, "save_ckpts", True))
    ns.seed = int(cfg.training.seed)
    ns.deterministic = bool(_cfg_get(cfg.training, "deterministic", False))
    ns.n_workers = int(cfg.data.num_workers)
    ns.other_arg = "default"
    return ns


def _legacy_erm_sidecar_path(
    cfg,
    *,
    steps: int,
    train_envs_raw: Iterable[float] | str,
    test_envs_raw: Iterable[float] | str,
) -> Path | None:
    if int(_cfg_get(cfg.training, "erm_pretrain_iters", 0)) <= 0:
        return None

    args_ns = _legacy_namespace(cfg, steps=steps, train_envs_raw=train_envs_raw, test_envs_raw=test_envs_raw)
    erm_args = vars(copy.deepcopy(args_ns))
    for key in (
        "algorithm",
        "penalty_weight",
        "alpha",
        "groupdro_eta",
        "lr_factor_reduction",
        "lr_cos_sched",
        "steps",
        "save_ckpts",
    ):
        erm_args.pop(key, None)

    ckpt_name = hashlib.md5(str(erm_args).encode("utf-8")).hexdigest()
    output_root = Path(str(_cfg_get(cfg.training, "output_root", "./iro_exp")))
    return output_root / "ckpts" / f"{ckpt_name}.pkl"


def _build_algorithm(cfg, *, input_shape: tuple[int, ...], n_targets: int, loss_fn, steps: int):
    hparams = _algorithm_hparams(cfg, steps=steps)
    net = _build_network(cfg, input_shape=input_shape, n_targets=n_targets)
    algorithm_class = algorithms.get_algorithm_class(hparams["algorithm"])
    algorithm = algorithm_class(net, hparams, loss_fn)
    return algorithm, hparams


def train_cmnist_iro(cfg):
    """Run the CMNIST training loop from config."""

    _seed_all(int(cfg.training.seed), deterministic=bool(_cfg_get(cfg.training, "deterministic", False)))
    device = _resolve_device(str(cfg.training.device))
    use_cuda = device.startswith("cuda")

    loss_name = str(_cfg_get(cfg.training, "loss_fn", "nll")).lower()
    n_targets, loss_fn, int_target = _loss_setup(loss_name)

    train_envs_raw = _cfg_get(cfg.data, "cmnist_train_envs", [0.1, 0.2])
    test_envs_raw = _cfg_get(cfg.data, "cmnist_test_envs", [0.9])
    train_env_ps = _parse_env_spec(train_envs_raw, field_name="data.cmnist_train_envs")
    test_env_ps = _parse_env_spec(test_envs_raw, field_name="data.cmnist_test_envs")
    test_env_names = [_env_name(p) for p in test_env_ps]
    test_env_ms = _env_name(float(cfg.data.cmnist_test_env_ms))

    envs = get_cmnist_datasets(
        cfg.data.root,
        train_envs=train_env_ps,
        test_envs=test_env_ps,
        label_noise_rate=float(cfg.data.cmnist_label_noise_rate),
        cuda=use_cuda,
        int_target=int_target,
        subsample=bool(cfg.data.cmnist_subsample),
        use_test_set=bool(cfg.data.cmnist_use_test_set),
        download=bool(cfg.data.download),
    )
    train_envs, test_envs = envs[: len(train_env_ps)], envs[len(train_env_ps) :]
    if not train_envs:
        raise ValueError("CMNIST training requires at least one training environment.")

    input_shape = tuple(train_envs[0].tensors[0].size()[1:])
    n_train_samples = int(train_envs[0].tensors[0].size()[0])
    configured_steps = _cfg_get(cfg.training, "steps", None)
    if configured_steps is None:
        steps = max(1, int(cfg.training.epochs) * 100)
    else:
        steps = int(configured_steps)
        if steps <= 0:
            steps = max(1, int(cfg.training.epochs) * 100)

    train_loaders = [
        FastDataLoader(dataset=env, batch_size=int(cfg.data.batch_size), num_workers=int(cfg.data.num_workers))
        for env in train_envs
    ]
    test_loaders = [
        FastDataLoader(dataset=env, batch_size=int(cfg.data.batch_size), num_workers=int(cfg.data.num_workers))
        for env in test_envs
    ]
    train_minibatches_iterator = zip(*train_loaders)

    algorithm, hparams = _build_algorithm(
        cfg,
        input_shape=input_shape,
        n_targets=n_targets,
        loss_fn=loss_fn,
        steps=steps,
    )
    algorithm.to(device)

    start_step = 1
    legacy_erm_ckpt_path = _legacy_erm_sidecar_path(
        cfg,
        steps=steps,
        train_envs_raw=train_envs_raw,
        test_envs_raw=test_envs_raw,
    )
    if legacy_erm_ckpt_path is not None and legacy_erm_ckpt_path.exists():
        loaded = torch.load(legacy_erm_ckpt_path, map_location=device)
        if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            loaded = loaded["state_dict"]
        algorithm.load_state_dict(loaded, strict=False)
        start_step = int(hparams["erm_pretrain_iters"]) + 1
        print(f"ERM-pretrained model loaded: {legacy_erm_ckpt_path.stem}.")

    # For IRO/InfTask evaluation on env shifts.
    h_alphas_train = [0.0 if p <= 0.2 else 1.0 for p in test_env_ps]
    use_alpha = hparams["algorithm"] in {"iro", "inftask"}

    history = []
    best_acc = 0.0
    best_weights = _state_dict_copy(algorithm)
    eval_freq = max(1, int(hparams["eval_freq"]))
    steps_per_epoch = n_train_samples / max(int(cfg.data.batch_size), 1)
    start_time = time.time()
    step_since_eval = 0

    for step in range(start_step, steps + 1):
        if hparams["lr_cos_sched"] and hparams["algorithm"] != "erm":
            if not hasattr(algorithm, "optimizer"):
                raise AttributeError("CMNIST LR cosine schedule requires algorithm.optimizer.")
            if hparams["erm_pretrain_iters"] == 0:
                _adjust_learning_rate(algorithm.optimizer, step, hparams["lr"], steps)
            elif step > hparams["erm_pretrain_iters"] > 0:
                lr_ = hparams["lr"] / hparams["lr_factor_reduction"]
                steps_ = steps - hparams["erm_pretrain_iters"]
                step_ = step - hparams["erm_pretrain_iters"]
                _adjust_learning_rate(algorithm.optimizer, step_, lr_, steps_)

        try:
            minibatch_train = next(train_minibatches_iterator)
        except StopIteration:
            train_minibatches_iterator = zip(*train_loaders)
            minibatch_train = next(train_minibatches_iterator)

        step_values = algorithm.update(minibatch_train)

        if step % eval_freq == 0 or step == steps:
            eval_metrics = _evaluate_test_envs(
                algorithm,
                test_loaders,
                test_env_names,
                loss_fn,
                device,
                use_alpha=use_alpha,
                h_alphas=h_alphas_train,
            )
            available_envs = [str(m["env"]) for m in eval_metrics]
            acc_map = {str(m["env"]): float(m["acc"]) for m in eval_metrics}
            if test_env_ms not in acc_map:
                raise ValueError(
                    f"Selection env '{test_env_ms}' not found in evaluated test envs: {available_envs}."
                )
            selection_acc = float(acc_map[test_env_ms])
            if selection_acc > best_acc:
                best_acc = selection_acc
                best_weights = _state_dict_copy(algorithm)

            results = {
                "step": int(step),
                "epoch": float(step / steps_per_epoch),
                "avg_step_time": (time.time() - start_time) / max(step - step_since_eval, 1),
            }
            for key, val in step_values.items():
                results[key] = float(val)
            for metric in eval_metrics:
                env = str(metric["env"])
                results[f"{env}_acc"] = float(metric["acc"])
                results[f"{env}_loss"] = float(metric["loss"])
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
                    "eval": eval_metrics,
                    "selection_env": test_env_ms,
                    "selection_acc": selection_acc,
                }
            )
            start_time = time.time()
            step_since_eval = 0

        if step == hparams["erm_pretrain_iters"] > 0 and bool(_cfg_get(cfg.training, "save_ckpts", True)):
            if legacy_erm_ckpt_path is not None:
                legacy_erm_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(algorithm.state_dict(), legacy_erm_ckpt_path)
                print("Saved ERM-pretrained model.")

    final_state_dict = _state_dict_copy(algorithm)

    # Final metrics with both final and best checkpoint.
    algorithm.load_state_dict(final_state_dict)
    final_metrics = _evaluate_test_envs(
        algorithm,
        test_loaders,
        test_env_names,
        loss_fn,
        device,
        use_alpha=use_alpha,
        h_alphas=h_alphas_train,
    )

    algorithm.load_state_dict(best_weights)
    best_metrics = _evaluate_test_envs(
        algorithm,
        test_loaders,
        test_env_names,
        loss_fn,
        device,
        use_alpha=use_alpha,
        h_alphas=h_alphas_train,
    )

    # Legacy final sweep over all color-shift environments for parity with train_sandbox.py.
    all_ps = [i / 10.0 for i in range(11)]
    all_env_names = [_env_name(p) for p in all_ps]
    all_envs = get_cmnist_datasets(
        cfg.data.root,
        train_envs=(),
        test_envs=tuple(all_ps),
        label_noise_rate=float(cfg.data.cmnist_label_noise_rate),
        cuda=use_cuda,
        int_target=int_target,
        subsample=bool(cfg.data.cmnist_subsample),
        use_test_set=True,
        download=bool(cfg.data.download),
    )
    all_loaders = [
        FastDataLoader(dataset=env, batch_size=5000, num_workers=int(cfg.data.num_workers))
        for env in all_envs
    ]
    h_alphas_test = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    legacy_metrics: dict[str, float] = {}
    for ms_name, weights in (("final", final_state_dict), ("best", best_weights)):
        algorithm.load_state_dict(weights)
        for i, (env_name, env_loader) in enumerate(zip(all_env_names, all_loaders)):
            if use_alpha:
                eval_alpha = float(h_alphas_test[i])
                acc = misc.accuracy(algorithm, env_loader, device, alpha=eval_alpha)
                loss = misc.loss(algorithm, env_loader, loss_fn, device, alpha=eval_alpha)
            else:
                acc = misc.accuracy(algorithm, env_loader, device)
                loss = misc.loss(algorithm, env_loader, loss_fn, device)
            legacy_metrics[f"{env_name}_acc_{ms_name}"] = float(acc)
            legacy_metrics[f"{env_name}_loss_{ms_name}"] = float(loss)

        misc.cvar(algorithm, all_loaders, loss_fn, device, all_ps, invariant=not use_alpha)
        print(f"\n{ms_name} accuracies:")
        acc_keys = [k for k in sorted(legacy_metrics.keys()) if k.endswith(f"_acc_{ms_name}")]
        misc.print_row([k.replace(f"_acc_{ms_name}", "") for k in acc_keys], colwidth=5)
        misc.print_row([round(float(legacy_metrics[k]), 3) for k in acc_keys], colwidth=5)

    algorithm.load_state_dict(best_weights)

    result = {
        "algorithm": algorithm,
        "history": history,
        "test_metrics": final_metrics,
        "best_test_metrics": best_metrics,
        "device": device,
        "dataset": "cmnist",
        "steps": steps,
        "algorithm_name": hparams["algorithm"],
        "final_state_dict": final_state_dict,
        "best_state_dict": best_weights,
        "selection_env": test_env_ms,
        **legacy_metrics,
    }
    if legacy_erm_ckpt_path is not None:
        result["legacy_erm_ckpt_path"] = str(legacy_erm_ckpt_path)
    return result


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


def _eval_env_setup(cfg, *, split: str) -> tuple[tuple[float, ...], bool]:
    normalized = split.lower()
    if normalized == "test":
        return _parse_env_spec(_cfg_get(cfg.data, "cmnist_test_envs", [0.9]), field_name="data.cmnist_test_envs"), bool(
            _cfg_get(cfg.data, "cmnist_use_test_set", False)
        )
    if normalized == "all":
        return tuple(i / 10.0 for i in range(11)), True
    raise ValueError("eval.split must be one of: test | all.")


def eval_cmnist_iro(cfg):
    """Evaluate a CMNIST checkpoint at fixed test-time alpha."""

    checkpoint_path = str(_cfg_get(cfg.eval, "checkpoint_path", "")).strip()
    if not checkpoint_path:
        raise ValueError("eval.checkpoint_path is required for `iro eval`.")

    _seed_all(int(cfg.training.seed), deterministic=bool(_cfg_get(cfg.training, "deterministic", False)))
    device = _resolve_device(str(cfg.training.device))
    use_cuda = device.startswith("cuda")

    split = str(_cfg_get(cfg.eval, "split", "test")).lower()
    eval_env_ps, use_test_set = _eval_env_setup(cfg, split=split)
    eval_env_names = [_env_name(p) for p in eval_env_ps]

    loss_name = str(_cfg_get(cfg.training, "loss_fn", "nll")).lower()
    n_targets, loss_fn, int_target = _loss_setup(loss_name)
    eval_batch_size = int(_cfg_get(cfg.eval, "batch_size", None) or cfg.data.batch_size)

    eval_envs = get_cmnist_datasets(
        cfg.data.root,
        train_envs=(),
        test_envs=eval_env_ps,
        label_noise_rate=float(cfg.data.cmnist_label_noise_rate),
        cuda=use_cuda,
        int_target=int_target,
        subsample=bool(cfg.data.cmnist_subsample),
        use_test_set=use_test_set,
        download=False,
    )
    if not eval_envs:
        raise ValueError("No evaluation environments were constructed for CMNIST.")

    input_shape = tuple(eval_envs[0].tensors[0].size()[1:])
    steps = int(_cfg_get(cfg.training, "steps", None) or max(1, int(cfg.training.epochs) * 100))

    algorithm, hparams = _build_algorithm(
        cfg,
        input_shape=input_shape,
        n_targets=n_targets,
        loss_fn=loss_fn,
        steps=steps,
    )
    algorithm.to(device)
    algorithm.load_state_dict(_load_checkpoint_state_dict(checkpoint_path, device), strict=False)

    eval_loaders = [
        FastDataLoader(dataset=env, batch_size=eval_batch_size, num_workers=int(cfg.data.num_workers)) for env in eval_envs
    ]
    use_alpha = hparams["algorithm"] in {"iro", "inftask"}
    eval_alpha = float(max(0.0, min(1.0, float(_cfg_get(cfg.eval, "alpha", 0.8)))))
    h_alphas = [eval_alpha for _ in eval_env_ps]

    metrics = _evaluate_test_envs(
        algorithm,
        eval_loaders,
        eval_env_names,
        loss_fn,
        device,
        use_alpha=use_alpha,
        h_alphas=h_alphas,
    )

    result = {
        "dataset": "cmnist",
        "device": device,
        "split": split,
        "algorithm_name": hparams["algorithm"],
        "eval_alpha": eval_alpha if use_alpha else None,
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
    }
    for metric in metrics:
        env = str(metric["env"])
        result[f"{env}_acc_eval"] = float(metric["acc"])
        result[f"{env}_loss_eval"] = float(metric["loss"])
    return result
