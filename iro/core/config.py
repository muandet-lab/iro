"""Experiment configuration loading for the CMNIST-first layout."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

try:
    from hydra import compose, initialize_config_dir
    from hydra.errors import ConfigCompositionException
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover - graceful fallback before hydra is installed
    compose = None  # type: ignore[assignment]
    initialize_config_dir = None  # type: ignore[assignment]
    ConfigCompositionException = None  # type: ignore[assignment,misc]
    DictConfig = dict  # type: ignore[assignment,misc]
    OmegaConf = None  # type: ignore[assignment]


@dataclass
class TrainingConfig:
    seed: int = 0
    device: str = "auto"
    deterministic: bool = False
    epochs: int = 30
    steps: int | None = None
    lr: float = 2e-4
    lr_factor_reduction: float = 1.0
    lr_cos_sched: bool = False
    weight_decay: float = 0.0
    erm_pretrain_iters: int = 0
    eval_freq: int = 50
    loss_fn: str = "nll"
    output_root: str = "./iro_exp"
    exp_name: str = "reproduce"
    save_ckpts: bool = True
    capture_logs: bool = True
    write_artifacts: bool = True


@dataclass
class IROConfig:
    algorithm: str = "iro"
    penalty_weight: float = 1000.0
    groupdro_eta: float = 1.0
    alpha: float = 0.8
    alpha_samples: int = 10
    pareto_num_samples: int = 5


@dataclass
class ModelConfig:
    name: str = "filmedmlp"
    hidden_sizes: list[int] = field(default_factory=lambda: [390])
    dropout: float = 0.2
    pretrained: bool = False


@dataclass
class DataConfig:
    source: str = "cmnist"
    dataset_name: str = "cmnist"
    root: str = "data/cmnist"
    root_dir: str = ""
    data_dir: str = ""
    download: bool = False
    batch_size: int = 16
    num_workers: int = 0
    iwildcam_eval_split: str = "all"
    n_envs_per_batch: int = 4
    uniform_over_groups: bool = True
    debug_data: bool = False
    debug_train_size: int = 256
    debug_eval_size: int = 128
    debug_group_limit: int = 0
    iwildcam_image_size: int = 224
    iwildcam_eval_resize: int = 256
    cmnist_train_envs: list[float] | str = field(default_factory=lambda: [0.1, 0.2])
    cmnist_test_envs: list[float] | str = field(default_factory=lambda: [0.9])
    cmnist_test_env_ms: float = 0.9
    cmnist_label_noise_rate: float = 0.25
    cmnist_subsample: bool = True
    cmnist_use_test_set: bool = False


@dataclass
class EvalConfig:
    checkpoint_path: str = ""
    alpha: float = 0.8
    split: str = "test"
    batch_size: int | None = None


@dataclass
class ExecutorConfig:
    exec_name: str = "experiment"
    output_dir: str = "outputs"
    log_dir: str = "outputs/logs"


@dataclass
class SlurmConfig:
    cores: int = 1
    nodes: int = 1
    time: str = "0-00:30"
    memory: str = "5G"
    partition: str = "cpu-batch"
    account: str = ""
    email: str = ""
    email_type: str = "FAIL"
    log_dir: str = "outputs/slurm_logs"
    job_name: str = "experiment"
    exclude: str = ""
    constraint: str = ""
    gres: str = ""
    julia_path: str = "~/.juliaup/bin"


@dataclass
class ExperimentConfig:
    experiment: str = "cmnist_iro"
    master_seed: int = 42
    training: TrainingConfig = field(default_factory=TrainingConfig)
    iro: IROConfig = field(default_factory=IROConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_dataclass(cls, raw: dict[str, Any]):
    allowed = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    return cls(**filtered)


def _parse_merged_config(merged: dict[str, Any], fallback_experiment: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=str(merged.get("experiment", fallback_experiment)),
        master_seed=int(merged.get("master_seed", 42)),
        training=_build_dataclass(TrainingConfig, merged.get("training", {})),
        iro=_build_dataclass(IROConfig, merged.get("iro", {})),
        model=_build_dataclass(ModelConfig, merged.get("model", {})),
        data=_build_dataclass(DataConfig, merged.get("data", {})),
        eval=_build_dataclass(EvalConfig, merged.get("eval", {})),
        executor=_build_dataclass(ExecutorConfig, merged.get("executor", {})),
        slurm=_build_dataclass(SlurmConfig, merged.get("slurm", {})),
    )


def _parse_override(override: str) -> tuple[list[str], Any, bool] | None:
    if "=" not in override:
        return None
    left, right = override.split("=", 1)
    allow_create = left.startswith("+")
    key = left[1:] if allow_create else left
    if key in {"experiments", "experiment"}:
        return None
    parts = [p for p in key.split(".") if p]
    if not parts:
        return None
    try:
        value: Any = yaml.safe_load(right)
    except Exception:
        value = right
    return parts, value, allow_create


def _set_nested(mapping: dict[str, Any], path: list[str], value: Any, *, allow_create: bool) -> None:
    cur = mapping
    for key in path[:-1]:
        if key not in cur:
            if not allow_create:
                raise KeyError(".".join(path))
            cur[key] = {}
        if not isinstance(cur[key], dict):
            if not allow_create:
                raise KeyError(".".join(path))
            cur[key] = {}
        cur = cur[key]
    leaf = path[-1]
    if not allow_create and leaf not in cur:
        raise KeyError(".".join(path))
    cur[leaf] = value


def _apply_overrides(merged: dict[str, Any], overrides: list[str] | None, *, strict: bool) -> dict[str, Any]:
    out = copy.deepcopy(merged)
    for raw in overrides or []:
        parsed = _parse_override(raw)
        if parsed is None:
            continue
        path, value, allow_create = parsed
        try:
            _set_nested(out, path, value, allow_create=allow_create)
        except KeyError as exc:
            if strict:
                msg = (
                    f"Config override failed for '{raw}'. Use keys defined in config/base_config.yaml or the "
                    "selected experiment config. For new keys, prefix with '+'."
                )
                raise ValueError(msg) from exc
    return out


def _override_sets_training_seed(overrides: list[str] | None) -> bool:
    for raw in overrides or []:
        parsed = _parse_override(raw)
        if parsed is None:
            continue
        path, _value, _allow_create = parsed
        if path == ["training", "seed"]:
            return True
    return False


def _experiment_sets_training_seed(exp_cfg: dict[str, Any]) -> bool:
    training = exp_cfg.get("training")
    return isinstance(training, dict) and "seed" in training


def _resolve_master_seed_for_training_seed(
    merged: dict[str, Any],
    *,
    experiment_has_seed: bool,
    override_has_seed: bool,
) -> dict[str, Any]:
    if experiment_has_seed or override_has_seed:
        return merged

    if "master_seed" not in merged:
        return merged

    out = copy.deepcopy(merged)
    training = out.get("training")
    if not isinstance(training, dict):
        training = {}
        out["training"] = training
    training["seed"] = int(out["master_seed"])
    return out


def _compose_with_explicit_precedence(
    experiment: str,
    root: Path,
    overrides: list[str] | None,
    *,
    strict_overrides: bool,
) -> dict[str, Any]:
    base_cfg = _read_yaml(root / "base_config.yaml")
    exp_cfg = _read_yaml(root / "experiments" / f"{experiment}.yaml")

    merged = _deep_merge(copy.deepcopy(base_cfg), exp_cfg)
    merged = _apply_overrides(merged, overrides, strict=strict_overrides)
    return _resolve_master_seed_for_training_seed(
        merged,
        experiment_has_seed=_experiment_sets_training_seed(exp_cfg),
        override_has_seed=_override_sets_training_seed(overrides),
    )


def load_experiment_config(
    experiment: str = "cmnist_iro",
    *,
    config_root: str | None = None,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    root = Path(config_root) if config_root is not None else Path.cwd() / "config"

    # Keep Hydra in the loop for override validation when available.
    if compose is not None and initialize_config_dir is not None and OmegaConf is not None:
        hydra_overrides = [f"experiments={experiment}"] + list(overrides or [])
        try:
            with initialize_config_dir(version_base=None, config_dir=str(root.resolve())):
                cfg: DictConfig = compose(config_name="base_config", overrides=hydra_overrides)
        except Exception as exc:
            if ConfigCompositionException is not None and isinstance(exc, ConfigCompositionException):
                raise ValueError(
                    "Config override failed. Use keys defined in config/base_config.yaml or the selected "
                    "experiment config. For new keys, prefix with '+'. "
                    "Make sure an explicit experiment is selected (CLI: python -m iro train --experiment <name>). "
                    f"Details: {exc}"
                ) from exc
            raise

        _ = OmegaConf.to_container(cfg, resolve=True)
        merged = _compose_with_explicit_precedence(
            experiment=experiment,
            root=root,
            overrides=overrides,
            strict_overrides=False,
        )
        return _parse_merged_config(merged, fallback_experiment=experiment)

    merged = _compose_with_explicit_precedence(
        experiment=experiment,
        root=root,
        overrides=overrides,
        strict_overrides=True,
    )
    return _parse_merged_config(merged, fallback_experiment=experiment)
