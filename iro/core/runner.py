"""Top-level CMNIST training/evaluation dispatcher for IRO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from iro.core.artifacts import ArtifactContext
from iro.core.config import ExperimentConfig, load_experiment_config

TrainerFn = Callable[[ExperimentConfig], dict[str, Any]]
EvaluatorFn = Callable[[ExperimentConfig], dict[str, Any]]


def _train_cmnist(cfg: ExperimentConfig) -> dict[str, Any]:
    from iro.training import train_cmnist_iro

    return train_cmnist_iro(cfg)


def _eval_cmnist(cfg: ExperimentConfig) -> dict[str, Any]:
    from iro.training import eval_cmnist_iro

    return eval_cmnist_iro(cfg)


EXPERIMENT_REGISTRY: dict[str, dict[str, Any]] = {
    "cmnist_iro": {
        "source": "cmnist",
        "dataset": "cmnist",
        "trainer": _train_cmnist,
        "evaluator": _eval_cmnist,
    },
}


def _supported_experiments() -> str:
    return ", ".join(supported_experiments())


def supported_experiments() -> tuple[str, ...]:
    return tuple(sorted(EXPERIMENT_REGISTRY.keys()))


def _normalize_overrides(overrides: Iterable[str] | None) -> tuple[str | None, list[str]]:
    experiment: str | None = None
    cleaned: list[str] = []

    for item in overrides or []:
        if item.startswith("experiments="):
            experiment = item.split("=", 1)[1]
            continue
        if item.startswith("experiment="):
            experiment = item.split("=", 1)[1]
            continue
        cleaned.append(item)

    return experiment, cleaned


def _validate_route(cfg: ExperimentConfig, experiment: str) -> dict[str, Any]:
    route = EXPERIMENT_REGISTRY.get(experiment)
    if route is None:
        raise ValueError(
            f"Unsupported experiment '{experiment}'. Supported experiments: {_supported_experiments()}."
        )

    expected_source = str(route["source"]).lower()
    expected_dataset = str(route["dataset"]).lower()
    resolved_source = cfg.data.source.lower()
    resolved_dataset = cfg.data.dataset_name.lower()

    if resolved_source != expected_source or resolved_dataset != expected_dataset:
        raise ValueError(
            f"Config mismatch for experiment '{experiment}': expected data.source='{expected_source}' and "
            f"data.dataset_name='{expected_dataset}', but got data.source='{cfg.data.source}' and "
            f"data.dataset_name='{cfg.data.dataset_name}'."
        )
    return route


def train_from_config(cfg: ExperimentConfig, experiment: str) -> dict[str, Any]:
    with ArtifactContext(cfg, experiment=experiment) as artifacts:
        try:
            route = _validate_route(cfg, experiment)
            trainer = route.get("trainer")
            if trainer is None:
                raise ValueError(f"No trainer is registered for experiment '{experiment}'.")

            result = trainer(cfg)
            artifacts.write_success(result)
            ckpt_meta = artifacts.save_checkpoints(result)
            if artifacts.enabled:
                result["artifacts"] = {
                    **artifacts.as_metadata(),
                    **ckpt_meta,
                }
            return result
        except Exception as exc:
            artifacts.write_failure(exc)
            raise


def evaluate_from_config(cfg: ExperimentConfig, experiment: str) -> dict[str, Any]:
    with ArtifactContext(cfg, experiment=experiment) as artifacts:
        try:
            route = _validate_route(cfg, experiment)
            evaluator = route.get("evaluator")
            if evaluator is None:
                raise ValueError(f"No evaluator is registered for experiment '{experiment}'.")

            result = evaluator(cfg)
            artifacts.write_success(result)
            if artifacts.enabled:
                result["artifacts"] = {
                    **artifacts.as_metadata(),
                }
            return result
        except Exception as exc:
            artifacts.write_failure(exc)
            raise


def run_training(
    *,
    experiment: str | None = None,
    config_name: str = "base_config",
    config_path: str | None = None,
    overrides: list[str] | None = None,
    enable_beartype: bool = False,
) -> dict[str, Any]:
    """Run training for an explicitly selected experiment."""

    if config_name != "base_config":
        raise ValueError("Only config_name='base_config' is currently supported.")

    if enable_beartype:
        try:
            from beartype.claw import beartype_this_package

            beartype_this_package()
        except ImportError:
            pass

    experiment_from_overrides, cleaned_overrides = _normalize_overrides(overrides)
    if experiment is not None and experiment_from_overrides is not None and experiment != experiment_from_overrides:
        raise ValueError(
            f"Conflicting experiment values: experiment='{experiment}' and "
            f"overrides experiment='{experiment_from_overrides}'."
        )

    resolved_experiment = experiment or experiment_from_overrides
    if not resolved_experiment:
        raise ValueError(
            "No experiment specified. Pass experiment='<name>' or include override "
            "'experiments=<name>'. CLI usage: python -m iro train --experiment <name>."
        )
    if resolved_experiment not in EXPERIMENT_REGISTRY:
        raise ValueError(
            f"Unsupported experiment '{resolved_experiment}'. Supported experiments: {_supported_experiments()}."
        )

    resolved_config_root = config_path
    if resolved_config_root is None:
        package_config = Path(__file__).resolve().parents[2] / "config"
        resolved_config_root = str(package_config if package_config.exists() else Path.cwd() / "config")

    cfg = load_experiment_config(
        experiment=resolved_experiment,
        config_root=resolved_config_root,
        overrides=cleaned_overrides,
    )

    return train_from_config(cfg, resolved_experiment)


def run_evaluation(
    *,
    experiment: str,
    config_path: str | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Compose config and run evaluation for an explicitly selected experiment."""

    if experiment not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Unsupported experiment '{experiment}'. Supported experiments: {_supported_experiments()}.")

    resolved_config_root = config_path
    if resolved_config_root is None:
        package_config = Path(__file__).resolve().parents[2] / "config"
        resolved_config_root = str(package_config if package_config.exists() else Path.cwd() / "config")

    cfg = load_experiment_config(
        experiment=experiment,
        config_root=resolved_config_root,
        overrides=list(overrides or []),
    )
    return evaluate_from_config(cfg, experiment)
