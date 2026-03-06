"""CLI command implementations for IRO training/evaluation."""

from __future__ import annotations

import typer

from iro.core import evaluate_from_config, load_experiment_config, supported_experiments, train_from_config


def register_commands(app: typer.Typer) -> None:
    def _run_or_exit(fn):
        try:
            return fn()
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    @app.command("train")
    def train_cmd(
        experiment: str = typer.Option(
            ...,
            "--experiment",
            "-e",
            help="Experiment config name under config/experiments (example: cmnist_iro).",
        ),
        config_path: str = typer.Option("config", help="Config root path."),
        override: list[str] = typer.Option(
            None,
            "--override",
            "-o",
            help="Hydra-style override, repeatable (example: -o data.root=/path).",
        ),
    ) -> None:
        if experiment not in supported_experiments():
            raise typer.BadParameter(
                f"Unsupported experiment '{experiment}'. Supported experiments: {', '.join(supported_experiments())}.",
                param_hint="--experiment",
            )
        cfg = _run_or_exit(
            lambda: load_experiment_config(
                experiment=experiment,
                config_root=config_path,
                overrides=list(override or []),
            )
        )
        typer.echo(
            f"run experiment={experiment} source={cfg.data.source} "
            f"dataset={cfg.data.dataset_name} root={cfg.data.root}"
        )
        result = _run_or_exit(lambda: train_from_config(cfg, experiment))

        dataset = result.get("dataset", cfg.data.dataset_name)
        device = result.get("device", "unknown")
        typer.echo(f"dataset={dataset} device={device} completed")

        artifacts = result.get("artifacts", {})
        if artifacts:
            typer.echo(f"results_file={artifacts.get('results_file')}")
            if artifacts.get("ckpt_final"):
                typer.echo(f"ckpt_final={artifacts.get('ckpt_final')}")
            if artifacts.get("ckpt_best"):
                typer.echo(f"ckpt_best={artifacts.get('ckpt_best')}")

    @app.command("eval")
    def eval_cmd(
        experiment: str = typer.Option(
            ...,
            "--experiment",
            "-e",
            help="Experiment config name under config/experiments (example: cmnist_iro).",
        ),
        config_path: str = typer.Option("config", help="Config root path."),
        override: list[str] = typer.Option(
            None,
            "--override",
            "-o",
            help="Hydra-style override, repeatable (example: -o eval.checkpoint_path=/path/to/ckpt.pkl).",
        ),
    ) -> None:
        if experiment not in supported_experiments():
            raise typer.BadParameter(
                f"Unsupported experiment '{experiment}'. Supported experiments: {', '.join(supported_experiments())}.",
                param_hint="--experiment",
            )
        cfg = _run_or_exit(
            lambda: load_experiment_config(
                experiment=experiment,
                config_root=config_path,
                overrides=list(override or []),
            )
        )
        typer.echo(
            f"eval experiment={experiment} source={cfg.data.source} dataset={cfg.data.dataset_name} "
            f"root={cfg.data.root} split={cfg.eval.split} alpha={cfg.eval.alpha}"
        )
        result = _run_or_exit(lambda: evaluate_from_config(cfg, experiment))

        dataset = result.get("dataset", cfg.data.dataset_name)
        device = result.get("device", "unknown")
        metrics = result.get("metrics", [])

        if isinstance(metrics, list) and metrics:
            first_metric = metrics[0] if isinstance(metrics[0], dict) else {}
            if {"env", "acc", "loss"}.issubset(first_metric):
                typer.echo(f"dataset={dataset} device={device} split={result.get('split', cfg.eval.split)}")
                for metric in metrics:
                    env = metric.get("env")
                    alpha = metric.get("alpha")
                    acc = metric.get("acc")
                    loss = metric.get("loss")
                    typer.echo(f"env={env} alpha={alpha} acc={acc:.6f} loss={loss:.6f}")
            else:
                typer.echo(f"dataset={dataset} device={device} split={result.get('split', cfg.eval.split)}")
                for metric in metrics:
                    if not isinstance(metric, dict):
                        typer.echo(str(metric))
                        continue
                    split_name = metric.get("split", "unknown")
                    alpha = metric.get("alpha")
                    acc = metric.get("accuracy", metric.get("acc"))
                    macro_recall = metric.get("macro_recall", metric.get("recall-macro_all"))
                    macro_f1 = metric.get("macro_f1", metric.get("F1-macro_all"))

                    parts = [f"split={split_name}"]
                    if alpha is not None:
                        parts.append(f"alpha={alpha}")
                    if acc is not None:
                        parts.append(f"acc={float(acc):.6f}")
                    if macro_recall is not None:
                        parts.append(f"macro_recall={float(macro_recall):.6f}")
                    if macro_f1 is not None:
                        parts.append(f"macro_f1={float(macro_f1):.6f}")
                    typer.echo(" ".join(parts))
        else:
            typer.echo(f"dataset={dataset} device={device} evaluation_completed")

        artifacts = result.get("artifacts", {})
        if artifacts:
            typer.echo(f"results_file={artifacts.get('results_file')}")
