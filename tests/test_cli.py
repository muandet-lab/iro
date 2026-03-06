from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

import iro.cli.train as train_cli_mod
from iro.cli import app

runner = CliRunner()


def _fake_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(source="cmnist", dataset_name="cmnist", root="data/cmnist"),
        eval=SimpleNamespace(split="test", alpha=0.8),
    )


def test_cli_exposes_train_and_eval_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "eval" in result.stdout
    assert "run-training" not in result.stdout
    assert "train-cmnist" not in result.stdout


def test_removed_commands_are_rejected() -> None:
    result = runner.invoke(app, ["train-cmnist"])
    assert result.exit_code != 0
    assert "No such command" in result.output


def test_unknown_experiment_fails_with_supported_list() -> None:
    result = runner.invoke(app, ["train", "--experiment", "does_not_exist"])
    assert result.exit_code != 0
    assert "Unsupported experiment 'does_not_exist'" in result.output
    assert "Supported experiments:" in result.output


def test_cli_train_prints_artifact_paths(monkeypatch) -> None:
    monkeypatch.setattr(train_cli_mod, "load_experiment_config", lambda **_kwargs: _fake_cfg())
    monkeypatch.setattr(
        train_cli_mod,
        "train_from_config",
        lambda cfg, experiment: {
            "dataset": "cmnist",
            "device": "cpu",
            "artifacts": {
                "results_file": "/tmp/results.jsonl",
                "ckpt_final": "/tmp/final.pkl",
                "ckpt_best": "/tmp/best.pkl",
            },
        },
    )

    result = runner.invoke(app, ["train", "--experiment", "cmnist_iro"])
    assert result.exit_code == 0
    assert "results_file=/tmp/results.jsonl" in result.output
    assert "ckpt_final=/tmp/final.pkl" in result.output
    assert "ckpt_best=/tmp/best.pkl" in result.output


def test_cli_eval_prints_cmnist_metrics(monkeypatch) -> None:
    monkeypatch.setattr(train_cli_mod, "load_experiment_config", lambda **_kwargs: _fake_cfg())
    monkeypatch.setattr(
        train_cli_mod,
        "evaluate_from_config",
        lambda cfg, experiment: {
            "dataset": "cmnist",
            "device": "cpu",
            "split": "test",
            "metrics": [
                {"env": "0.1", "alpha": 0.8, "acc": 0.75, "loss": 0.6},
                {"env": "0.9", "alpha": 0.8, "acc": 0.55, "loss": 0.9},
            ],
        },
    )

    result = runner.invoke(app, ["eval", "--experiment", "cmnist_iro"])
    assert result.exit_code == 0
    assert "dataset=cmnist device=cpu split=test" in result.output
    assert "env=0.1 alpha=0.8 acc=0.750000 loss=0.600000" in result.output
    assert "env=0.9 alpha=0.8 acc=0.550000 loss=0.900000" in result.output


def test_cli_eval_prints_iwildcam_metrics(monkeypatch) -> None:
    monkeypatch.setattr(train_cli_mod, "load_experiment_config", lambda **_kwargs: _fake_cfg())
    monkeypatch.setattr(
        train_cli_mod,
        "evaluate_from_config",
        lambda cfg, experiment: {
            "dataset": "iwildcam",
            "device": "cpu",
            "split": "val,test",
            "metrics": [
                {"split": "val", "alpha": 0.8, "accuracy": 0.44, "macro_recall": 0.33, "macro_f1": 0.22},
                {"split": "test", "alpha": 0.8, "accuracy": 0.40, "macro_recall": 0.30, "macro_f1": 0.20},
            ],
        },
    )

    result = runner.invoke(app, ["eval", "--experiment", "cmnist_iro"])
    assert result.exit_code == 0
    assert "dataset=iwildcam device=cpu split=val,test" in result.output
    assert "split=val alpha=0.8 acc=0.440000 macro_recall=0.330000 macro_f1=0.220000" in result.output
