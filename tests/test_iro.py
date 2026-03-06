from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from iro import aggregation_function, run_evaluation, run_training
from iro.core import load_experiment_config, supported_experiments, train_from_config
from iro.core.artifacts import _compute_args_id
from iro.utility.algorithms import IRO
from iro.utility.networks import FHatNetwork, FiLMClassifierHead, HyperNetwork


def test_cvar_matches_tail_mean() -> None:
    risks = torch.tensor([1.0, 2.0, 3.0, 10.0])
    alpha = 0.75
    agg = aggregation_function("cvar")
    val = agg.aggregate(risks, alpha)
    assert torch.isclose(val, torch.tensor(10.0))


def test_var_matches_linear_quantile() -> None:
    risks = torch.tensor([1.0, 2.0, 3.0, 10.0])
    alpha = 0.75
    agg = aggregation_function("var")
    val = agg.aggregate(risks, alpha)
    expected = torch.quantile(risks, torch.tensor(alpha), interpolation="linear")
    assert torch.isclose(val, expected)


def test_cvar_diff_executes_on_tensor_risks() -> None:
    risks = torch.tensor([1.0, 2.0, 4.0, 6.0, 10.0])
    alpha = 0.5
    agg = aggregation_function("cvar-diff")
    val = agg.aggregate(risks, alpha)
    expected = torch.stack(
        [torch.quantile(risks, torch.tensor((1 - alpha) * (i / 5) + alpha)) for i in range(5)]
    ).mean()
    assert val.ndim == 0
    assert torch.isfinite(val)
    assert torch.isclose(val, expected)


def test_entropic_and_ph_return_scalar_tensors() -> None:
    risks = torch.tensor([1.0, 2.0, 3.0, 4.0])
    entropic = aggregation_function("entropic").aggregate(risks, eta=0.5)
    ph = aggregation_function("ph").aggregate(risks, xi=1.5)
    assert entropic.ndim == 0
    assert ph.ndim == 0
    assert torch.isfinite(entropic)
    assert torch.isfinite(ph)


class _TinyAlphaNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if alpha.ndim == 1:
            alpha = alpha.unsqueeze(1)
        return self.linear(x) + (alpha * 0.0)


def test_iro_one_step_update_smoke() -> None:
    hparams = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "erm_pretrain_iters": 1,
        "lr_factor_reduction": 1.0,
        "alpha": 0.8,
    }
    algo = IRO(_TinyAlphaNet(), hparams, torch.nn.MSELoss())
    minibatches = [
        (torch.randn(4, 2), torch.randn(4, 1)),
        (torch.randn(4, 2), torch.randn(4, 1)),
    ]
    out = algo.update(minibatches)
    assert "loss" in out
    assert isinstance(out["loss"], float)


def test_fhat_and_hypernetwork_shapes() -> None:
    fhat = FHatNetwork(input_size=8, hidden_sizes=[16, 8], output_size=1)
    x = torch.randn(32, 8)
    y = fhat(x)
    assert y.shape == (32, 1)

    hyper = HyperNetwork(input_dim=8, output_dim=1, hidden_sizes=[])
    out = hyper(x, torch.tensor(0.8))
    assert out.shape == (32, 1)

    film_head = FiLMClassifierHead(feature_dim=8, num_classes=3, hidden_sizes=[16])
    logits = film_head(x, torch.tensor(0.8))
    assert logits.shape == (32, 3)


def test_supported_experiments_include_cmnist_and_iwildcam() -> None:
    assert supported_experiments() == ("cmnist_iro", "iwildcam_iro")


def test_run_training_requires_experiment() -> None:
    with pytest.raises(ValueError, match="No experiment specified"):
        run_training()


def test_run_evaluation_rejects_unknown_experiment() -> None:
    with pytest.raises(ValueError, match="Unsupported experiment"):
        run_evaluation(experiment="cmnist_film_iro")


def test_old_experiment_config_name_is_not_available() -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        load_experiment_config("cmnist_film_iro")


def test_train_from_config_mismatch_fails_fast(tmp_path: Path) -> None:
    cfg = load_experiment_config(
        "cmnist_iro",
        overrides=[
            "data.source=synthetic",
            f"training.output_root={tmp_path.as_posix()}",
            "training.exp_name=mismatch",
        ],
    )
    with pytest.raises(ValueError, match="Config mismatch"):
        train_from_config(cfg, "cmnist_iro")

    results_files = list((tmp_path / "results" / "mismatch").glob("*.jsonl"))
    assert len(results_files) == 1
    record = json.loads(results_files[0].read_text(encoding="utf-8").strip())
    assert record["status"] == "failed"
    assert "Config mismatch" in record["error"]["message"]


def test_cmnist_experiment_config_loads() -> None:
    cfg = load_experiment_config("cmnist_iro")
    assert cfg.data.source == "cmnist"
    assert cfg.data.dataset_name == "cmnist"
    assert cfg.data.batch_size == 25000
    assert cfg.data.cmnist_train_envs == [0.01, 0.12, 0.0, 0.0, 0.99, 0.5, 0.7, 0.01, 0.0, 0.0, 0.14]
    assert cfg.data.cmnist_test_envs == [0.1, 0.5, 0.9]
    assert cfg.data.cmnist_test_env_ms == 0.9
    assert cfg.model.name == "filmedmlp"
    assert cfg.model.hidden_sizes == [390]
    assert cfg.training.steps == 600
    assert cfg.training.loss_fn == "nll"
    assert cfg.training.eval_freq == 50
    assert cfg.iro.algorithm == "iro"
    assert cfg.iro.penalty_weight == 1000.0
    assert cfg.iro.groupdro_eta == 1.0
    assert cfg.iro.alpha == 0.4


def test_iwildcam_experiment_config_loads() -> None:
    cfg = load_experiment_config("iwildcam_iro")
    assert cfg.data.source == "iwildcam"
    assert cfg.data.dataset_name == "iwildcam"
    assert cfg.data.iwildcam_eval_split == "all"
    assert cfg.data.n_envs_per_batch == 4
    assert cfg.model.name == "film_resnet18"
    assert cfg.training.loss_fn == "cross_ent"


def test_master_seed_defaults_to_training_seed() -> None:
    cfg = load_experiment_config("cmnist_iro")
    assert cfg.master_seed == 42
    assert cfg.training.seed == 42


def test_training_seed_override_wins_over_master_seed() -> None:
    cfg = load_experiment_config("cmnist_iro", overrides=["master_seed=42", "training.seed=7"])
    assert cfg.master_seed == 42
    assert cfg.training.seed == 7


def test_master_seed_override_applies_when_training_seed_not_overridden() -> None:
    cfg = load_experiment_config("cmnist_iro", overrides=["master_seed=123"])
    assert cfg.master_seed == 123
    assert cfg.training.seed == 123


def test_executor_slurm_defaults_load() -> None:
    cfg = load_experiment_config("cmnist_iro")
    assert cfg.executor.exec_name == "experiment"
    assert cfg.executor.output_dir == "outputs"
    assert cfg.executor.log_dir == "outputs/logs"
    assert cfg.slurm.cores == 1
    assert cfg.slurm.nodes == 1
    assert cfg.slurm.time == "0-00:30"
    assert cfg.slurm.memory == "5G"
    assert cfg.slurm.partition == "cpu-batch"
    assert cfg.slurm.email_type == "FAIL"
    assert cfg.slurm.log_dir == "outputs/slurm_logs"
    assert cfg.slurm.julia_path == "~/.juliaup/bin"


def test_executor_slurm_overrides_apply() -> None:
    cfg = load_experiment_config(
        "cmnist_iro",
        overrides=["executor.exec_name=foo", "slurm.nodes=2", "slurm.time=1-00:00"],
    )
    assert cfg.executor.exec_name == "foo"
    assert cfg.slurm.nodes == 2
    assert cfg.slurm.time == "1-00:00"


def test_args_id_ignores_master_seed() -> None:
    payload_a = {"master_seed": 1, "training": {"seed": 1, "epochs": 1}, "experiment": "x"}
    payload_b = {"master_seed": 999, "training": {"seed": 999, "epochs": 1}, "experiment": "x"}
    assert _compute_args_id(payload_a) == _compute_args_id(payload_b)
