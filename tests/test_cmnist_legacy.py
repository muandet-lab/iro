from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

import iro.training.train_cmnist as cmnist_mod


def _make_env_dataset(env_p: float, n: int = 10) -> TensorDataset:
    x = torch.zeros(n, 1, 1, 2, dtype=torch.float32)
    x[:, 0, 0, 0] = float(env_p)
    x[:, 0, 0, 1] = torch.linspace(0.0, 1.0, steps=n)
    y = torch.full((n, 1), 1.0 if env_p > 0.5 else 0.0, dtype=torch.float32)
    return TensorDataset(x, y)


def _make_env_dataset_int(env_p: float, n: int = 10) -> TensorDataset:
    x = torch.zeros(n, 1, 1, 2, dtype=torch.float32)
    x[:, 0, 0, 0] = float(env_p)
    x[:, 0, 0, 1] = torch.linspace(0.0, 1.0, steps=n)
    y = torch.full((n,), 1 if env_p > 0.5 else 0, dtype=torch.long)
    return TensorDataset(x, y)


def _base_cfg(**updates):
    cfg = SimpleNamespace(
        training=SimpleNamespace(
            seed=0,
            device="cpu",
            deterministic=False,
            epochs=1,
            steps=2,
            lr=1e-3,
            lr_factor_reduction=1.0,
            lr_cos_sched=False,
            weight_decay=0.0,
            erm_pretrain_iters=0,
            eval_freq=1,
            loss_fn="nll",
            output_root="./iro_exp",
            exp_name="reproduce",
            save_ckpts=True,
        ),
        data=SimpleNamespace(
            root="unused",
            batch_size=10,
            num_workers=0,
            download=False,
            cmnist_train_envs=[0.1, 0.2],
            cmnist_test_envs=[0.1, 0.9],
            cmnist_test_env_ms=0.9,
            cmnist_label_noise_rate=0.25,
            cmnist_subsample=True,
            cmnist_use_test_set=False,
        ),
        model=SimpleNamespace(
            name="mlp",
            hidden_sizes=[8],
            dropout=0.0,
        ),
        iro=SimpleNamespace(
            algorithm="iro",
            penalty_weight=1000.0,
            alpha=0.8,
            groupdro_eta=1.0,
        ),
        eval=SimpleNamespace(
            checkpoint_path="",
            alpha=0.8,
            split="test",
            batch_size=None,
        ),
    )
    for section, section_updates in updates.items():
        target = getattr(cfg, section)
        for key, value in section_updates.items():
            setattr(target, key, value)
    return cfg


class FakeBinaryAlgorithm(torch.nn.Module):
    def __init__(self, network, hparams, loss_fn):
        super().__init__()
        self.step = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(0.0))], lr=float(hparams["lr"]))

    def update(self, minibatches, unlabeled=None):  # noqa: D401
        with torch.no_grad():
            self.step += 1.0
        return {"loss": float(1.0 / (1.0 + self.step.item()))}

    def predict(self, x, alpha=None):
        flat = x.view(x.size(0), -1)
        env = flat[:, 0]
        idx = flat[:, 1]
        logits = torch.zeros_like(env)
        step = int(self.step.item())
        if step <= 1:
            # Step 1: good on low env, poor on high env.
            logits = torch.where(env > 0.5, torch.where(idx < 0.2, 3.0, -3.0), torch.full_like(env, -3.0))
        else:
            # Step 2+: good on high env, bad on low env.
            logits = torch.where(env > 0.5, torch.full_like(env, 3.0), torch.full_like(env, 3.0))
        return logits.view(-1, 1)


class FakeMultiClassAlgorithm(FakeBinaryAlgorithm):
    def predict(self, x, alpha=None):
        flat = x.view(x.size(0), -1)
        env = flat[:, 0]
        logits = torch.stack([-env, env], dim=1)
        return logits


class LRProbeAlgorithm(torch.nn.Module):
    def __init__(self, network, hparams, loss_fn):
        super().__init__()
        self.step = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(0.0))], lr=float(hparams["lr"]))
        self.seen_lrs: list[float] = []

    def update(self, minibatches, unlabeled=None):
        self.seen_lrs.append(float(self.optimizer.param_groups[0]["lr"]))
        with torch.no_grad():
            self.step += 1.0
        return {"loss": 1.0}

    def predict(self, x, alpha=None):
        return torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)


def _fake_dataset_factory(calls: list[dict]):
    def fake_get_cmnist_datasets(
        root,
        train_envs=(0.1, 0.2),
        test_envs=(0.9,),
        label_noise_rate=0.25,
        dataset_transform=None,
        subsample=True,
        int_target=False,
        cuda=None,
        use_test_set=False,
        download=True,
    ):
        calls.append(
            {
                "root": root,
                "train_envs": tuple(float(p) for p in train_envs),
                "test_envs": tuple(float(p) for p in test_envs),
                "int_target": bool(int_target),
                "subsample": bool(subsample),
                "use_test_set": bool(use_test_set),
            }
        )
        out = []
        for p in train_envs:
            if int_target:
                out.append(_make_env_dataset_int(float(p), n=10))
            else:
                out.append(_make_env_dataset(float(p), n=10))
        for p in test_envs:
            if int_target:
                out.append(_make_env_dataset_int(float(p), n=10))
            else:
                out.append(_make_env_dataset(float(p), n=10))
        return out

    return fake_get_cmnist_datasets


def test_parse_env_spec_alias_and_csv() -> None:
    assert cmnist_mod._parse_env_spec([0.1, 0.2], field_name="envs") == (0.1, 0.2)
    assert cmnist_mod._parse_env_spec("default", field_name="envs") == (0.1, 0.2)
    assert cmnist_mod._parse_env_spec("gray", field_name="envs") == (0.5, 0.5)
    assert cmnist_mod._parse_env_spec("0.1,0.5,0.9", field_name="envs") == (0.1, 0.5, 0.9)
    with pytest.raises(ValueError, match="must contain at least one"):
        cmnist_mod._parse_env_spec("", field_name="envs")


def test_cmnist_legacy_outputs_and_selection(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(cmnist_mod, "get_cmnist_datasets", _fake_dataset_factory(calls))
    monkeypatch.setattr(cmnist_mod.algorithms, "get_algorithm_class", lambda _name: FakeBinaryAlgorithm)

    cfg = _base_cfg()
    result = cmnist_mod.train_cmnist_iro(cfg)
    assert calls[0]["train_envs"] == (0.1, 0.2)
    assert calls[0]["test_envs"] == (0.1, 0.9)
    assert result["selection_env"] == "0.9"
    assert "0_acc_final" in result
    assert "1_acc_best" in result
    assert "final_state_dict" in result
    assert "best_state_dict" in result
    assert float(result["best_state_dict"]["step"]) >= 2.0


@pytest.mark.parametrize(
    ("loss_name", "expected_int_target", "algo_class"),
    [
        ("nll", False, FakeBinaryAlgorithm),
        ("cross_ent", True, FakeMultiClassAlgorithm),
    ],
)
def test_loss_fn_switch_controls_int_target(monkeypatch, loss_name, expected_int_target, algo_class):
    calls: list[dict] = []
    monkeypatch.setattr(cmnist_mod, "get_cmnist_datasets", _fake_dataset_factory(calls))
    monkeypatch.setattr(cmnist_mod.algorithms, "get_algorithm_class", lambda _name: algo_class)

    cfg = _base_cfg(training={"steps": 1, "loss_fn": loss_name})
    cmnist_mod.train_cmnist_iro(cfg)
    assert calls[0]["int_target"] is expected_int_target


def test_lr_cosine_schedule_without_pretrain(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(cmnist_mod, "get_cmnist_datasets", _fake_dataset_factory(calls))
    monkeypatch.setattr(cmnist_mod.algorithms, "get_algorithm_class", lambda _name: LRProbeAlgorithm)

    cfg = _base_cfg(
        training={
            "steps": 4,
            "lr": 0.1,
            "lr_cos_sched": True,
            "erm_pretrain_iters": 0,
            "eval_freq": 10,
        }
    )
    result = cmnist_mod.train_cmnist_iro(cfg)
    seen_lrs = result["algorithm"].seen_lrs
    expected = [0.1 * 0.5 * (1.0 + math.cos(math.pi * step / 4.0)) for step in range(1, 5)]
    assert seen_lrs == pytest.approx(expected)


def test_lr_cosine_schedule_with_pretrain(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(cmnist_mod, "get_cmnist_datasets", _fake_dataset_factory(calls))
    monkeypatch.setattr(cmnist_mod.algorithms, "get_algorithm_class", lambda _name: LRProbeAlgorithm)

    cfg = _base_cfg(
        training={
            "steps": 5,
            "lr": 0.1,
            "lr_cos_sched": True,
            "erm_pretrain_iters": 2,
            "lr_factor_reduction": 2.0,
            "eval_freq": 10,
            "save_ckpts": False,
        }
    )
    result = cmnist_mod.train_cmnist_iro(cfg)
    seen_lrs = result["algorithm"].seen_lrs
    expected = [
        0.1,
        0.1,
        0.05 * 0.5 * (1.0 + math.cos(math.pi * 1.0 / 3.0)),
        0.05 * 0.5 * (1.0 + math.cos(math.pi * 2.0 / 3.0)),
        0.05 * 0.5 * (1.0 + math.cos(math.pi * 3.0 / 3.0)),
    ]
    assert seen_lrs == pytest.approx(expected)


def test_legacy_erm_sidecar_save_and_load(monkeypatch, tmp_path):
    calls: list[dict] = []
    monkeypatch.setattr(cmnist_mod, "get_cmnist_datasets", _fake_dataset_factory(calls))
    monkeypatch.setattr(cmnist_mod.algorithms, "get_algorithm_class", lambda _name: FakeBinaryAlgorithm)

    cfg = _base_cfg(
        training={
            "steps": 3,
            "erm_pretrain_iters": 2,
            "output_root": str(tmp_path),
            "save_ckpts": True,
            "eval_freq": 10,
        }
    )
    sidecar = cmnist_mod._legacy_erm_sidecar_path(
        cfg,
        steps=3,
        train_envs_raw=cfg.data.cmnist_train_envs,
        test_envs_raw=cfg.data.cmnist_test_envs,
    )
    assert sidecar is not None

    first_result = cmnist_mod.train_cmnist_iro(cfg)
    assert sidecar.exists()
    saved = torch.load(sidecar, map_location="cpu")
    assert float(saved["step"]) == pytest.approx(2.0)
    assert first_result["legacy_erm_ckpt_path"] == str(sidecar)

    torch.save({"step": torch.tensor(9.0)}, sidecar)
    second_result = cmnist_mod.train_cmnist_iro(cfg)
    assert float(second_result["final_state_dict"]["step"]) == pytest.approx(10.0)


def test_selection_env_must_exist(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(cmnist_mod, "get_cmnist_datasets", _fake_dataset_factory(calls))
    monkeypatch.setattr(cmnist_mod.algorithms, "get_algorithm_class", lambda _name: FakeBinaryAlgorithm)

    cfg = _base_cfg(data={"cmnist_test_env_ms": 0.5})
    with pytest.raises(ValueError, match="Selection env '0.5' not found"):
        cmnist_mod.train_cmnist_iro(cfg)


def test_eval_cmnist_checkpoint_load_and_fixed_alpha(monkeypatch, tmp_path):
    calls: list[dict] = []
    monkeypatch.setattr(cmnist_mod, "get_cmnist_datasets", _fake_dataset_factory(calls))
    monkeypatch.setattr(cmnist_mod.algorithms, "get_algorithm_class", lambda _name: FakeBinaryAlgorithm)

    ckpt_path = tmp_path / "model.pkl"
    torch.save({"state_dict": {"step": torch.tensor(2.0)}}, ckpt_path)

    cfg = _base_cfg(
        training={"steps": 1, "loss_fn": "nll"},
        iro={"algorithm": "iro"},
        eval={
            "checkpoint_path": str(ckpt_path),
            "split": "test",
            "alpha": 0.3,
            "batch_size": 4,
        },
    )

    result = cmnist_mod.eval_cmnist_iro(cfg)
    assert result["split"] == "test"
    assert result["dataset"] == "cmnist"
    assert result["eval_alpha"] == pytest.approx(0.3)
    assert len(result["metrics"]) == 2
    assert all(metric["alpha"] == pytest.approx(0.3) for metric in result["metrics"])
    assert "0.1_acc_eval" in result
    assert "0.9_loss_eval" in result
    assert calls[0]["train_envs"] == ()
    assert calls[0]["test_envs"] == (0.1, 0.9)
    assert calls[0]["use_test_set"] is False


def test_eval_cmnist_split_all_returns_11_envs(monkeypatch, tmp_path):
    calls: list[dict] = []
    monkeypatch.setattr(cmnist_mod, "get_cmnist_datasets", _fake_dataset_factory(calls))
    monkeypatch.setattr(cmnist_mod.algorithms, "get_algorithm_class", lambda _name: FakeBinaryAlgorithm)

    ckpt_path = tmp_path / "raw_state.pkl"
    torch.save({"step": torch.tensor(1.0)}, ckpt_path)

    cfg = _base_cfg(
        training={"steps": 1, "loss_fn": "nll"},
        iro={"algorithm": "erm"},
        eval={
            "checkpoint_path": str(ckpt_path),
            "split": "all",
            "alpha": 0.9,
        },
    )

    result = cmnist_mod.eval_cmnist_iro(cfg)
    assert result["split"] == "all"
    assert len(result["metrics"]) == 11
    assert result["eval_alpha"] is None
    assert all(metric["alpha"] is None for metric in result["metrics"])
    assert calls[0]["train_envs"] == ()
    assert calls[0]["test_envs"] == tuple(i / 10.0 for i in range(11))
    assert calls[0]["use_test_set"] is True
