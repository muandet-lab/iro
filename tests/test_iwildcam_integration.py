from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import iro.data.iwildcam as iwildcam_data
import iro.training.train_iwildcam as iwildcam_train


class FakeSubset:
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[int(self.indices[idx])]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, metadata

    @property
    def metadata_array(self):
        return self.dataset.metadata_array[self.indices]


class FakeGrouper:
    def metadata_to_group(self, metadata):
        return metadata[:, 0].long()


class FakeWildsDataset:
    def __init__(self):
        self.n_classes = 3
        self.eval_calls: list[dict] = []
        self.metadata_array = torch.tensor(
            [[0], [0], [1], [1], [2], [2]],
            dtype=torch.long,
        )
        self._x = torch.randn(6, 1, 2, 2)
        self._y = torch.tensor([0, 1, 1, 2, 0, 2], dtype=torch.long)

    def __getitem__(self, idx: int):
        return self._x[idx], self._y[idx], self.metadata_array[idx]

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        self.eval_calls.append({"prediction_fn": prediction_fn, "n": int(y_true.numel())})
        preds = prediction_fn(y_pred) if prediction_fn is not None else y_pred
        acc = float((preds.view(-1) == y_true.view(-1)).float().mean().item())
        return {
            "acc_avg": acc,
            "recall-macro_all": acc,
            "F1-macro_all": acc,
        }, "fake"


class TinyNet(torch.nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(4, n_classes)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor | None = None) -> torch.Tensor:
        out = self.linear(x.view(x.size(0), -1))
        if alpha is None:
            return out
        alpha = alpha.to(device=x.device, dtype=x.dtype)
        if alpha.ndim == 0:
            alpha = alpha.view(1, 1)
        elif alpha.ndim == 1:
            alpha = alpha.view(-1, 1)
        if alpha.size(0) == 1:
            alpha = alpha.expand(x.size(0), 1)
        return out + alpha[:, :1]


class FakeAlgorithm(torch.nn.Module):
    def __init__(self, network, hparams, loss_fn):
        super().__init__()
        self.network = network
        self.loss_fn = loss_fn
        self.hparams = hparams
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=float(hparams["lr"]))
        self.seen_minibatch_counts: list[int] = []

    def update(self, minibatches, unlabeled=None):
        self.seen_minibatch_counts.append(len(minibatches))
        loss = torch.tensor(0.0, device=minibatches[0][0].device)
        for x, y in minibatches:
            logits = self.network(x)
            loss = loss + self.loss_fn(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu())}

    def predict(self, x, alpha=None):
        if alpha is None:
            return self.network(x)
        return self.network(x, alpha)


@dataclass
class Bundle:
    dataset: FakeWildsDataset
    grouper: FakeGrouper
    train_data: FakeSubset
    eval_data: dict[str, FakeSubset]
    eval_splits: tuple[str, ...]


def _cfg(algorithm: str = "iro"):
    return SimpleNamespace(
        training=SimpleNamespace(
            seed=0,
            device="cpu",
            deterministic=False,
            epochs=1,
            steps=2,
            lr=1e-2,
            lr_factor_reduction=1.0,
            lr_cos_sched=False,
            weight_decay=0.0,
            erm_pretrain_iters=0,
            eval_freq=1,
        ),
        data=SimpleNamespace(
            root="unused",
            root_dir="",
            data_dir="",
            download=False,
            batch_size=4,
            num_workers=0,
            iwildcam_eval_split="val,test",
            n_envs_per_batch=2,
            uniform_over_groups=True,
            debug_data=False,
            debug_train_size=256,
            debug_eval_size=128,
            debug_group_limit=0,
        ),
        model=SimpleNamespace(
            name="film_resnet18",
            hidden_sizes=[16],
            dropout=0.0,
            pretrained=False,
        ),
        iro=SimpleNamespace(
            algorithm=algorithm,
            penalty_weight=1000.0,
            alpha=0.8,
            groupdro_eta=1.0,
        ),
        eval=SimpleNamespace(
            checkpoint_path="",
            alpha=0.8,
            split="all",
            batch_size=None,
        ),
    )


def test_parse_iwildcam_eval_splits_variants() -> None:
    assert iwildcam_data.parse_iwildcam_eval_splits("all") == ("val", "test", "id_val", "id_test")
    assert iwildcam_data.parse_iwildcam_eval_splits("val,test") == ("val", "test")
    assert iwildcam_data.parse_iwildcam_eval_splits(["id_val", "id_test"]) == ("id_val", "id_test")
    with pytest.raises(ValueError, match="Invalid iWildCam split"):
        iwildcam_data.parse_iwildcam_eval_splits("bad_split")


def test_iwildcam_transform_size_resolution_from_config() -> None:
    cfg = _cfg()
    assert iwildcam_data.resolve_iwildcam_image_size(cfg) == 224
    assert iwildcam_data.resolve_iwildcam_eval_resize(cfg, image_size=224) == 256

    cfg.data.iwildcam_image_size = 448
    cfg.data.iwildcam_eval_resize = 512
    assert iwildcam_data.resolve_iwildcam_image_size(cfg) == 448
    assert iwildcam_data.resolve_iwildcam_eval_resize(cfg, image_size=448) == 512


def test_debug_subsample_and_group_split_helpers(monkeypatch) -> None:
    monkeypatch.setattr(iwildcam_data, "WILDSSubset", FakeSubset)
    dataset = FakeWildsDataset()
    subset = FakeSubset(dataset, indices=np.arange(6), transform=None)
    limited = iwildcam_data._debug_subsample_subset(
        subset,
        max_samples=3,
        grouper=FakeGrouper(),
        max_groups=2,
    )
    assert len(limited.indices) <= 3
    kept_groups = dataset.metadata_array[limited.indices, 0].unique().tolist()
    assert set(kept_groups).issubset({0, 1})

    x = torch.randn(6, 1, 2, 2)
    y = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    metadata = torch.tensor([[1], [1], [2], [3], [2], [3]], dtype=torch.long)
    minibatches = iwildcam_data.split_group_batch_to_minibatches(x, y, metadata, grouper=FakeGrouper())
    assert [mb[0].shape[0] for mb in minibatches] == [2, 2, 2]


def test_build_iwildcam_train_loader_selects_standard_or_group(monkeypatch) -> None:
    calls = []

    def fake_get_train_loader(loader, dataset, batch_size, **kwargs):
        calls.append((loader, batch_size, kwargs))
        return ["loader"]

    monkeypatch.setattr(iwildcam_data, "get_train_loader", fake_get_train_loader)
    cfg = _cfg(algorithm="erm")
    bundle = Bundle(
        dataset=FakeWildsDataset(),
        grouper=FakeGrouper(),
        train_data=FakeSubset(FakeWildsDataset(), np.arange(4), None),
        eval_data={},
        eval_splits=(),
    )

    iwildcam_data.build_iwildcam_train_loader(cfg, bundle, algorithm="erm")
    iwildcam_data.build_iwildcam_train_loader(cfg, bundle, algorithm="iro")

    assert calls[0][0] == "standard"
    assert calls[1][0] == "group"
    assert calls[1][2]["n_groups_per_batch"] == 2
    assert calls[1][2]["distinct_groups"] is True


def test_build_iwildcam_train_loader_adjusts_groups_for_debug_subset(monkeypatch) -> None:
    calls = []

    def fake_get_train_loader(loader, dataset, batch_size, **kwargs):
        calls.append((loader, batch_size, kwargs))
        return ["loader"]

    monkeypatch.setattr(iwildcam_data, "get_train_loader", fake_get_train_loader)
    cfg = _cfg(algorithm="iro")
    cfg.data.batch_size = 32
    cfg.data.n_envs_per_batch = 4
    dataset = FakeWildsDataset()
    train_subset = FakeSubset(dataset, np.array([0, 1, 2, 3, 4]), transform=None)  # groups {0,1,2}
    bundle = Bundle(
        dataset=dataset,
        grouper=FakeGrouper(),
        train_data=train_subset,
        eval_data={},
        eval_splits=(),
    )

    with pytest.warns(RuntimeWarning, match="Adjusted data.n_envs_per_batch"):
        iwildcam_data.build_iwildcam_train_loader(cfg, bundle, algorithm="iro")

    assert calls[0][0] == "group"
    assert calls[0][2]["n_groups_per_batch"] == 2


def test_train_iwildcam_uses_group_minibatches_and_dataset_eval(monkeypatch) -> None:
    dataset = FakeWildsDataset()
    train_subset = FakeSubset(dataset, np.array([0, 1, 2, 3]), transform=None)
    eval_subset = FakeSubset(dataset, np.array([0, 1, 2, 3]), transform=None)
    bundle = Bundle(
        dataset=dataset,
        grouper=FakeGrouper(),
        train_data=train_subset,
        eval_data={"val": eval_subset, "test": eval_subset},
        eval_splits=("val", "test"),
    )

    train_batch = (
        dataset._x[:4],
        dataset._y[:4],
        dataset.metadata_array[:4],
    )
    eval_batch = (
        dataset._x[:4],
        dataset._y[:4],
        dataset.metadata_array[:4],
    )

    monkeypatch.setattr(iwildcam_train, "build_iwildcam_data_bundle", lambda cfg: bundle)
    monkeypatch.setattr(iwildcam_train, "build_iwildcam_train_loader", lambda cfg, bundle, algorithm: [train_batch])
    monkeypatch.setattr(iwildcam_train, "build_iwildcam_eval_loader", lambda cfg, subset: [eval_batch])
    monkeypatch.setattr(iwildcam_train.networks, "FiLMedResNetClassifier", lambda **kwargs: TinyNet())
    monkeypatch.setattr(iwildcam_train.algorithms, "get_algorithm_class", lambda _name: FakeAlgorithm)

    cfg = _cfg(algorithm="iro")
    result = iwildcam_train.train_iwildcam_iro(cfg)

    assert result["dataset"] == "iwildcam"
    assert "val_acc_final" in result
    assert "test_acc_best" in result
    assert result["algorithm"].seen_minibatch_counts[0] >= 2
    assert dataset.eval_calls
    assert all(call["prediction_fn"] is not None for call in dataset.eval_calls)


def test_eval_iwildcam_outputs_split_metrics(monkeypatch, tmp_path) -> None:
    dataset = FakeWildsDataset()
    subset = FakeSubset(dataset, np.array([0, 1, 2, 3]), transform=None)
    bundle = Bundle(
        dataset=dataset,
        grouper=FakeGrouper(),
        train_data=subset,
        eval_data={"val": subset, "test": subset},
        eval_splits=("val", "test"),
    )

    eval_batch = (
        dataset._x[:4],
        dataset._y[:4],
        dataset.metadata_array[:4],
    )

    monkeypatch.setattr(iwildcam_train, "build_iwildcam_data_bundle", lambda cfg: bundle)
    monkeypatch.setattr(iwildcam_train, "build_iwildcam_eval_loader", lambda cfg, subset: [eval_batch])
    monkeypatch.setattr(iwildcam_train.networks, "FiLMedResNetClassifier", lambda **kwargs: TinyNet())
    monkeypatch.setattr(iwildcam_train.algorithms, "get_algorithm_class", lambda _name: FakeAlgorithm)

    cfg = _cfg(algorithm="erm")
    ckpt = tmp_path / "ckpt.pkl"
    model = TinyNet()
    torch.save(model.state_dict(), ckpt)
    cfg.eval.checkpoint_path = str(ckpt)
    cfg.eval.split = "val,test"

    result = iwildcam_train.eval_iwildcam_iro(cfg)

    assert result["dataset"] == "iwildcam"
    assert len(result["metrics"]) == 2
    assert "val_acc_eval" in result
    assert "test_macro_f1_eval" in result
