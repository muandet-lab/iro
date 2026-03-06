"""WILDS-backed iWildCam data integration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torchvision import transforms
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSSubset

IWILDCAM_EVAL_SPLITS = ("val", "test", "id_val", "id_test")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class IWildCamDataBundle:
    """Container with dataset/subsets/grouper and selected evaluation splits."""

    dataset: Any
    grouper: CombinatorialGrouper
    train_data: WILDSSubset
    eval_data: dict[str, WILDSSubset]
    eval_splits: tuple[str, ...]


def parse_iwildcam_eval_splits(raw: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Parse eval split spec into an ordered tuple of valid WILDS iWildCam splits."""

    if raw is None:
        return IWILDCAM_EVAL_SPLITS

    if isinstance(raw, (list, tuple)):
        items = [str(part).strip().lower() for part in raw if str(part).strip()]
    else:
        text = str(raw).strip().lower()
        if not text or text == "all":
            return IWILDCAM_EVAL_SPLITS
        items = [part.strip().lower() for part in text.split(",") if part.strip()]

    if not items:
        return IWILDCAM_EVAL_SPLITS

    out: list[str] = []
    for split in items:
        if split == "all":
            return IWILDCAM_EVAL_SPLITS
        if split not in IWILDCAM_EVAL_SPLITS:
            valid = ", ".join(IWILDCAM_EVAL_SPLITS)
            raise ValueError(f"Invalid iWildCam split '{split}'. Expected one of: {valid}, or 'all'.")
        if split not in out:
            out.append(split)
    return tuple(out)


def resolve_iwildcam_root(*, root: str | None, root_dir: str | None, data_dir: str | None) -> str:
    """Resolve preferred WILDS root directory from supported config aliases."""

    for candidate in (root_dir, data_dir, root):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return "data"


def iwildcam_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def iwildcam_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _debug_subsample_subset(
    subset: WILDSSubset,
    *,
    max_samples: int,
    grouper: CombinatorialGrouper,
    max_groups: int,
) -> WILDSSubset:
    indices = np.asarray(subset.indices)

    if max_groups > 0 and indices.size > 0:
        metadata = subset.dataset.metadata_array[indices]
        group_ids = grouper.metadata_to_group(metadata)
        present_groups = torch.unique(group_ids)
        keep_groups = present_groups[: max_groups]
        keep_mask = torch.isin(group_ids, keep_groups)
        indices = indices[keep_mask.cpu().numpy()]

    if max_samples > 0 and indices.size > max_samples:
        indices = indices[:max_samples]

    return WILDSSubset(subset.dataset, indices, subset.transform)


def build_iwildcam_data_bundle(cfg) -> IWildCamDataBundle:
    """Create WILDS iWildCam dataset, subsets and grouper from config."""

    root = resolve_iwildcam_root(
        root=getattr(cfg.data, "root", None),
        root_dir=getattr(cfg.data, "root_dir", None),
        data_dir=getattr(cfg.data, "data_dir", None),
    )
    download = bool(getattr(cfg.data, "download", False))

    try:
        dataset = get_dataset(dataset="iwildcam", root_dir=root, download=download)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "iWildCam dataset was not found. Set data.download=true to download through WILDS, "
            f"or place data under '{root}'."
        ) from exc

    eval_splits = parse_iwildcam_eval_splits(getattr(cfg.data, "iwildcam_eval_split", "all"))

    train_data = dataset.get_subset("train", transform=iwildcam_train_transform())
    eval_data = {split: dataset.get_subset(split, transform=iwildcam_eval_transform()) for split in eval_splits}

    grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=["location"])

    if bool(getattr(cfg.data, "debug_data", False)):
        train_data = _debug_subsample_subset(
            train_data,
            max_samples=max(1, int(getattr(cfg.data, "debug_train_size", 256))),
            grouper=grouper,
            max_groups=max(0, int(getattr(cfg.data, "debug_group_limit", 0))),
        )
        debug_eval_size = max(1, int(getattr(cfg.data, "debug_eval_size", 128)))
        debug_group_limit = max(0, int(getattr(cfg.data, "debug_group_limit", 0)))
        eval_data = {
            split: _debug_subsample_subset(
                subset,
                max_samples=debug_eval_size,
                grouper=grouper,
                max_groups=debug_group_limit,
            )
            for split, subset in eval_data.items()
        }

    return IWildCamDataBundle(
        dataset=dataset,
        grouper=grouper,
        train_data=train_data,
        eval_data=eval_data,
        eval_splits=eval_splits,
    )


def needs_group_minibatches(algorithm: str) -> bool:
    """Return True for algorithms that need per-environment minibatches."""

    return str(algorithm).strip().lower() != "erm"


def build_iwildcam_train_loader(cfg, bundle: IWildCamDataBundle, *, algorithm: str):
    """Build train loader according to WILDS conventions for the selected algorithm."""

    batch_size = int(cfg.data.batch_size)
    num_workers = int(cfg.data.num_workers)

    if not needs_group_minibatches(algorithm):
        return get_train_loader(
            "standard",
            bundle.train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    n_groups_per_batch = max(1, int(getattr(cfg.data, "n_envs_per_batch", 1)))
    uniform_over_groups = bool(getattr(cfg.data, "uniform_over_groups", True))

    if batch_size % n_groups_per_batch != 0:
        raise ValueError(
            "For WILDS group loaders, data.batch_size must be divisible by data.n_envs_per_batch. "
            f"Got batch_size={batch_size}, n_envs_per_batch={n_groups_per_batch}."
        )

    return get_train_loader(
        "group",
        bundle.train_data,
        batch_size=batch_size,
        grouper=bundle.grouper,
        n_groups_per_batch=n_groups_per_batch,
        uniform_over_groups=uniform_over_groups,
        distinct_groups=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_iwildcam_eval_loader(cfg, subset: WILDSSubset):
    """Build standard evaluation loader for a WILDS subset."""

    batch_size = int(getattr(cfg.eval, "batch_size", None) or cfg.data.batch_size)
    return get_eval_loader(
        "standard",
        subset,
        batch_size=batch_size,
        num_workers=int(cfg.data.num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def split_group_batch_to_minibatches(
    x: torch.Tensor,
    y: torch.Tensor,
    metadata: torch.Tensor,
    *,
    grouper: CombinatorialGrouper,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split one WILDS grouped batch into per-group minibatches for algorithms."""

    group_ids = grouper.metadata_to_group(metadata)
    group_to_indices: dict[int, list[int]] = {}
    for idx, group in enumerate(group_ids.tolist()):
        group_to_indices.setdefault(int(group), []).append(idx)

    minibatches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for idxs in group_to_indices.values():
        idx_tensor = torch.tensor(idxs, device=x.device, dtype=torch.long)
        minibatches.append((x.index_select(0, idx_tensor), y.index_select(0, idx_tensor)))
    return minibatches


__all__ = [
    "IWILDCAM_EVAL_SPLITS",
    "IWildCamDataBundle",
    "parse_iwildcam_eval_splits",
    "resolve_iwildcam_root",
    "build_iwildcam_data_bundle",
    "build_iwildcam_train_loader",
    "build_iwildcam_eval_loader",
    "needs_group_minibatches",
    "split_group_batch_to_minibatches",
]
