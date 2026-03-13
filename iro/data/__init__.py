"""Dataset exports for supported experiment workflows."""

from __future__ import annotations

from .base_dataset import BaseDataset
from .cmnist_dataset import get_cmnist_datasets
from .custom_dataset import CustomDataset

IWILDCAM_EVAL_SPLITS = ("val", "test", "id_val", "id_test")
_iwildcam_import_error: ModuleNotFoundError | None = None

try:
    from .iwildcam import (
        IWILDCAM_EVAL_SPLITS,
        build_iwildcam_data_bundle,
        build_iwildcam_eval_loader,
        build_iwildcam_train_loader,
        parse_iwildcam_eval_splits,
        split_group_batch_to_minibatches,
    )
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "") != "wilds":
        raise
    _iwildcam_import_error = exc

    def _raise_wilds_missing(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "Missing optional dependency 'wilds' required for iWildCam support. "
            "Install it with `python -m pip install wilds` (or reinstall with `python -m pip install -e .`)."
        ) from _iwildcam_import_error

    parse_iwildcam_eval_splits = _raise_wilds_missing
    build_iwildcam_data_bundle = _raise_wilds_missing
    build_iwildcam_train_loader = _raise_wilds_missing
    build_iwildcam_eval_loader = _raise_wilds_missing
    split_group_batch_to_minibatches = _raise_wilds_missing

__all__ = [
    "BaseDataset",
    "CustomDataset",
    "get_cmnist_datasets",
    "IWILDCAM_EVAL_SPLITS",
    "parse_iwildcam_eval_splits",
    "build_iwildcam_data_bundle",
    "build_iwildcam_train_loader",
    "build_iwildcam_eval_loader",
    "split_group_batch_to_minibatches",
]
