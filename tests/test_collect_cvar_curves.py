from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "collect_cvar_curves.py"
    spec = importlib.util.spec_from_file_location("collect_cvar_curves", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_alpha_grid_and_cvar() -> None:
    mod = _load_module()
    assert mod.parse_alpha_grid_spec("0,0.5,1") == (0.0, 0.5, 1.0)
    assert mod.parse_alpha_grid_spec("") == mod.DEFAULT_ALPHA_GRID
    risks = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    assert mod.cvar_from_risks(risks, 0.5) == 3.5


def test_select_best_run_records_filters_by_experiment(tmp_path: Path) -> None:
    mod = _load_module()

    out = tmp_path / "out"
    ckpt_dir = out / "ckpts"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "run_iw_best.pkl").write_bytes(b"ok")
    (ckpt_dir / "run_cm_best.pkl").write_bytes(b"ok")

    records = [
        {
            "status": "ok",
            "source": "iwildcam",
            "dataset_name": "iwildcam",
            "algorithm": "iro",
            "seed": 0,
            "run_id": "run_iw",
            "output_root": str(out),
            "val_acc_best": 0.4,
            "config": {"training": {"output_root": str(out)}},
        },
        {
            "status": "ok",
            "source": "cmnist",
            "dataset_name": "cmnist",
            "algorithm": "iro",
            "seed": 0,
            "run_id": "run_cm",
            "output_root": str(out),
            "0.9_acc_best": 0.8,
            "config": {"training": {"output_root": str(out)}},
        },
    ]

    iw = mod.select_best_run_records(
        records,
        experiment="iwildcam_iro",
        algorithms=("iro",),
        selection_metric="val_acc_best",
        ckpt_kind="best",
    )
    cm = mod.select_best_run_records(
        records,
        experiment="cmnist_iro",
        algorithms=("iro",),
        selection_metric="0.9_acc_best",
        ckpt_kind="best",
    )

    assert len(iw) == 1 and iw[0].run_id == "run_iw"
    assert len(cm) == 1 and cm[0].run_id == "run_cm"

