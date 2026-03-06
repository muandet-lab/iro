from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def test_full_grid_dry_run_includes_all_algorithms_and_oracle() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "submit_cmnist_full_grid.sh"

    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["ARRAY_RANGE"] = "0-0"
    env["IRO_REPO_ROOT"] = str(repo_root)
    env["IRO_DATA_ROOT"] = "/tmp/cmnist"

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout

    assert "Total configuration arrays: 84" in out
    assert "data.cmnist_train_envs=gray" in out

    found = set(re.findall(r"iro\.algorithm=([a-z]+)", out))
    expected = {"erm", "eqrm", "irm", "groupdro", "vrex", "iga", "sd", "iro", "inftask", "iid"}
    assert expected.issubset(found)


def test_submit_iwildcam_dry_run_lists_required_algorithms() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "submit_iwildcam.sh"

    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["ARRAY_RANGE"] = "0-0"
    env["IRO_REPO_ROOT"] = str(repo_root)
    env["IRO_DATA_ROOT"] = "/tmp/iwildcam"
    env["IRO_ALGORITHMS"] = "erm,iro,groupdro"

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout
    assert "IRO_ALGORITHM=erm" in out
    assert "IRO_ALGORITHM=iro" in out
    assert "IRO_ALGORITHM=groupdro" in out
    assert "slurm/iwildcam.sbatch" in out


def test_smoke_iwildcam_fails_when_dataset_missing_and_no_download() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "smoke_iwildcam.sh"

    env = os.environ.copy()
    env["IRO_DATA_ROOT"] = "/tmp/definitely_missing_iwildcam_dataset"
    env["IRO_DOWNLOAD"] = "false"

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "dataset not found" in proc.stderr.lower()
