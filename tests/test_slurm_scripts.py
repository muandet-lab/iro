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
