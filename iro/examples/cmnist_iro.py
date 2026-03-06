"""CMNIST example using package-level training API.

Usage:
    python -m iro.examples.cmnist_iro
"""

from __future__ import annotations

from dataclasses import asdict

from iro.core import load_experiment_config
from iro.examples._io import create_run_dir, write_json
from iro.training import train_cmnist_iro


def main() -> None:
    cfg = load_experiment_config(experiment="cmnist_iro")
    try:
        result = train_cmnist_iro(cfg)
    except RuntimeError as exc:
        msg = str(exc)
        if "Error downloading" in msg or "Dataset not found" in msg:
            raise RuntimeError(
                "CMNIST data could not be prepared. Ensure network access for --download, or pre-download MNIST "
                f"into '{cfg.data.root}' and set data.download=false in config/experiments/cmnist_iro.yaml. "
                f"Original error: {msg}"
            ) from exc
        raise

    dataset = result.get("dataset")
    if dataset != "cmnist":
        raise RuntimeError(f"Expected CMNIST run but got dataset={dataset!r}.")

    run_dir = create_run_dir("outputs", "cmnist_iro")
    write_json(run_dir / "config.json", asdict(cfg))
    write_json(run_dir / "result.json", result)

    print(
        {
            "dataset": dataset,
            "device": result.get("device", "unknown"),
            "test_metrics": result.get("test_metrics"),
            "config_path": "config/experiments/cmnist_iro.yaml",
            "output_dir": str(run_dir),
        }
    )


if __name__ == "__main__":
    main()
