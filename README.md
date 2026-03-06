# IRO: Imprecise Risk Optimization

`iro` is a Python library for Imprecise Risk Optimization (IRO), an optimization paradigm which allows users to maintain non-committal generalization preferences during training, providing users with flexibility of choice at test time.

Our framework, at its core, is built upon the ICML 2024 paper [Domain Generalisation via Imprecise Learning](https://arxiv.org/abs/2404.04669) by Anurag Singh, Siu Lun Chau, Shahine Bouabid, and Krikamol Muandet.
--------

## Scope

This repository currently supports two experiment routes:
- `cmnist_iro`
- `iwildcam_iro`

Supported interfaces:
- CLI: `iro train`, `iro eval`
- Python API: `run_training(...)`, `run_evaluation(...)`

## Installation

```bash
git clone https://github.com/muandet-lab/iro.git
cd iro
pip install -e .
```

## Quick Start

Run CMNIST training:

```bash
iro train --experiment cmnist_iro -o data.root=data/cmnist -o data.download=true -o training.steps=20
```

Evaluate a saved CMNIST checkpoint:

```bash
iro eval --experiment cmnist_iro \
  -o eval.checkpoint_path=./iro_exp/ckpts/<run_id>_final.pkl \
  -o eval.split=test \
  -o eval.alpha=0.8
```

Run iWildCam-WILDS training (WILDS dataset interface + grouped minibatches):

```bash
iro train --experiment iwildcam_iro \
  -o data.root=/path/to/iwildcam_root \
  -o data.download=false \
  -o iro.algorithm=iro \
  -o data.n_envs_per_batch=4 \
  -o data.iwildcam_eval_split=val,test,id_val,id_test
```

Evaluate a saved iWildCam checkpoint:

```bash
iro eval --experiment iwildcam_iro \
  -o eval.checkpoint_path=./iro_exp/ckpts/<run_id>_best.pkl \
  -o eval.split=all \
  -o eval.alpha=0.8
```

## Python API

```python
from iro import run_evaluation, run_training

train_result = run_training(
    experiment="cmnist_iro",
    overrides=["data.root=data/cmnist", "data.download=true", "training.steps=20"],
)

eval_result = run_evaluation(
    experiment="cmnist_iro",
    overrides=[
        "eval.checkpoint_path=./iro_exp/ckpts/<run_id>_final.pkl",
        "eval.split=all",
        "eval.alpha=0.8",
    ],
)
```

## Run Artifacts

By default, training writes:

- `./iro_exp/logs/<exp_name>/out.txt`
- `./iro_exp/logs/<exp_name>/err.txt`
- `./iro_exp/results/<exp_name>/<run_id>.jsonl`
- `./iro_exp/ckpts/<run_id>_final.pkl`
- `./iro_exp/ckpts/<run_id>_best.pkl`

## Documentation

- Usage: `docs/usage.md`
- Installation: `docs/installation.md`
- SLURM workflow: `docs/slurm_cluster.md`

## License

MIT
