# Usage

This page shows the CMNIST workflow for `iro` from CLI and Python.

## Supported Experiment

- `cmnist_iro`

## CLI Basics

Show command help:

```bash
iro --help
iro train --help
iro eval --help
```

Run CMNIST training:

```bash
iro train --experiment cmnist_iro \
  -o data.root=data/cmnist \
  -o data.download=true \
  -o training.steps=600
```

Evaluate a CMNIST checkpoint on configured test environments:

```bash
iro eval --experiment cmnist_iro \
  -o eval.checkpoint_path=./iro_exp/ckpts/<run_id>_final.pkl \
  -o eval.split=test \
  -o eval.alpha=0.8
```

Evaluate across all CMNIST shifts (`0.0..1.0`):

```bash
iro eval --experiment cmnist_iro \
  -o eval.checkpoint_path=./iro_exp/ckpts/<run_id>_final.pkl \
  -o eval.split=all \
  -o eval.alpha=0.8
```

## Common Overrides

- `training.steps=...`
- `training.seed=...`
- `iro.algorithm=erm|eqrm|irm|groupdro|vrex|iga|sd|iro|inftask|iid`
- `iro.penalty_weight=...`
- `iro.groupdro_eta=...`
- `iro.alpha=...`
- `data.cmnist_train_envs=...`
- `data.cmnist_test_envs=...`

## Python API

```python
from iro import run_evaluation, run_training

train_result = run_training(
    experiment="cmnist_iro",
    overrides=["data.root=data/cmnist", "data.download=true", "training.steps=600"],
)

eval_result = run_evaluation(
    experiment="cmnist_iro",
    overrides=[
        "eval.checkpoint_path=./iro_exp/ckpts/<run_id>_best.pkl",
        "eval.split=all",
        "eval.alpha=0.8",
    ],
)
```

## Output Artifacts

By default, training writes:

```text
iro_exp/
  logs/<exp_name>/out.txt
  logs/<exp_name>/err.txt
  results/<exp_name>/<run_id>.jsonl
  ckpts/<run_id>_final.pkl
  ckpts/<run_id>_best.pkl
```

Default config values:

- `training.output_root=./iro_exp`
- `training.exp_name=reproduce`
- `training.save_ckpts=true`
- `training.capture_logs=true`
- `training.write_artifacts=true`
