# Usage

This page shows the CMNIST and iWildCam-WILDS workflows for `iro` from CLI and Python.

## Supported Experiment

- `cmnist_iro`
- `iwildcam_iro`

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

Run iWildCam training:

```bash
iro train --experiment iwildcam_iro \
  -o data.root=/path/to/iwildcam_root \
  -o data.download=false \
  -o iro.algorithm=iro \
  -o model.name=film_resnet50 \
  -o model.pretrained=true \
  -o data.iwildcam_image_size=448 \
  -o data.iwildcam_eval_resize=512 \
  -o training.steps=20000 \
  -o data.n_envs_per_batch=4 \
  -o data.iwildcam_eval_split=val,test,id_val,id_test
```

Run iWildCam in debug mode with small subsets:

```bash
iro train --experiment iwildcam_iro \
  -o data.root=/path/to/iwildcam_root \
  -o data.debug_data=true \
  -o training.steps=2 \
  -o iro.algorithm=groupdro
```

Evaluate an iWildCam checkpoint:

```bash
iro eval --experiment iwildcam_iro \
  -o eval.checkpoint_path=./iro_exp/ckpts/<run_id>_best.pkl \
  -o eval.split=all \
  -o eval.alpha=0.8
```

Collect Figure-2-style CVaR curves on iWildCam (ERM/GroupDRO/IRO):

```bash
python scripts/collect_cvar_curves.py \
  $HOME/CISPA-scratch/c01josh/iro/runs/iwildcam_long/results \
  --experiment iwildcam_iro \
  --data-root $HOME/CISPA-scratch/c01josh/datasets/iwildcam \
  --output-dir $HOME/CISPA-scratch/c01josh/iro/collected_results/iwildcam_cvar \
  --split val \
  --algorithms erm,groupdro,iro \
  --alpha-grid 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
```

The same collector is experiment-agnostic. Example for CMNIST:

```bash
python scripts/collect_cvar_curves.py \
  /path/to/cmnist/results \
  --experiment cmnist_iro \
  --data-root /path/to/cmnist_root \
  --split all \
  --algorithms erm,groupdro,iro
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
- `data.iwildcam_eval_split=val,test,id_val,id_test|all`
- `data.n_envs_per_batch=...`
- `data.uniform_over_groups=true|false`
- `data.debug_data=true|false`
- `data.iwildcam_image_size=224|448`
- `data.iwildcam_eval_resize=256|512`
- `model.name=film_resnet18|film_resnet50`
- `model.pretrained=true|false`

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
