# SLURM Cluster Runbook

This guide documents CMNIST and iWildCam workflows on a SLURM cluster.

## Why This Setup

Cluster workflows usually require:
- explicit SLURM resource limits and walltime
- writable shared storage for datasets/results
- short-lived local scratch during runs
- job-array orchestration for seed sweeps

## Included Files

- `scripts/iro_train_slurm.sbatch`
- `scripts/overrides/cmnist_paper_repro.txt`
- `scripts/overrides/cmnist_full_grid_base.txt`
- `scripts/submit_cmnist_repro_array.sh`
- `scripts/submit_cmnist_full_grid.sh`
- `scripts/collect_cmnist_table.py`
- `scripts/iwildcam_train_slurm.sbatch`
- `scripts/submit_iwildcam.sh`
- `scripts/smoke_iwildcam.sh`

## Generic Single Run

```bash
IRO_REPO_ROOT=$HOME/projects/iro \
IRO_EXPERIMENT=cmnist_iro \
IRO_DATA_ROOT=$HOME/data/cmnist \
IRO_EXP_NAME=cmnist_single \
IRO_SEED=0 \
IRO_OVERRIDES_FILE=$HOME/projects/iro/scripts/overrides/cmnist_paper_repro.txt \
sbatch $HOME/projects/iro/scripts/iro_train_slurm.sbatch
```

## Paper-Style Seed Array (10 seeds)

```bash
cd /path/to/iro
ARRAY_RANGE=0-9 \
IRO_REPO_ROOT=$HOME/projects/iro \
IRO_DATA_ROOT=$HOME/data/cmnist \
MAIL_USER=<your_email> \
IRO_SLURM_PARTITION=a100 \
IRO_SLURM_GRES=gpu:A100:1 \
./scripts/submit_cmnist_repro_array.sh
```

## Full Multi-Algorithm Sweep (All 10 Algorithms + Oracle)

Preview submissions:

```bash
cd /path/to/iro
DRY_RUN=1 \
ARRAY_RANGE=0-9 \
IRO_REPO_ROOT=$HOME/projects/iro \
IRO_DATA_ROOT=$HOME/data/cmnist \
MAIL_USER=<your_email> \
IRO_SLURM_PARTITION=a100 \
IRO_SLURM_GRES=gpu:A100:1 \
./scripts/submit_cmnist_full_grid.sh
```

Expected preview summary:
- `Total configuration arrays: 84`

Submit sweep:

```bash
cd /path/to/iro
DRY_RUN=0 \
ARRAY_RANGE=0-9 \
IRO_REPO_ROOT=$HOME/projects/iro \
IRO_DATA_ROOT=$HOME/data/cmnist \
MAIL_USER=<your_email> \
IRO_EXP_NAME_PREFIX=cmnist_reproduce \
IRO_RESULTS_ARCHIVE_DIR=$HOME/results/iro/cmnist_reproduce \
IRO_SLURM_PARTITION=a100 \
IRO_SLURM_GRES=gpu:A100:1 \
./scripts/submit_cmnist_full_grid.sh
```

## Collect Table

```bash
python scripts/collect_cmnist_table.py \
  $HOME/results/iro/cmnist_reproduce \
  --model-selection-env 0.9 \
  --model-selection-type best \
  --test-envs all
```

## Resource Overrides

```bash
sbatch \
  --partition=a100 \
  --gres=gpu:A100:1 \
  --cpus-per-task=16 \
  --mem=96G \
  --time=24:00:00 \
  scripts/iro_train_slurm.sbatch
```

## iWildCam submit helper (ERM/IRO/GroupDRO)

Preview the submissions:

```bash
cd /path/to/iro
DRY_RUN=1 \
ARRAY_RANGE=0-2 \
IRO_REPO_ROOT=$HOME/projects/iro \
IRO_DATA_ROOT=$HOME/data/iwildcam \
IRO_DEBUG_DATA=1 \
./scripts/submit_iwildcam.sh
```

Submit:

```bash
cd /path/to/iro
DRY_RUN=0 \
ARRAY_RANGE=0-9 \
IRO_REPO_ROOT=$HOME/projects/iro \
IRO_DATA_ROOT=$HOME/data/iwildcam \
IRO_DEBUG_DATA=0 \
./scripts/submit_iwildcam.sh
```

## iWildCam smoke check

```bash
cd /path/to/iro
IRO_DATA_ROOT=$HOME/data/iwildcam \
IRO_DOWNLOAD=false \
./scripts/smoke_iwildcam.sh
```
