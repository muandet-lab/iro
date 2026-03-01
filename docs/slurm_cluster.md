# SLURM Cluster Runbook

This guide documents the CMNIST reproduction workflow on a SLURM cluster.

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

## Generic Single Run

```bash
IRO_REPO_ROOT=$HOME/CISPA-projects/iro \
IRO_EXPERIMENT=cmnist_iro \
IRO_DATA_ROOT=$HOME/CISPA-projects/datasets/cmnist \
IRO_EXP_NAME=cmnist_single \
IRO_SEED=0 \
IRO_OVERRIDES_FILE=$HOME/CISPA-projects/iro/scripts/overrides/cmnist_paper_repro.txt \
sbatch $HOME/CISPA-projects/iro/scripts/iro_train_slurm.sbatch
```

## Paper-Style Seed Array (10 seeds)

```bash
cd /path/to/iro
ARRAY_RANGE=0-9 \
IRO_REPO_ROOT=$HOME/CISPA-projects/iro \
IRO_DATA_ROOT=$HOME/CISPA-projects/datasets/cmnist \
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
IRO_REPO_ROOT=$HOME/CISPA-projects/iro \
IRO_DATA_ROOT=$HOME/CISPA-projects/datasets/cmnist \
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
IRO_REPO_ROOT=$HOME/CISPA-projects/iro \
IRO_DATA_ROOT=$HOME/CISPA-projects/datasets/cmnist \
MAIL_USER=<your_email> \
IRO_EXP_NAME_PREFIX=cmnist_reproduce \
IRO_RESULTS_ARCHIVE_DIR=$HOME/CISPA-work/$USER/iro_results/cmnist_reproduce \
IRO_SLURM_PARTITION=a100 \
IRO_SLURM_GRES=gpu:A100:1 \
./scripts/submit_cmnist_full_grid.sh
```

## Collect Table

```bash
python scripts/collect_cmnist_table.py \
  $HOME/CISPA-work/$USER/iro_results/cmnist_reproduce \
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

Use `IRO_SLURM_PARTITION=r7525` with `IRO_SLURM_GRES=gpu:A100:1` to target the R7525 A100 nodes.
