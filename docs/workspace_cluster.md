# Workspace Cluster Runbook

This guide documents a workspace-backed SLURM workflow for `iro`.

## Why this setup

This setup is useful when your cluster provides:
- SLURM scheduling with explicit time limits
- temporary node-local storage (for fast I/O)
- workspace/shared storage (`ws`) for persisted artifacts
- Kerberos-backed access (`kinit`, `klist`)

## Included files

- `scripts/workspace/iro_train_workspace.sbatch`
- `scripts/workspace/submit_cmnist_full_grid.sh`
- `scripts/workspace/collect_cmnist_table.py`

## Prerequisites

1. Ensure Kerberos is valid:
```bash
kinit -r 7d -l 24h
klist
```
2. Ensure repo and datasets live on writable shared storage.

## Generic training template

```bash
IRO_REPO_ROOT=$HOME/projects/iro \
IRO_EXPERIMENT=cmnist_iro \
IRO_DATA_ROOT=$HOME/data/cmnist \
IRO_EXP_NAME=cmnist_single \
IRO_SEED=0 \
IRO_OVERRIDES_FILE=$HOME/projects/iro/scripts/overrides/cmnist_paper_repro.txt \
sbatch $HOME/projects/iro/scripts/workspace/iro_train_workspace.sbatch
```

What this template does:
- checks Kerberos validity (`klist -s`)
- allocates a workspace via `ws create`
- runs `python -m iro train ...` with `srun`
- writes run artifacts under `/tmp/job-<id>/iro_exp`
- syncs logs/results/checkpoints into the workspace on exit

## CMNIST full sweep

Preview:

```bash
cd /path/to/iro
DRY_RUN=1 \
ARRAY_RANGE=0-9 \
IRO_REPO_ROOT=$HOME/projects/iro \
IRO_DATA_ROOT=$HOME/data/cmnist \
MAIL_USER=<your_email> \
./scripts/workspace/submit_cmnist_full_grid.sh
```

Submit:

```bash
cd /path/to/iro
DRY_RUN=0 \
ARRAY_RANGE=0-9 \
IRO_REPO_ROOT=$HOME/projects/iro \
IRO_DATA_ROOT=$HOME/data/cmnist \
MAIL_USER=<your_email> \
IRO_EXP_NAME_PREFIX=cmnist_reproduce \
IRO_RESULTS_ARCHIVE_DIR=$HOME/results/iro/cmnist_reproduce \
./scripts/workspace/submit_cmnist_full_grid.sh
```

Collect table:

```bash
python scripts/workspace/collect_cmnist_table.py \
  $HOME/results/iro/cmnist_reproduce \
  --model-selection-env 0.9 \
  --model-selection-type best \
  --test-envs all
```

## Resource tuning

```bash
sbatch \
  --partition=gpu \
  --gres=gpu:A100:1 \
  --cpus-per-task=16 \
  --mem=96G \
  --time=24:00:00 \
  scripts/workspace/iro_train_workspace.sbatch
```

## Notes

- Defaults: `IRO_WORKSPACE_SHARE=work`, `IRO_WORKSPACE_DURATION=7 00:00:00`.
- For short smoke runs, use:
```bash
IRO_WORKSPACE_SHARE=scratch IRO_WORKSPACE_DURATION=2:00:00 \
sbatch scripts/workspace/iro_train_workspace.sbatch
```
- Optional read-only guard: set `IRO_READ_ONLY_PREFIX` to block dataset paths under a forbidden prefix.
