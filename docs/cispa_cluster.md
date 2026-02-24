# CISPA Cluster Runbook

This guide provides a policy-aligned way to run `iro` on CISPA CSS with SLURM.

## Why this setup

The CISPA CSS compute/storage guidance and examples emphasize:
- use SLURM with explicit time limits
- keep `~/CISPA-home` read-only on compute nodes (do not write job outputs there)
- use shared storage (`CISPA-work`, `CISPA-scratch`, `CISPA-projects`) for data/results
- keep Kerberos valid for shared filesystem access (`kinit`, `klist`)
- use the proper partitions/resources (`gpu` with A100 for GPU jobs)

This repository now includes templates that enforce those defaults.

## Files added

- `scripts/cispa/iro_train_cispa.sbatch`
- `scripts/cispa/overrides/cmnist_paper_repro.txt`
- `scripts/cispa/overrides/cmnist_full_grid_base.txt`
- `scripts/cispa/submit_cmnist_repro_array.sh`
- `scripts/cispa/submit_cmnist_full_grid.sh`
- `scripts/cispa/collect_cmnist_table.py`

## Prerequisites on CISPA

1. Authenticate and renewable Kerberos ticket:
```bash
kinit -r 7d -l 24h
klist
```
2. Ensure repo and dataset paths are on shared writable storage for job output/data.
   - recommended repo location: `~/CISPA-projects/...` or `~/CISPA-work/...`
   - do not write outputs to `~/CISPA-home` on compute nodes

## Generic training template

Submit one run with environment variables:

```bash
IRO_REPO_ROOT=$HOME/CISPA-projects/iro \
IRO_EXPERIMENT=cmnist_film_iro \
IRO_DATA_ROOT=$HOME/CISPA-projects/datasets/cmnist \
IRO_EXP_NAME=cmnist_single \
IRO_SEED=0 \
IRO_OVERRIDES_FILE=$HOME/CISPA-projects/iro/scripts/cispa/overrides/cmnist_paper_repro.txt \
sbatch $HOME/CISPA-projects/iro/scripts/cispa/iro_train_cispa.sbatch
```

What this template does:
- checks Kerberos validity (`klist -s`)
- creates a CISPA workspace via `ws create`
- runs `python -m iro train ...` via `srun`
- writes artifacts to `/tmp/job-<id>/iro_exp` during execution
- syncs logs/results/checkpoints from `/tmp` into the workspace on exit

## CMNIST paper-style array run (10 seeds)

```bash
cd /path/to/iro
ARRAY_RANGE=0-9 \
IRO_REPO_ROOT=$HOME/CISPA-projects/iro \
IRO_DATA_ROOT=$HOME/CISPA-projects/datasets/cmnist \
MAIL_USER=<your_cispa_email> \
./scripts/cispa/submit_cmnist_repro_array.sh
```

## Full multi-algorithm CMNIST sweep (Table-style reproduction)

Preview all array submissions without launching:

```bash
cd /path/to/iro
DRY_RUN=1 \
ARRAY_RANGE=0-9 \
IRO_REPO_ROOT=$HOME/CISPA-projects/iro \
IRO_DATA_ROOT=$HOME/CISPA-projects/datasets/cmnist \
MAIL_USER=<your_cispa_email> \
./scripts/cispa/submit_cmnist_full_grid.sh
```

Submit the full sweep:

```bash
cd /path/to/iro
DRY_RUN=0 \
ARRAY_RANGE=0-9 \
IRO_REPO_ROOT=$HOME/CISPA-projects/iro \
IRO_DATA_ROOT=$HOME/CISPA-projects/datasets/cmnist \
MAIL_USER=<your_cispa_email> \
IRO_EXP_NAME_PREFIX=cmnist_reproduce \
IRO_RESULTS_ARCHIVE_DIR=$HOME/CISPA-work/$USER/iro_results/cmnist_reproduce \
./scripts/cispa/submit_cmnist_full_grid.sh
```

Collect the table from run artifacts:

```bash
python scripts/cispa/collect_cmnist_table.py \
  $HOME/CISPA-work/$USER/iro_results/cmnist_reproduce \
  --model-selection-env 0.9 \
  --model-selection-type best \
  --test-envs all
```

## Tuning resources safely

You can override SLURM resources at submission time without editing the script:

```bash
sbatch \
  --partition=gpu \
  --gres=gpu:A100:1 \
  --cpus-per-task=16 \
  --mem=96G \
  --time=24:00:00 \
  scripts/cispa/iro_train_cispa.sbatch
```

Always keep a realistic `--time` to avoid forced TIMEOUT termination.

## Notes

- Default workspace share in the script is `work` and default workspace duration is `7 00:00:00`.
- You can switch to scratch for short-lived tests:
```bash
IRO_WORKSPACE_SHARE=scratch IRO_WORKSPACE_DURATION=2:00:00 sbatch scripts/cispa/iro_train_cispa.sbatch
```
- If you need additional experiment overrides, add them line-by-line to an overrides file and point `IRO_OVERRIDES_FILE` to it.
