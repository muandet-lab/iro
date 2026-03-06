#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${ARRAY_RANGE:=0-9}"
: "${IRO_REPO_ROOT:=${REPO_ROOT}}"
: "${IRO_DATA_ROOT:=$HOME/data/cmnist}"
: "${IRO_WORKSPACE_SHARE:=work}"
: "${IRO_WORKSPACE_DURATION:=7 00:00:00}"
: "${IRO_DOWNLOAD:=false}"
: "${IRO_EXP_NAME:=cmnist_paper_repro}"
: "${IRO_SLURM_PARTITION:=a100}"
: "${IRO_SLURM_GRES:=gpu:A100:1}"
: "${IRO_SLURM_CPUS_PER_TASK:=8}"
: "${IRO_SLURM_MEM:=64G}"
: "${IRO_SLURM_TIME:=12:00:00}"

OVERRIDES_FILE="${REPO_ROOT}/scripts/overrides/cmnist_paper_repro.txt"
SBATCH_SCRIPT="${REPO_ROOT}/scripts/iro_train_slurm.sbatch"

export IRO_EXPERIMENT=cmnist_iro
export IRO_REPO_ROOT
export IRO_DATA_ROOT
export IRO_WORKSPACE_SHARE
export IRO_WORKSPACE_DURATION
export IRO_DOWNLOAD
export IRO_EXP_NAME
export IRO_OVERRIDES_FILE="${OVERRIDES_FILE}"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "ERROR: missing sbatch script ${SBATCH_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${OVERRIDES_FILE}" ]]; then
  echo "ERROR: missing overrides file ${OVERRIDES_FILE}" >&2
  exit 1
fi

echo "Submitting array ${ARRAY_RANGE} with overrides ${OVERRIDES_FILE}"
sbatch_args=(
  --array="${ARRAY_RANGE}"
  --partition="${IRO_SLURM_PARTITION}"
  --cpus-per-task="${IRO_SLURM_CPUS_PER_TASK}"
  --mem="${IRO_SLURM_MEM}"
  --time="${IRO_SLURM_TIME}"
)
if [[ -n "${IRO_SLURM_GRES}" ]]; then
  sbatch_args+=(--gres="${IRO_SLURM_GRES}")
fi
if [[ -n "${MAIL_USER:-}" ]]; then
  sbatch_args+=(--mail-user="${MAIL_USER}")
fi
sbatch "${sbatch_args[@]}" "${SBATCH_SCRIPT}"
