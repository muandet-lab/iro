#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${ARRAY_RANGE:=0-2}"
: "${DRY_RUN:=1}"
: "${IRO_REPO_ROOT:=${REPO_ROOT}}"
: "${IRO_DATA_ROOT:=$HOME/data/iwildcam}"
: "${IRO_DOWNLOAD:=false}"
: "${IRO_EXP_NAME_PREFIX:=iwildcam}"
: "${IRO_ALGORITHMS:=erm,iro,groupdro}"
: "${IRO_DEBUG_DATA:=1}"
: "${IRO_EVAL_SPLITS:=val,test,id_val,id_test}"
: "${IRO_N_ENVS_PER_BATCH:=4}"
: "${IRO_UNIFORM_OVER_GROUPS:=true}"
: "${IRO_STEPS:=20000}"
: "${IRO_BATCH_SIZE:=16}"
: "${IRO_MODEL_NAME:=film_resnet18}"
: "${IRO_MODEL_PRETRAINED:=true}"
: "${IRO_IMAGE_SIZE:=448}"
: "${IRO_EVAL_RESIZE:=512}"
: "${IRO_SLURM_PARTITION:=a100}"
: "${IRO_SLURM_GRES:=gpu:A100:1}"
: "${IRO_SLURM_CPUS_PER_TASK:=8}"
: "${IRO_SLURM_MEM:=64G}"
: "${IRO_SLURM_TIME:=12:00:00}"

SBATCH_SCRIPT="${REPO_ROOT}/scripts/iwildcam_train_slurm.sbatch"
if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "ERROR: missing sbatch script ${SBATCH_SCRIPT}" >&2
  exit 1
fi

IFS=',' read -r -a algorithms <<< "${IRO_ALGORITHMS}"
submit_count=0

for raw_alg in "${algorithms[@]}"; do
  alg="$(echo "${raw_alg}" | xargs | tr '[:upper:]' '[:lower:]')"
  [[ -z "${alg}" ]] && continue

  exp_name="${IRO_EXP_NAME_PREFIX}_${alg}"
  export IRO_REPO_ROOT
  export IRO_DATA_ROOT
  export IRO_DOWNLOAD
  export IRO_ALGORITHM="${alg}"
  export IRO_EXP_NAME="${exp_name}"
  export IRO_DEBUG_DATA
  export IRO_EVAL_SPLITS
  export IRO_N_ENVS_PER_BATCH
  export IRO_UNIFORM_OVER_GROUPS
  export IRO_STEPS
  export IRO_BATCH_SIZE
  export IRO_MODEL_NAME
  export IRO_MODEL_PRETRAINED
  export IRO_IMAGE_SIZE
  export IRO_EVAL_RESIZE

  sbatch_cmd=(
    sbatch
    --array="${ARRAY_RANGE}"
    --partition="${IRO_SLURM_PARTITION}"
    --cpus-per-task="${IRO_SLURM_CPUS_PER_TASK}"
    --mem="${IRO_SLURM_MEM}"
    --time="${IRO_SLURM_TIME}"
  )
  if [[ -n "${IRO_SLURM_GRES}" ]]; then
    sbatch_cmd+=(--gres="${IRO_SLURM_GRES}")
  fi
  if [[ -n "${MAIL_USER:-}" ]]; then
    sbatch_cmd+=(--mail-user="${MAIL_USER}")
  fi
  sbatch_cmd+=("${SBATCH_SCRIPT}")

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN IRO_ALGORITHM=%q IRO_EXP_NAME=%q IRO_DEBUG_DATA=%q ' "${alg}" "${exp_name}" "${IRO_DEBUG_DATA}"
    printf '%q ' "${sbatch_cmd[@]}"
    echo
  else
    "${sbatch_cmd[@]}"
  fi
  submit_count=$((submit_count + 1))
done

echo "Algorithms submitted: ${submit_count}"
echo "Array range: ${ARRAY_RANGE}"
echo "Data root: ${IRO_DATA_ROOT}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Set DRY_RUN=0 to submit."
fi
