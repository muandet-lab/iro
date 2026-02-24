#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${ARRAY_RANGE:=0-9}"
: "${DRY_RUN:=1}"
: "${IRO_REPO_ROOT:=${REPO_ROOT}}"
: "${IRO_DATA_ROOT:=$HOME/CISPA-projects/datasets/cmnist}"
: "${IRO_WORKSPACE_SHARE:=work}"
: "${IRO_WORKSPACE_DURATION:=7 00:00:00}"
: "${IRO_DOWNLOAD:=false}"
: "${IRO_EXPERIMENT:=cmnist_film_iro}"
: "${IRO_EXP_NAME_PREFIX:=cmnist_reproduce}"
: "${IRO_RESULTS_ARCHIVE_DIR:=$HOME/CISPA-work/$USER/iro_results/${IRO_EXP_NAME_PREFIX}}"

BASE_OVERRIDES_FILE="${REPO_ROOT}/scripts/cispa/overrides/cmnist_full_grid_base.txt"
SBATCH_SCRIPT="${REPO_ROOT}/scripts/cispa/iro_train_cispa.sbatch"

if [[ ! -f "${BASE_OVERRIDES_FILE}" ]]; then
  echo "ERROR: missing base overrides file ${BASE_OVERRIDES_FILE}" >&2
  exit 1
fi
if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "ERROR: missing sbatch script ${SBATCH_SCRIPT}" >&2
  exit 1
fi

export IRO_EXPERIMENT
export IRO_REPO_ROOT
export IRO_DATA_ROOT
export IRO_WORKSPACE_SHARE
export IRO_WORKSPACE_DURATION
export IRO_DOWNLOAD
export IRO_OVERRIDES_FILE="${BASE_OVERRIDES_FILE}"
export IRO_RESULTS_ARCHIVE_DIR

submit_count=0

submit_config() {
  local exp_name="$1"
  local extra_overrides="$2"

  local sbatch_cmd=(sbatch --array="${ARRAY_RANGE}")
  if [[ -n "${MAIL_USER:-}" ]]; then
    sbatch_cmd+=(--mail-user="${MAIL_USER}")
  fi
  sbatch_cmd+=("${SBATCH_SCRIPT}")

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN IRO_EXP_NAME=%q IRO_EXTRA_OVERRIDES=%q ' "${exp_name}" "${extra_overrides}"
    printf '%q ' "${sbatch_cmd[@]}"
    echo
  else
    IRO_EXP_NAME="${exp_name}" IRO_EXTRA_OVERRIDES="${extra_overrides}" "${sbatch_cmd[@]}"
  fi
  submit_count=$((submit_count + 1))
}

penalties=(1000 5000 10000 50000 100000)
sd_penalties=(1 5 10 50 100)
groupdro_etas=(0.001 0.01 0.1 0.5 1.0)
eqrm_alphas=(-100 -500 -1000 -5000 -10000)
erm_pretrain_steps=(0 400)
erm_total_steps=(400 600 1000)

# ERM and gray-ERM (oracle row in old scripts).
for train_env in default gray; do
  for steps in "${erm_total_steps[@]}"; do
    exp_name="${IRO_EXP_NAME_PREFIX}_erm_${train_env}_s${steps}"
    extra="iro.algorithm=erm;data.cmnist_train_envs=${train_env};training.steps=${steps};training.erm_pretrain_iters=0;training.lr_cos_sched=false;training.save_ckpts=false"
    submit_config "${exp_name}" "${extra}"
  done
done

# Non-ERM algorithms.
for pretrain in "${erm_pretrain_steps[@]}"; do
  for eta in "${groupdro_etas[@]}"; do
    exp_name="${IRO_EXP_NAME_PREFIX}_groupdro_pre${pretrain}_eta${eta}"
    extra="iro.algorithm=groupdro;training.steps=1000;training.erm_pretrain_iters=${pretrain};training.lr_cos_sched=true;training.save_ckpts=true;iro.groupdro_eta=${eta}"
    submit_config "${exp_name}" "${extra}"
  done

  for pen in "${sd_penalties[@]}"; do
    exp_name="${IRO_EXP_NAME_PREFIX}_sd_pre${pretrain}_pen${pen}"
    extra="iro.algorithm=sd;training.steps=1000;training.erm_pretrain_iters=${pretrain};training.lr_cos_sched=true;training.save_ckpts=true;iro.penalty_weight=${pen}"
    submit_config "${exp_name}" "${extra}"
  done

  for pen in "${penalties[@]}"; do
    exp_name="${IRO_EXP_NAME_PREFIX}_iga_pre${pretrain}_pen${pen}"
    extra="iro.algorithm=iga;training.steps=1000;training.erm_pretrain_iters=${pretrain};training.lr_cos_sched=true;training.save_ckpts=true;iro.penalty_weight=${pen}"
    submit_config "${exp_name}" "${extra}"
  done

  for pen in "${penalties[@]}"; do
    exp_name="${IRO_EXP_NAME_PREFIX}_irm_pre${pretrain}_pen${pen}"
    extra="iro.algorithm=irm;training.steps=600;training.erm_pretrain_iters=${pretrain};training.lr_cos_sched=true;training.save_ckpts=true;iro.penalty_weight=${pen}"
    submit_config "${exp_name}" "${extra}"
  done

  for pen in "${penalties[@]}"; do
    exp_name="${IRO_EXP_NAME_PREFIX}_vrex_pre${pretrain}_pen${pen}"
    extra="iro.algorithm=vrex;training.steps=600;training.erm_pretrain_iters=${pretrain};training.lr_cos_sched=true;training.save_ckpts=true;iro.penalty_weight=${pen}"
    submit_config "${exp_name}" "${extra}"
  done

  for alpha in "${eqrm_alphas[@]}"; do
    exp_name="${IRO_EXP_NAME_PREFIX}_eqrm_pre${pretrain}_a${alpha}"
    extra="iro.algorithm=eqrm;training.steps=600;training.erm_pretrain_iters=${pretrain};training.lr_cos_sched=true;training.save_ckpts=true;iro.alpha=${alpha}"
    submit_config "${exp_name}" "${extra}"
  done
done

echo "Total configuration arrays: ${submit_count}"
echo "Seeds per array: ${ARRAY_RANGE}"
echo "Total jobs: (number of indices in ARRAY_RANGE) x ${submit_count}"
echo "Result JSONL archive dir: ${IRO_RESULTS_ARCHIVE_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Set DRY_RUN=0 to submit."
fi
