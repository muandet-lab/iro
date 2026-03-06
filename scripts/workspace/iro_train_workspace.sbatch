#!/bin/bash
#SBATCH --job-name=iro_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=/tmp/job-%j.out

set -euo pipefail

if ! command -v ws >/dev/null 2>&1; then
  echo "ERROR: 'ws' command not found. Run this on a workspace-enabled cluster host." >&2
  exit 1
fi

if ! klist -s; then
  echo "ERROR: no valid Kerberos ticket. Run 'kinit -r 7d -l 24h' before submitting jobs." >&2
  exit 1
fi

# Avoid bytecode writes into source trees on restricted filesystems.
export PYTHONDONTWRITEBYTECODE=1

: "${IRO_EXPERIMENT:=cmnist_iro}"
: "${IRO_REPO_ROOT:=$HOME/projects/iro}"
: "${IRO_DATA_ROOT:=$HOME/data/cmnist}"
: "${IRO_DOWNLOAD:=false}"
: "${IRO_WORKSPACE_SHARE:=work}"
: "${IRO_WORKSPACE_DURATION:=7 00:00:00}"
: "${IRO_EXP_NAME:=workspace_${IRO_EXPERIMENT}}"
: "${IRO_OVERRIDES_FILE:=}"
: "${IRO_EXTRA_OVERRIDES:=}"
: "${IRO_RESULTS_ARCHIVE_DIR:=}"
: "${IRO_ENV_ACTIVATE:=}"
: "${IRO_READ_ONLY_PREFIX:=}"

if [[ -z "${IRO_SEED:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  IRO_SEED="${SLURM_ARRAY_TASK_ID}"
fi

if [[ -n "${IRO_READ_ONLY_PREFIX}" && "${IRO_DATA_ROOT}" == "${IRO_READ_ONLY_PREFIX}"* ]]; then
  echo "ERROR: IRO_DATA_ROOT points to a configured read-only prefix: ${IRO_READ_ONLY_PREFIX}" >&2
  echo "Set IRO_DATA_ROOT to a writable shared path." >&2
  exit 1
fi

if [[ ! -d "${IRO_REPO_ROOT}" ]]; then
  echo "ERROR: IRO_REPO_ROOT not found: ${IRO_REPO_ROOT}" >&2
  exit 1
fi

JOBDATADIR="$(ws create "${IRO_WORKSPACE_SHARE}" --space "${SLURM_JOB_ID}" --duration "${IRO_WORKSPACE_DURATION}")"
JOBTMPDIR="/tmp/job-${SLURM_JOB_ID}"
mkdir -p "${JOBTMPDIR}" "${JOBDATADIR}"

sync_artifacts() {
  local status=$?
  if [[ -f "/tmp/job-${SLURM_JOB_ID}.out" ]]; then
    cp -f "/tmp/job-${SLURM_JOB_ID}.out" "${JOBDATADIR}/slurm.out" || true
  fi
  if [[ -d "${JOBTMPDIR}" ]]; then
    mkdir -p "${JOBDATADIR}/job_tmp"
    rsync -a "${JOBTMPDIR}/" "${JOBDATADIR}/job_tmp/" || true
  fi
  if [[ -n "${IRO_RESULTS_ARCHIVE_DIR}" ]]; then
    mkdir -p "${IRO_RESULTS_ARCHIVE_DIR}" || true
    if [[ -d "${JOBTMPDIR}/iro_exp/results/${IRO_EXP_NAME}" ]]; then
      find "${JOBTMPDIR}/iro_exp/results/${IRO_EXP_NAME}" -maxdepth 1 -type f -name '*.jsonl' -exec cp -f {} "${IRO_RESULTS_ARCHIVE_DIR}/" \; || true
    fi
  fi
  echo "Workspace artifacts: ${JOBDATADIR}"
  return ${status}
}
trap sync_artifacts EXIT

cd "${IRO_REPO_ROOT}"

if [[ -n "${IRO_ENV_ACTIVATE}" ]]; then
  if [[ ! -f "${IRO_ENV_ACTIVATE}" ]]; then
    echo "ERROR: IRO_ENV_ACTIVATE points to a missing file: ${IRO_ENV_ACTIVATE}" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${IRO_ENV_ACTIVATE}"
fi

cmd=(
  python -m iro train
  --experiment "${IRO_EXPERIMENT}"
  -o "data.root=${IRO_DATA_ROOT}"
  -o "data.download=${IRO_DOWNLOAD}"
  -o "training.output_root=${JOBTMPDIR}/iro_exp"
  -o "training.exp_name=${IRO_EXP_NAME}"
  -o "training.device=auto"
)

if [[ -n "${IRO_SEED:-}" ]]; then
  cmd+=( -o "training.seed=${IRO_SEED}" )
fi

if [[ -n "${IRO_OVERRIDES_FILE}" ]]; then
  if [[ ! -f "${IRO_OVERRIDES_FILE}" ]]; then
    echo "ERROR: IRO_OVERRIDES_FILE not found: ${IRO_OVERRIDES_FILE}" >&2
    exit 1
  fi
  while IFS= read -r override; do
    [[ -z "${override}" ]] && continue
    [[ "${override}" =~ ^# ]] && continue
    cmd+=( -o "${override}" )
  done < "${IRO_OVERRIDES_FILE}"
fi

if [[ -n "${IRO_EXTRA_OVERRIDES}" ]]; then
  IFS=';' read -r -a extra_overrides <<< "${IRO_EXTRA_OVERRIDES}"
  for raw_override in "${extra_overrides[@]}"; do
    override="$(echo "${raw_override}" | xargs)"
    [[ -z "${override}" ]] && continue
    [[ "${override}" =~ ^# ]] && continue
    cmd+=( -o "${override}" )
  done
fi

echo "Launching on host $(hostname)"
echo "SLURM job id: ${SLURM_JOB_ID}"
if [[ -n "${IRO_SEED:-}" ]]; then
  echo "Seed: ${IRO_SEED}"
fi
printf 'Command:'
printf ' %q' "${cmd[@]}"
echo

srun "${cmd[@]}"
