#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${IRO_DATA_ROOT:=data/iwildcam}"
: "${IRO_DOWNLOAD:=false}"
: "${SMOKE_STEPS:=2}"
: "${SMOKE_BATCH_SIZE:=8}"
: "${SMOKE_N_ENVS_PER_BATCH:=2}"

if [[ "${IRO_DOWNLOAD}" != "true" && ! -f "${IRO_DATA_ROOT}/iwildcam_v2.0/metadata.csv" ]]; then
  echo "ERROR: iWildCam dataset not found at ${IRO_DATA_ROOT}/iwildcam_v2.0 and IRO_DOWNLOAD=false." >&2
  echo "Set IRO_DOWNLOAD=true to let WILDS download the dataset, or provide an existing IRO_DATA_ROOT." >&2
  exit 1
fi

OUTPUT_ROOT="$(mktemp -d /tmp/iwildcam_smoke.XXXXXX)"
trap 'rm -rf "${OUTPUT_ROOT}"' EXIT

run_smoke() {
  local alg="$1"
  local exp_name="smoke_iwildcam_${alg}"

  echo "Running smoke for algorithm=${alg}"
  python -m iro train \
    --experiment iwildcam_iro \
    -o "data.root=${IRO_DATA_ROOT}" \
    -o "data.root_dir=${IRO_DATA_ROOT}" \
    -o "data.data_dir=${IRO_DATA_ROOT}" \
    -o "data.download=${IRO_DOWNLOAD}" \
    -o "data.debug_data=true" \
    -o "data.debug_train_size=256" \
    -o "data.debug_eval_size=128" \
    -o "data.n_envs_per_batch=${SMOKE_N_ENVS_PER_BATCH}" \
    -o "data.batch_size=${SMOKE_BATCH_SIZE}" \
    -o "iro.algorithm=${alg}" \
    -o "training.steps=${SMOKE_STEPS}" \
    -o "training.eval_freq=1" \
    -o "training.output_root=${OUTPUT_ROOT}" \
    -o "training.exp_name=${exp_name}" \
    -o "training.device=cpu" \
    -o "model.name=film_resnet18"

  local result_dir="${OUTPUT_ROOT}/results/${exp_name}"
  local jsonl
  jsonl="$(ls -1t "${result_dir}"/*.jsonl | head -n 1)"
  if [[ -z "${jsonl}" ]]; then
    echo "ERROR: no JSONL result was written for ${alg}." >&2
    exit 1
  fi

  python - "${jsonl}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not lines:
    raise SystemExit(f"No JSON records in {path}")
json.loads(lines[-1])
print(f"JSON parse OK: {path}")
PY
}

run_smoke erm
run_smoke iro
run_smoke groupdro

echo "iWildCam smoke checks completed."
