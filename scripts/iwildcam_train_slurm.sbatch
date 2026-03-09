#!/bin/bash
#SBATCH --job-name=iwildcam_iro
#SBATCH --partition=a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=slurm_logs/iwildcam-%A_%a.out

set -euo pipefail

: "${IRO_REPO_ROOT:=$HOME/projects/iro}"
: "${IRO_DATA_ROOT:=$HOME/data/iwildcam}"
: "${IRO_DOWNLOAD:=false}"
: "${IRO_ALGORITHM:=iro}"
: "${IRO_DEBUG_DATA:=0}"
: "${IRO_EVAL_SPLITS:=val,test,id_val,id_test}"
: "${IRO_N_ENVS_PER_BATCH:=4}"
: "${IRO_UNIFORM_OVER_GROUPS:=true}"
: "${IRO_STEPS:=20000}"
: "${IRO_BATCH_SIZE:=16}"
: "${IRO_MODEL_NAME:=film_resnet18}"
: "${IRO_MODEL_PRETRAINED:=true}"
: "${IRO_IMAGE_SIZE:=448}"
: "${IRO_EVAL_RESIZE:=512}"
: "${IRO_EXP_NAME:=iwildcam_${IRO_ALGORITHM}}"

if [[ -z "${IRO_SEED:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  IRO_SEED="${SLURM_ARRAY_TASK_ID}"
fi

export IRO_EXPERIMENT="iwildcam_iro"
export IRO_REPO_ROOT
export IRO_DATA_ROOT
export IRO_DOWNLOAD
export IRO_EXP_NAME

base_overrides="iro.algorithm=${IRO_ALGORITHM};model.name=${IRO_MODEL_NAME};model.pretrained=${IRO_MODEL_PRETRAINED};training.steps=${IRO_STEPS};data.batch_size=${IRO_BATCH_SIZE};data.iwildcam_image_size=${IRO_IMAGE_SIZE};data.iwildcam_eval_resize=${IRO_EVAL_RESIZE};data.iwildcam_eval_split=${IRO_EVAL_SPLITS};data.n_envs_per_batch=${IRO_N_ENVS_PER_BATCH};data.uniform_over_groups=${IRO_UNIFORM_OVER_GROUPS};data.debug_data=${IRO_DEBUG_DATA};data.root_dir=${IRO_DATA_ROOT};data.data_dir=${IRO_DATA_ROOT}"
if [[ -n "${IRO_EXTRA_OVERRIDES:-}" ]]; then
  base_overrides="${base_overrides};${IRO_EXTRA_OVERRIDES}"
fi
export IRO_EXTRA_OVERRIDES="${base_overrides}"

bash "${IRO_REPO_ROOT}/scripts/iro_train_slurm.sbatch"
