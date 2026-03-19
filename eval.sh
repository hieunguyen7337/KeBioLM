#!/bin/bash -l
#PBS -N kebiolm_eval_unified_ppi
#PBS -l walltime=10:00:00
#PBS -l mem=16gb
#PBS -l ncpus=1
#PBS -l ngpus=1
#PBS -j eo
#PBS -m abe

set -euo pipefail

ROOT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
MODEL_PATH="${MODEL_PATH:-model}"
DATA_DIR="${DATA_DIR:-data/Unified_PPI_binary}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation_result/Unified_PPI_binary}"
TASK_NAME="${TASK_NAME:-unified_ppi}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-kebiolm_py38}"
RUN_DEBUG="${RUN_DEBUG:-1}"

echo '================================================'
echo "CWD = ${ROOT_DIR}"
echo '================================================'
cd "${ROOT_DIR}"

echo '=========='
echo 'Load CUDA & cuDNN modules'
echo '=========='
if command -v module >/dev/null 2>&1; then
  module load CUDA/12.6.0 || true
  module load cuDNN/9.5.0.50-CUDA-12.6.0 || true
else
  echo 'module command not found; skipping module load'
fi

echo '=========='
echo 'Fix CUDA/XLA path issues'
echo '=========='
if [ -n "${CUDA_HOME:-}" ]; then
  export CUDA_DIR="${CUDA_HOME}"
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
else
  echo 'CUDA_HOME is not set; leaving CUDA_DIR and XLA_FLAGS unchanged'
fi
export TF_XLA_FLAGS="${TF_XLA_FLAGS:---tf_xla_auto_jit=0}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"

echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CUDA_DIR=${CUDA_DIR:-}"
echo "XLA_FLAGS=${XLA_FLAGS:-}"
echo "TF_XLA_FLAGS=${TF_XLA_FLAGS}"
echo "TF_FORCE_GPU_ALLOW_GROWTH=${TF_FORCE_GPU_ALLOW_GROWTH}"

echo '=========='
echo 'Activate conda env'
echo '=========='
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"

echo '=========='
echo 'Environment diagnostics'
echo '=========='
nvidia-smi || true
which python
python --version

if [ "${RUN_DEBUG}" = "1" ]; then
  echo '=========='
  echo 'Run debug scripts'
  echo '=========='
  bash debug_env.sh
  bash debug_unified_ppi.sh
fi

echo '========'
echo 'Run eval'
echo '========'
python -u -m relation_extraction.run \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --do_eval --do_predict \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --overwrite_output_dir \
  --overwrite_cache

echo 'Done.'
