#!/bin/bash -l
#PBS -N kebiolm_eval_custom_re
#PBS -l walltime=10:00:00
#PBS -l mem=64gb
#PBS -l ncpus=1
#PBS -l ngpus=1
#PBS -j eo
#PBS -m abe

set -euo pipefail

echo '================================================'
echo "CWD = ${PBS_O_WORKDIR}"
echo '================================================'
cd "$PBS_O_WORKDIR"

echo '=========='
echo 'Activate conda env'
echo '=========='
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kebiolm_py38

nvidia-smi || true
which python

export MODEL_PATH="${MODEL_PATH:-model}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

echo '========'
echo 'Run eval suite'
echo '========'
bash eval.sh

echo 'Done.'
