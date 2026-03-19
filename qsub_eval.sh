#!/bin/bash -l
#PBS -N kebiolm_eval_DDI
#PBS -l walltime=10:00:00
#PBS -l mem=64gb
#PBS -l ncpus=1
#PBS -l ngpus=1
#PBS -j eo
#PBS -m abe

echo '================================================'
echo "CWD = ${PBS_O_WORKDIR}"
echo '================================================'
cd "$PBS_O_WORKDIR"

echo '=========='
echo 'Activate conda env'
echo '=========='
# Robust conda activation across PBS environments
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kebiolm_py38

# Helpful runtime info (GPU status and library versions)
nvidia-smi || true
which python

echo '========'
echo 'Run eval'
echo '========'
cd re
python -u run.py \
  --task_name "DDI" \
  --data_dir "../data/DDI/" \
  --model_name_or_path "../model" \
  --output_dir "../evaluation_result/DDI" \
  --do_eval \
  --max_seq_length 256 \
  --per_device_eval_batch_size 16 \
  --overwrite_output_dir
  
echo 'Done.'
