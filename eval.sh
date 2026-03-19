#!/bin/bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-model}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

run_eval() {
  local task_name="$1"
  local data_dir="$2"
  local output_dir="$3"
  local max_seq_length="$4"

  echo '========'
  echo "Run eval: ${task_name}"
  echo '========'
  python -u -m relation_extraction.run \
    --task_name "${task_name}" \
    --data_dir "${data_dir}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${output_dir}" \
    --do_eval --do_predict \
    --max_seq_length "${max_seq_length}" \
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
    --overwrite_output_dir \
    --overwrite_cache
}

echo '================================================'
echo "MODEL_PATH = ${MODEL_PATH}"
echo "EVAL_BATCH_SIZE = ${EVAL_BATCH_SIZE}"
echo '================================================'
echo 'Datasets prepared for KeBioLM eval:'
echo '  1. Unified_PPI binary'
echo '  2. Phos full-label'
echo '  3. Phos binary'
echo 'Note: each folder mirrors the same converted file into train/dev/test'
echo 'so eval-only runs can build a label map without changing relation_extraction.run.'

run_eval "unified_ppi" "data/Unified_PPI_binary" "evaluation_result/Unified_PPI_binary" 256
run_eval "phos_full" "data/Phos_full" "evaluation_result/Phos_full" 256
run_eval "phos_binary" "data/Phos_binary" "evaluation_result/Phos_binary" 256

echo 'Done.'
