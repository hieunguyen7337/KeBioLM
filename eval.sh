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

export ROOT_DIR
export MODEL_PATH
export DATA_DIR
export OUTPUT_DIR
export TASK_NAME
export MAX_SEQ_LENGTH
export EVAL_BATCH_SIZE
export CONDA_ENV_NAME
export RUN_DEBUG

echo '================================================'
echo "CWD = ${ROOT_DIR}"
echo '================================================'
cd "${ROOT_DIR}"

echo '=========='
echo 'Load CUDA & cuDNN modules'
echo '=========='
if command -v module >/dev/null 2>&1; then
  module load CUDA/12.6.0 >/dev/null 2>&1 || echo 'CUDA/12.6.0 module not found; continuing'
  module load cuDNN/9.5.0.50-CUDA-12.6.0 >/dev/null 2>&1 || echo 'cuDNN/9.5.0.50-CUDA-12.6.0 module not found; continuing'
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
  echo 'Debug: environment'
  echo '=========='
  echo "PWD = $(pwd)"
  echo "CONDA_DEFAULT_ENV = ${CONDA_DEFAULT_ENV:-}"
  echo "CUDA_HOME = ${CUDA_HOME:-}"
  echo "CUDA_DIR = ${CUDA_DIR:-}"
  echo "XLA_FLAGS = ${XLA_FLAGS:-}"
  echo "TF_XLA_FLAGS = ${TF_XLA_FLAGS:-}"
  echo "TF_FORCE_GPU_ALLOW_GROWTH = ${TF_FORCE_GPU_ALLOW_GROWTH:-}"

  python - <<'EOF'
import importlib

modules = ["torch", "transformers", "tensorflow", "sklearn"]
for name in modules:
    try:
        module = importlib.import_module(name)
        print(f"{name}: OK version={getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}: FAIL {exc}")

try:
    import torch
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print("torch.cuda.current_device:", torch.cuda.current_device())
        print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch CUDA probe failed:", exc)

try:
    import tensorflow as tf
    print("tensorflow GPUs:", tf.config.list_physical_devices("GPU"))
except Exception as exc:
    print("tensorflow GPU probe failed:", exc)
EOF

  echo '=========='
  echo 'Debug: Unified_PPI_binary dataset'
  echo '=========='
  echo "DATA_DIR = ${DATA_DIR}"
  echo "MODEL_PATH = ${MODEL_PATH}"
  echo "TASK_NAME = ${TASK_NAME}"
  echo "MAX_SEQ_LENGTH = ${MAX_SEQ_LENGTH}"

  python - <<'EOF'
import csv
import os
from collections import Counter
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
for split in ["train", "dev", "test"]:
    path = data_dir / f"{split}.tsv"
    label_counter = Counter()
    rows = 0
    bad_cols = 0
    bad_markers = 0

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            rows += 1
            if len(row) != 3:
                bad_cols += 1
                continue
            text = row[1]
            label_counter[row[2]] += 1
            if text.count("@E1$") != 1 or text.count("@E2$") != 1:
                bad_markers += 1

    print(path.as_posix())
    print("  rows:", rows)
    print("  labels:", dict(label_counter))
    print("  bad_cols:", bad_cols)
    print("  bad_markers:", bad_markers)

print("model_path_exists:", Path(os.environ["MODEL_PATH"]).exists())
EOF

  python - <<'EOF'
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"])
specials_before = len(tokenizer)
tokenizer.add_tokens(["@E1$", "@E2$"])
specials_after = len(tokenizer)
print("tokenizer_len_before:", specials_before)
print("tokenizer_len_after:", specials_after)
print("added_tokens:", specials_after - specials_before)
print("encode(@E1$):", tokenizer.encode("@E1$", add_special_tokens=False))
print("encode(@E2$):", tokenizer.encode("@E2$", add_special_tokens=False))
EOF

  python - <<'EOF'
import os
from transformers import AutoTokenizer
from relation_extraction.utils import RelationExtractionDataset, Split

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"])
dataset = RelationExtractionDataset(
    data_dir=os.environ["DATA_DIR"],
    tokenizer=tokenizer,
    task=os.environ["TASK_NAME"],
    max_seq_length=int(os.environ["MAX_SEQ_LENGTH"]),
    overwrite_cache=True,
    mode=Split.dev,
)
print("dev_dataset_len:", len(dataset))
first = dataset[0]
print("first_example_label:", first.label)
print("first_example_first_entity_position:", first.first_entity_position)
print("first_example_second_entity_position:", first.second_entity_position)
EOF
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
