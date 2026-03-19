#!/bin/bash -l

set -euo pipefail

echo '================================================'
echo 'Debug: environment'
echo '================================================'
echo "PWD = $(pwd)"
echo "CONDA_DEFAULT_ENV = ${CONDA_DEFAULT_ENV:-}"
echo "CUDA_HOME = ${CUDA_HOME:-}"
echo "CUDA_DIR = ${CUDA_DIR:-}"
echo "XLA_FLAGS = ${XLA_FLAGS:-}"
echo "TF_XLA_FLAGS = ${TF_XLA_FLAGS:-}"
echo "TF_FORCE_GPU_ALLOW_GROWTH = ${TF_FORCE_GPU_ALLOW_GROWTH:-}"

nvidia-smi || true
which python
python --version

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
    if torch.cuda.is_available():
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
