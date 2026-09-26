#!/usr/bin/env bash
# Source manually before future work. Sourcing does not start a job.
RECAP_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$RECAP_PROJECT_ROOT/.venv/bin:$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME="$RECAP_PROJECT_ROOT/.cache/openpi"
export HF_HOME="$RECAP_PROJECT_ROOT/.cache/huggingface"
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
