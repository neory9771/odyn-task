#!/usr/bin/env bash
# Explicit stages: dataset, doctor, prepare, probe, train, validate, validate-base, test, tensorboard.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
delivery_dir=$(cd -- "$script_dir/../.." && pwd)
repo_dir=$(cd -- "$delivery_dir/../.." && pwd)
default_python="$delivery_dir/.venv-local/bin/python"
if [[ ! -x "$default_python" ]]; then default_python="$repo_dir/.venv-sft-gpu/bin/python"; fi
python_bin=${ODYN_PYTHON:-"$default_python"}
config_path=${ODYN_SFT_CONFIG:-"$delivery_dir/configs/qwen-sft2-local.json"}
if [[ ! -x "$python_bin" ]]; then
  echo "Set ODYN_PYTHON to a Python environment with the delivery dependencies installed." >&2
  exit 2
fi
export PYTHONPATH="$delivery_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
command=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
  dataset)
    exec "$python_bin" -m odyn_sft.dataset_generation synthesis assemble \
      --teacher-dir "$repo_dir/runs/synthesis-targets-pilot2-20261007" \
      --source-suite "$repo_dir/runs/final-5000-1000-1000/suite" \
      --out "$delivery_dir/data/qwen-sft2-local" \
      --token-config "$config_path" --max-example-tokens 12000 "$@"
    ;;
  doctor)
    exec "$python_bin" -m odyn_sft.run_management.cli doctor "$@"
    ;;
  config|download|prepare|probe|train|validate|validate-base|test)
    exec "$python_bin" -m odyn_sft.run_management.cli "$command" --config "$config_path" "$@"
    ;;
  tensorboard)
    logs=$("$python_bin" -c 'import sys; from pathlib import Path; from odyn_sft.pytorch_sft.config import load_config; print(Path(load_config(sys.argv[1]).output))' "$config_path")
    exec "$python_bin" -m tensorboard.main --logdir "$logs" --host 127.0.0.1 --port 6006 "$@"
    ;;
  help)
    cat <<'HELP'
Usage: scripts/local/sft2.sh STAGE [arguments]
  dataset        Package complete teacher examples up to 12k tokens + 10 held-out draft cases
  doctor         Check the local CUDA/BF16/software environment
  download       Download pinned base model (only if it is absent from the cache)
  config         Print all resolved experiment settings
  prepare        Tokenize train/validation only; reject truncation and changed inputs
  probe          Discarded memory trial: --batch 1 --out PATH
  train          Fresh adapter or exact resume; temperature limit defaults to 90 C
  validate       Generate responses using selected SFT adapter on validation
  validate-base  Generate base Qwen responses with the same validation settings
  test           Held-out evaluation only after completed training
  tensorboard    Serve local training metrics at http://127.0.0.1:6006
Overrides: ODYN_PYTHON, ODYN_SFT_CONFIG, CUDA_VISIBLE_DEVICES.
Each stage is explicit; no background training or provider calls are started automatically.
HELP
    ;;
  *) echo "Unknown stage: $command" >&2; exit 2 ;;
esac
