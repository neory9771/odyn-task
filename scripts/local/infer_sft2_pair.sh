#!/usr/bin/env bash
# Sequential processes release all base-model VRAM before loading the adapter run.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
delivery_dir=$(cd -- "$script_dir/../.." && pwd)
repo_dir=$(cd -- "$delivery_dir/../.." && pwd)
default_python="$delivery_dir/.venv-local/bin/python"
if [[ ! -x "$default_python" ]]; then default_python="$repo_dir/.venv-sft-gpu/bin/python"; fi
python_bin=${ODYN_PYTHON:-"$default_python"}
config=${ODYN_SFT_CONFIG:-"$delivery_dir/configs/qwen-sft2-local.json"}
input=${1:-"$delivery_dir/data/qwen-sft2-local/validation_synthesis.jsonl"}
output_dir=${2:-"$delivery_dir/runs/qwen-sft2-inference-validation"}
adapter=${3:-"$delivery_dir/runs/qwen-sft2-local-short/train/best_adapter"}
if [[ $# -gt 3 ]]; then echo 'Usage: infer_sft2_pair.sh [INPUT_JSONL [OUTPUT_DIR [ADAPTER_DIR]]]' >&2; exit 2; fi
if [[ -e "$output_dir" ]]; then echo 'Use a new output directory.' >&2; exit 2; fi
mkdir -p "$output_dir"
export PYTHONPATH="$delivery_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
"$python_bin" -m odyn_sft.inference --config "$config" --input "$input" --out "$output_dir/base.jsonl"
"$python_bin" -m odyn_sft.inference --config "$config" --input "$input" \
  --adapter "$adapter" --out "$output_dir/sft2.jsonl"
"$python_bin" -m odyn_sft.inference.compare --base "$output_dir/base.jsonl" \
  --sft "$output_dir/sft2.jsonl" --out "$output_dir/comparison.json"
