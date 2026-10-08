#!/usr/bin/env bash
# Frozen Qwen E2E comparison: 5k-run SFT1 checkpoint with fixed pilot SFT2, with the same judge as Mistral.
set -uo pipefail
R=${RUN_ROOT:-/dev/shm/odyn-qwen-5k-e2e}
cd "$R" || exit 1
export HF_HOME=/workspace/.hf_home CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
LIMIT=${LIMIT:-1000}; CONC=${CONC:-64}; PORT=${PORT:-8012}
MODEL=Qwen/Qwen3-4B
REV=1cfa9a7208912126459214e8b04321603b3df60c
PACKAGE=/dev/shm/odyn-final-5000-1000-1000/suite
SFT1_ADAPTER=/workspace/odyn-task/runs/sft-h200-final-5000-1000-1000/train/checkpoint-000132-45289da7702b
SFT2_ADAPTER=/dev/shm/odyn-e2e/runs/qwen-sft2/train/checkpoint-000072-73f848fee9d5
JUDGE_CACHE=${JUDGE_CACHE:-/dev/shm/odyn-mistral-e2e/.cache/judge}
TAG=vllm-limit$LIMIT
(cd code && sha256sum -c --quiet CODE_SHA256) || { echo "CODE MISMATCH"; exit 2; }
sha256sum -c --quiet RUN_INPUT_SHA256 || { echo "RUN INPUT MISMATCH"; exit 2; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 2000 ] || { echo "GPU BUSY ${used}MiB"; exit 3; }
echo "== server start $(date -u +%FT%TZ)"
setsid /workspace/envs/vllm/bin/vllm serve "$MODEL" --revision "$REV" --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.92 --max-num-seqs 128 --enable-prefix-caching --seed 0 --generation-config vllm \
  --enable-lora --max-lora-rank 16 --max-loras 2 --lora-modules "sft1=$SFT1_ADAPTER" "sft2=$SFT2_ADAPTER" \
  --host 127.0.0.1 --port "$PORT" > "server-$TAG.log" 2>&1 &
SERVER=$!
cleanup() {
  kill -- -"$SERVER" 2>/dev/null || true
  sleep 5
  kill -9 -- -"$SERVER" 2>/dev/null || true
  mkdir -p /workspace/odyn-qwen-5k-e2e
  for item in runs configs CODE_SHA256 RUN_INPUT_SHA256 qwen_runs_inventory.json qwen_run_plan.json run_vllm.sh "run-$TAG.log" "server-$TAG.log"; do
    [ ! -e "$item" ] || cp -a "$item" /workspace/odyn-qwen-5k-e2e/
  done
}
trap cleanup EXIT
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 "$SERVER" 2>/dev/null || { echo "SERVER DIED"; tail -30 "server-$TAG.log"; exit 4; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "SERVER NOT READY"; exit 4; }
echo "== server ready $(date -u +%FT%TZ)"
export PYTHONPATH=$R/code/src
PY=/workspace/envs/hf/bin/python
common=(--sft1-config configs/qwen-sft1.json --sft2-config configs/qwen-sft2.json --package "$PACKAGE"
        --validation-report /dev/shm/odyn-e2e/inputs/judge-validation-report.json
        --credentials-file /dev/shm/odyn-e2e/private/credentials.json --judge-cache "$JUDGE_CACHE"
        --limit "$LIMIT" --workers 8 --engine vllm --vllm-url "http://127.0.0.1:$PORT/v1" --concurrency "$CONC"
        --disable-temperature-checks)
status=0
echo "== sft-sft start $(date -u +%FT%TZ)"
"$PY" -u -m odyn_sft.inference.e2e "${common[@]}" --out "runs/sft-sft-$TAG" \
  --sft1-adapter "$SFT1_ADAPTER" --sft2-adapter "$SFT2_ADAPTER"
rc=$?; [ "$rc" -eq 0 ] || status=1
echo "== sft-sft exit $rc $(date -u +%FT%TZ)"
echo "== ALL DONE status=$status $(date -u +%FT%TZ)"
exit "$status"
