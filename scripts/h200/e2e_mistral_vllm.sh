#!/usr/bin/env bash
# Mistral e2e via a vLLM server: sft1+sft2 adapters, then base+base, judged. LIMIT claims (default 1).
set -uo pipefail
R=/dev/shm/odyn-mistral-e2e
cd $R
export HF_HOME=/dev/shm/odyn-mistral-model-cache CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
LIMIT=${LIMIT:-1}; CONC=${CONC:-64}; PORT=${PORT:-8011}
MODEL=mistralai/Mistral-Small-24B-Instruct-2501; REV=9527884be6e5616bdd54de542f9ae13384489724
PACKAGE=/dev/shm/odyn-final-5000-1000-1000/suite
SFT1_ADAPTER=/dev/shm/odyn-mistral-1000-bf16/run/train/checkpoint-000112-cda6fdee928e
SFT2_ADAPTER=/dev/shm/odyn-mistral-sft2-research-1000/sft-oneepoch/runs/mistral-sft2-research-1000/train/checkpoint-000032-199db898c8b3
TAG=vllm-limit$LIMIT
(cd code && sha256sum -c --quiet CODE_SHA256) || { echo "CODE MISMATCH"; exit 2; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 2000 ] || { echo "GPU BUSY ${used}MiB"; exit 3; }
echo "== server start $(date -u +%FT%TZ)"
setsid /workspace/envs/vllm/bin/vllm serve $MODEL --revision $REV --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.92 --max-num-seqs 128 --enable-prefix-caching --seed 0 --generation-config vllm \
  --enable-lora --max-lora-rank 16 --max-loras 2 --lora-modules sft1=$SFT1_ADAPTER sft2=$SFT2_ADAPTER \
  --host 127.0.0.1 --port $PORT > server-$TAG.log 2>&1 &
SERVER=$!
trap 'kill -- -$SERVER 2>/dev/null; sleep 5; kill -9 -- -$SERVER 2>/dev/null' EXIT
for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "SERVER DIED"; tail -30 server-$TAG.log; exit 4; }; sleep 5; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "SERVER NOT READY"; exit 4; }
echo "== server ready $(date -u +%FT%TZ)"
export PYTHONPATH=$R/code/src
PY=/workspace/envs/hf/bin/python
common=(--sft1-config configs/mistral-sft1.json --sft2-config configs/mistral-sft2.json --package $PACKAGE
        --validation-report /dev/shm/odyn-e2e/inputs/judge-validation-report.json
        --credentials-file /dev/shm/odyn-e2e/private/credentials.json --judge-cache $R/.cache/judge
        --limit $LIMIT --workers 8 --engine vllm --vllm-url http://127.0.0.1:$PORT/v1 --concurrency $CONC)
status=0
echo "== sft-sft start $(date -u +%FT%TZ)"
$PY -u -m odyn_sft.inference.e2e "${common[@]}" --out runs/sft-sft-$TAG \
    --sft1-adapter $SFT1_ADAPTER --sft2-adapter $SFT2_ADAPTER; rc=$?; [ $rc -eq 0 ] || status=1
echo "== sft-sft exit $rc $(date -u +%FT%TZ)"
echo "== base-base start $(date -u +%FT%TZ)"
$PY -u -m odyn_sft.inference.e2e "${common[@]}" --out runs/base-base-$TAG; rc=$?; [ $rc -eq 0 ] || status=1
echo "== base-base exit $rc $(date -u +%FT%TZ)"
mkdir -p /workspace/odyn-mistral-e2e
cp -r runs configs code/CODE_SHA256 run_vllm_smoke.sh smoke-$TAG.log server-$TAG.log /workspace/odyn-mistral-e2e/ 2>/dev/null
echo "== ALL DONE status=$status $(date -u +%FT%TZ)"
