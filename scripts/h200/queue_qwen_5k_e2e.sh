#!/usr/bin/env bash
# Wait for this previous Qwen comparison and its GPU cleanup; never stop another job.
set -euo pipefail
R=${RUN_ROOT:-/dev/shm/odyn-qwen-5k-e2e}
PREVIOUS_ROOT=${PREVIOUS_ROOT:-/dev/shm/odyn-qwen-e2e}
PREVIOUS_PID=$(cat "$PREVIOUS_ROOT/queue.pid")
cd "$R"
state() {
  printf '{"state":"%s","previous_pid":%s,"updated_at_utc":"%s"}\n' \
    "$1" "$PREVIOUS_PID" "$(date -u +%FT%TZ)" > queue_state.json
}
state waiting_for_previous
echo "Waiting for previous Qwen PID $PREVIOUS_PID and GPU cleanup."
while :; do
  process_state=$(ps -p "$PREVIOUS_PID" -o stat= || true)
  case "$process_state" in ""|Z*) break ;; esac
  sleep 15
done
if ! grep -q '^== ALL DONE status=0 ' "$PREVIOUS_ROOT/run-vllm-limit1000.log"; then
  state blocked_previous_not_completed
  echo "previous Qwen did not finish both chains successfully; Qwen remains unstarted."
  exit 1
fi
for attempt in $(seq 1 12); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -ge 2000 ] || break
  sleep 10
done
if [ "$used" -ge 2000 ]; then
  state blocked_gpu_busy
  echo "GPU remains occupied (${used}MiB); no process was stopped."
  exit 1
fi
state running
echo "Starting queued Qwen step-132 comparison $(date -u +%FT%TZ)."
set +e
LIMIT=${LIMIT:-1000} CONC=${CONC:-64} bash run_vllm.sh > run-vllm-limit${LIMIT:-1000}.log 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then state completed; else state failed; fi
exit "$rc"
