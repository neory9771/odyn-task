#!/usr/bin/env bash
# Export/prepare on CPU, wait for an idle GPU, probe batch two, then train fresh.
set -euo pipefail
delivery_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$delivery_dir"
export PYTHONPATH="$delivery_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/dev/shm/odyn-mistral-model-cache}
python_bin=${ODYN_PYTHON:-/workspace/envs/hf/bin/python}
source_suite=${ODYN_SOURCE_SUITE:-/dev/shm/odyn-final-5000-1000-1000/suite}
run_dir="$delivery_dir/runs/mistral-sft2-5000"
mkdir -p "$run_dir"
config="$run_dir/effective-config.json"
if [[ -e "$run_dir/train" || -e "$config" ]]; then
  echo 'Fresh launch requires a new run directory; existing state will not be overwritten.' >&2
  exit 2
fi
"$python_bin" - "$delivery_dir" "$config" <<'PY'
import json,sys
from pathlib import Path
root, output = map(Path,sys.argv[1:])
data=json.loads((root/'configs/mistral-sft2-bf16-5000-h200.json').read_text())
data['workspace']=str(root)
output.write_text(json.dumps(data,indent=2)+'\n')
PY
if [[ ! -d data/synthesis-suite-5000 ]]; then
  "$python_bin" -m odyn_sft.dataset_generation synthesis export \
    --source-suite "$source_suite" --out data/synthesis-suite-5000
fi
"$python_bin" -m odyn_sft.dataset_generation synthesis validate --suite data/synthesis-suite-5000
"$python_bin" -m odyn_sft.run_management.cli prepare --config "$config"
# Visible foreign workloads also count as occupied; never kill another job.
"$python_bin" - <<'PY'
import subprocess,time
while True:
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(lines)!=1:
        raise RuntimeError('Expected exactly one visible GPU')
    memory,utilization=map(int,lines[0].split(','))
    if memory<2048 and utilization<10:
        print('GPU idle; starting memory probe.',flush=True)
        break
    print(f'Queued: GPU occupied ({memory} MiB, {utilization}% utilization).',flush=True)
    time.sleep(30)
PY
if ! "$python_bin" -m odyn_sft.run_management.cli probe --config "$config" \
  --batch 2 --out "$run_dir/probe-batch2.json" --disable-temperature-checks; then
  echo 'Batch-two probe failed; checking conservative batch one.'
  "$python_bin" - "$config" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text()); d['microbatch_size']=1
p.write_text(json.dumps(d,indent=2)+'\n')
PY
  "$python_bin" -m odyn_sft.run_management.cli probe --config "$config" \
    --batch 1 --out "$run_dir/probe-batch1.json" --disable-temperature-checks
fi
exec "$python_bin" -m odyn_sft.run_management.cli train --config "$config" --disable-temperature-checks
