#!/usr/bin/env bash
# Train a fresh SFT2 adapter from the already selected research-only package.
set -euo pipefail
delivery_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$delivery_dir"
export PYTHONPATH="$delivery_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/dev/shm/odyn-mistral-model-cache}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python_bin=${ODYN_PYTHON:-/workspace/envs/hf/bin/python}
run_dir="$delivery_dir/runs/mistral-sft2-research-1000"
mkdir -p "$run_dir"
config="$run_dir/effective-config.json"
if [[ -e "$run_dir/train" || -e "$config" ]]; then
  echo 'Use a fresh run directory; this launcher does not overwrite or resume.' >&2
  exit 2
fi
"$python_bin" - "$delivery_dir" "$config" <<'PY'
import json, sys
from pathlib import Path
root, output = map(Path, sys.argv[1:])
data = json.loads((root / 'configs/mistral-sft2-bf16-research-1000-h200.json').read_text())
data['workspace'] = str(root)
output.write_text(json.dumps(data, indent=2) + '\n')
PY
"$python_bin" -m odyn_sft.run_management.cli prepare --config "$config"
"$python_bin" - <<'PY'
import subprocess, time
while True:
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(output) != 1:
        raise RuntimeError('Expected one visible GPU')
    memory, utilization = map(int, output[0].split(','))
    if memory < 2048 and utilization < 10:
        break
    print(f'Waiting for idle GPU: {memory} MiB, {utilization}%.', flush=True)
    time.sleep(30)
PY
if ! "$python_bin" -m odyn_sft.run_management.cli probe --config "$config" \
    --batch 2 --out "$run_dir/probe-batch2.json" --disable-temperature-checks; then
  "$python_bin" - "$config" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]); data = json.loads(path.read_text())
data['microbatch_size'] = 1
path.write_text(json.dumps(data, indent=2) + '\n')
PY
  "$python_bin" -m odyn_sft.run_management.cli probe --config "$config" \
    --batch 1 --out "$run_dir/probe-batch1.json" --disable-temperature-checks
fi
exec "$python_bin" -m odyn_sft.run_management.cli train --config "$config" --disable-temperature-checks
