#!/usr/bin/env bash
# Install the delivery's pinned dependencies into a dedicated local environment.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
delivery_dir=$(cd -- "$script_dir/../.." && pwd)
env_dir=${ODYN_LOCAL_ENV:-"$delivery_dir/.venv-local"}
uv_bin=${UV_BIN:-uv}
if ! command -v "$uv_bin" >/dev/null 2>&1; then
  echo 'uv is required; set UV_BIN to its executable path.' >&2
  exit 2
fi
if [[ ! -x "$env_dir/bin/python" ]]; then
  "$uv_bin" venv --python 3.12 "$env_dir"
fi
"$uv_bin" pip install --python "$env_dir/bin/python" "$delivery_dir[test]"
"$env_dir/bin/python" -m odyn_sft.run_management.cli doctor
