"""GPU telemetry and cooperative termination for one owned worker process."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ..pytorch_sft.config import Config


def gpu_selector() -> str:
    """Select the physical GPU visible to CUDA, including UUID-based selection."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if len(visible) != 1 or not visible[0].strip() or visible[0].strip() == "-1":
        raise ValueError("Expose one GPU using CUDA_VISIBLE_DEVICES=0 (or its GPU UUID)")
    return visible[0].strip()


def read_gpu(selector: str) -> dict[str, float]:
    raw = subprocess.check_output(
        ["nvidia-smi", "-i", selector,
         "--query-gpu=temperature.gpu,memory.used,utilization.gpu,power.draw",
         "--format=csv,noheader,nounits"], text=True, timeout=10)
    temp, used, util, power = map(float, raw.strip().split(","))
    return dict(timestamp=time.time(), temperature_c=temp, used_mib=used,
                utilization_pct=util, power_w=power)


def run_supervised(command: str, config_path: Path, config: Config, *,
                   disable_temperature_checks: bool = False, max_temperature: float = 90,
                   interrupt_after_updates: int | None = None, batch: int | None = None,
                   out: str | None = None) -> int:
    """Forward signals, allow checkpoint cleanup, and never stop unrelated GPU tasks."""
    import torch
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("Expose exactly one BF16-capable GPU with CUDA_VISIBLE_DEVICES")
    selector = gpu_selector()
    path = Path(config.thermal_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = dict(temperature_checks_enabled=not disable_temperature_checks,
                    max_temperature_c=None if disable_temperature_checks else max_temperature,
                    sample_interval_seconds=2, telemetry_enabled=True,
                    sensor_failure_stops_training=not disable_temperature_checks,
                    gpu_selector=selector, cooling_mode="cooperative_stop")
    (path.parent / "thermal_settings.json").write_text(json.dumps(settings, indent=2))
    env = {**os.environ, "ODYN_GPU_SUPERVISED": "1",
           "ODYN_SUPERVISOR_SOURCE": str(Path(__file__).resolve())}
    if command == "probe":
        argv = [sys.executable, "-u", "-m", "odyn_sft.run_management.probe", "--config", str(config_path),
                "--batch", str(batch), "--out", str(Path(out).resolve())]
    else:
        argv = [sys.executable, "-u", "-m", "odyn_sft.pytorch_sft.cli", command,
                "--config", str(config_path)]
    if interrupt_after_updates is not None:
        argv += ["--interrupt-after-updates", str(interrupt_after_updates)]
    worker = subprocess.Popen(argv, env=env)
    stop_at: float | None = None

    def stop(*_: object) -> None:
        nonlocal stop_at
        if worker.poll() is None and stop_at is None:
            worker.send_signal(signal.SIGTERM)
            stop_at = time.monotonic()

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with path.open("a", buffering=1) as log:
            while worker.poll() is None:
                try:
                    sample = read_gpu(selector)
                    log.write(json.dumps(sample) + "\n")
                    if not disable_temperature_checks and sample["temperature_c"] >= max_temperature:
                        print("Temperature limit reached; requesting a checkpoint stop", flush=True)
                        stop()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    log.write(json.dumps({"timestamp": time.time(), "telemetry_error": str(error)}) + "\n")
                    if not disable_temperature_checks:
                        stop()
                if stop_at is not None and time.monotonic() - stop_at > 120:
                    worker.kill()
                time.sleep(2)
        return worker.wait()
    finally:
        if worker.poll() is None:
            stop()
            try:
                worker.wait(timeout=120)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
