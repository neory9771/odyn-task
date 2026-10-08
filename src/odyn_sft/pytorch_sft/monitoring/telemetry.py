"""Optional W&B publication with a stable identity across checkpoint resumes."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


class WandbMetrics:
    def __init__(self, config: Any, manifest: dict[str, Any], output: Path):
        self.run = None
        if not config.wandb_project:
            return
        import wandb

        identity = output / "wandb_identity.json"
        run_id = (
            json.loads(identity.read_text())["id"] if identity.exists() else uuid.uuid4().hex[:8]
        )
        identity.write_text(json.dumps({"id": run_id}))
        self.run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_name,
            id=run_id,
            resume="allow",
            config=manifest,
            dir=str(output),
            save_code=False,
        )
        self.run.define_metric("optimizer_step")
        self.run.define_metric("train/*", step_metric="optimizer_step")
        self.run.define_metric("val/*", step_metric="optimizer_step")
        (output / "wandb_run.json").write_text(json.dumps({"id": run_id, "url": self.run.url}))

    def log(self, prefix: str, values: dict[str, Any], step: int, epoch: float):
        if self.run is None:
            return
        payload = {"optimizer_step": step, "epoch": epoch}

        def flatten(data, path):
            for name, value in data.items():
                key = f"{path}/{name}"
                if isinstance(value, dict):
                    flatten(value, key)
                elif isinstance(value, (int, float)):
                    payload[key] = value

        flatten(values, prefix)
        self.run.log(payload)

    def finish(self):
        if self.run is not None:
            self.run.finish()
