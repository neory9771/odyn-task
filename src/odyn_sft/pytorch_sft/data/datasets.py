"""Training token storage and gated held-out reference loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from .preparation import encode_row, iter_jsonl


class TokenDataset:
    """Memory-mapped token storage avoids Python integers for repeated 25k prompts."""

    def __init__(self, root: Path, split: str, limit: int | None = None):
        self.tokens = np.memmap(root / f"{split}.tokens.bin", dtype=np.int32, mode="r")
        self.offsets = np.load(root / f"{split}.offsets.npy", mmap_mode="r")
        self.prompts = np.load(root / f"{split}.prompts.npy", mmap_mode="r")
        self.indices = list(range(len(self.prompts)))
        if limit is not None:
            # Development plumbing tests include the longest actual examples.
            self.indices.sort(
                key=lambda i: int(self.offsets[i + 1] - self.offsets[i]), reverse=True
            )
            self.indices = self.indices[:limit]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        i = self.indices[index]
        ids = torch.from_numpy(
            np.array(self.tokens[self.offsets[i] : self.offsets[i + 1]], dtype=np.int64)
        )
        labels = ids.clone()
        prompt = int(self.prompts[i])
        labels[:prompt] = -100
        return {
            "input_ids": ids.unsqueeze(0),
            "labels": labels.unsqueeze(0),
            "prompt_length": prompt,
            "source_index": i,
            "target_tokens": len(ids) - prompt,
        }


class HeldOutDataset:
    """Tokenize held-out references only after the final-training gate passes.

    Generation receives only the prompt slice; reference completion tokens are
    used solely for loss, never fed to model.generate.
    """

    def __init__(self, config: Config, tokenizer: Any):
        suite = Path(config.source_suite)
        from ...tasks import get_task

        task = get_task(config.task)
        inputs = list(iter_jsonl(task.record_path(suite, "test", inputs=True)))
        gold = list(iter_jsonl(task.record_path(suite, "test")))
        if not inputs or len(inputs) != len(gold):
            raise ValueError("Missing or misaligned held-out inputs and references")
        self.rows: list[dict[str, Any]] = []
        self.indices = list(range(len(inputs)))
        for index, (prompt, reference) in enumerate(zip(inputs, gold)):
            if prompt != {key: reference[key] for key in ("instruction", "input")}:
                raise ValueError("Held-out reference and prompt mismatch")
            encoded = encode_row(reference, tokenizer, config)
            import torch

            ids = torch.tensor(encoded["input_ids"], dtype=torch.long).unsqueeze(0)
            length = encoded["completion_mask"].count(0)
            labels = ids.clone()
            labels[:, :length] = -100
            self.rows.append(
                {
                    "input_ids": ids,
                    "labels": labels,
                    "prompt_length": length,
                    "source_index": index,
                    "target_tokens": ids.shape[1] - length,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]
