"""Right-padded microbatches within a single optimizer window."""

from __future__ import annotations

from typing import Any


def training_batches(
    records: list[dict[str, Any]],
    pad_token_id: int,
    max_batch: int,
    token_budget: int | None = None,
) -> list[dict[str, Any]]:
    """Batch within one optimizer window, without changing which records it owns.

    A measured padded-token budget permits larger batches for shorter sequences.
    Sort only inside the update window; token-normalized loss still covers every
    original completion exactly once. No examples are packed or truncated.
    """
    if max_batch < 1 or (token_budget is not None and token_budget < 1):
        raise ValueError("Batch and token budgets must be positive")
    if token_budget is None:
        return [
            collate_training(records[i : i + max_batch], pad_token_id)
            for i in range(0, len(records), max_batch)
        ]
    ordered = sorted(records, key=lambda row: row["input_ids"].shape[1], reverse=True)
    batches, pending = [], []
    width = 0
    for row in ordered:
        length = row["input_ids"].shape[1]
        if length > token_budget:
            raise ValueError("A complete sequence exceeds the measured token budget")
        next_width = max(width, length)
        if pending and (
            len(pending) == max_batch or next_width * (len(pending) + 1) > token_budget
        ):
            batches.append(collate_training(pending, pad_token_id))
            pending, width = [], 0
        pending.append(row)
        width = max(width, length)
    if pending:
        batches.append(collate_training(pending, pad_token_id))
    return batches


def collate_training(records: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    """Right-pad independent sequences; future padding never affects real tokens or contributes to loss."""
    import torch

    width = max(row["input_ids"].shape[1] for row in records)
    ids = torch.full((len(records), width), pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(records):
        length = row["input_ids"].shape[1]
        ids[index, :length] = row["input_ids"][0]
        labels[index, :length] = row["labels"][0]
        mask[index, :length] = 1
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": mask,
        "target_tokens": sum(row["target_tokens"] for row in records),
        "examples": len(records),
    }
