"""Exact causal completion loss without allocating logits for the whole prompt."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def completion_loss(
    model: nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    chunk_tokens: int = 128,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, int, int]:
    """Return summed NLL, supervised token count and teacher-forced correct tokens.

    Prompt states still participate in attention and gradients. Only the output
    projection is skipped for unsupervised positions. Checkpointing each head
    chunk avoids retaining all vocabulary-sized logits for backward.
    """
    causal_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    if causal_model.config.model_type not in ("qwen2", "qwen3", "mistral"):
        raise ValueError("The completion-only head path is tested for Qwen2/Qwen3/Mistral")
    # Flash SDPA cannot accept Transformers' dense padding mask. Right padding
    # is safe without that mask in causal training: real tokens cannot attend
    # to padding located in their future, and padded targets are ignored.
    # Reject left/internal padding rather than silently changing its semantics.
    if attention_mask is not None:
        if torch.any(attention_mask[:, 1:] > attention_mask[:, :-1]):
            raise ValueError("Completion batching requires right padding")
        if torch.any(labels[attention_mask == 0] != -100):
            raise ValueError("Padding labels must be excluded from loss")
    # An explicit all-valid mask also prevents accidental packed-mask inference.
    hidden = causal_model.model(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False
    ).last_hidden_state
    shifted_labels = labels[:, 1:]
    valid = shifted_labels != -100
    targets = shifted_labels[valid]
    states = hidden[:, :-1, :][valid]
    if not targets.numel():
        raise ValueError("No supervised completion tokens")
    head = causal_model.get_output_embeddings()
    correct = 0
    losses = []

    def project_loss(x: Tensor, y: Tensor) -> Tensor:
        logits = head(x)
        return F.cross_entropy(logits.float(), y, reduction="sum")

    for start in range(0, len(targets), chunk_tokens):
        x = states[start : start + chunk_tokens]
        y = targets[start : start + chunk_tokens]
        if torch.is_grad_enabled():
            losses.append(checkpoint(project_loss, x, y, use_reentrant=False))
        else:
            logits = head(x)
            losses.append(F.cross_entropy(logits.float(), y, reduction="sum"))
            correct += int((logits.argmax(-1) == y).sum())
    return torch.stack(losses).sum(), len(targets), correct
