"""Task-independent chat rendering and completion-only token masks."""

from typing import Any


def prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "user", "content": row["instruction"] + "\n\n" + row["input"]}]


def tokenize_row(
    row: dict[str, Any],
    tokenizer: Any,
    max_length: int,
    *,
    enable_thinking: bool | None = None,
    system_prompt: str | None = None,
) -> dict[str, list[int]]:
    prompt = prompt_messages(row)
    if system_prompt is not None:
        prompt = [{"role": "system", "content": system_prompt}] + prompt
    full = prompt + [{"role": "assistant", "content": row["output"]}]
    options = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    prefix = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, **options
    )
    tokens = tokenizer.apply_chat_template(
        full, tokenize=True, add_generation_prompt=False, **options
    )
    if tokens[: len(prefix)] != prefix or len(tokens) <= len(prefix):
        raise ValueError("Chat template prompt is not a prefix of the completed conversation")
    if len(tokens) > max_length:
        raise ValueError(f"Overlength example: {len(tokens)}>{max_length}; no truncation permitted")
    return {
        "input_ids": tokens,
        "attention_mask": [1] * len(tokens),
        "completion_mask": [0] * len(prefix) + [1] * (len(tokens) - len(prefix)),
    }
