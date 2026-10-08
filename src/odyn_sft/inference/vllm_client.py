"""Generate through a vLLM OpenAI-compatible server instead of an in-process HF model.

Prompts are rendered and tokenized here with the same tokenizer and chat template as
`runner.Engine`, then sent as token IDs, so both engines see identical prompt tokens.
Decoding is greedy. The server must serve the base model and, for each step with an
adapter, a LoRA module named after the step (`sft1`, `sft2`) loaded from the same path:

  vllm serve MODEL --revision REV --dtype bfloat16 --max-model-len 32768 \\
      --enable-prefix-caching --enable-lora --max-lora-rank 16 --max-loras 2 \\
      --lora-modules sft1=ADAPTER1 sft2=ADAPTER2
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..pytorch_sft.config import Config
from .runner import Engine, TemperatureStop, messages


class VllmEngine:
    """Same `generate(stage, row)` contract as `e2e.ChainEngine`; safe to call from many threads."""

    concurrent = True

    def __init__(self, configs: dict[str, Config], adapters: dict[str, Path], *, url: str,
                 max_temperature: float = 90, disable_temperature_checks: bool = False,
                 client: Any = None, tokenizer: Any = None):
        base = configs["sft1"]
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(base.model, revision=base.revision, local_files_only=True,
                                                      fix_mistral_regex=base.fix_mistral_regex)
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=url, api_key="EMPTY", timeout=3600, max_retries=0)
        self.configs, self.adapters, self.tokenizer, self.client = configs, adapters, tokenizer, client
        self.max_temperature, self.disable_temperature_checks = max_temperature, disable_temperature_checks
        self.last_check, self.thermal_reason = 0.0, None
        served = {m.id: m for m in client.models.list().data}
        if base.model not in served:
            raise ValueError(f"vLLM server does not serve {base.model}; serving {sorted(served)}")
        for stage, path in adapters.items():
            root = getattr(served.get(stage), "root", None)
            if root is None or Path(root).resolve() != path.resolve():
                raise ValueError(f"vLLM LoRA module {stage!r} must be served from {path}; found {root}")

    check_temperature = Engine.check_temperature

    def generate(self, stage: str, row: dict[str, Any]) -> dict[str, Any]:
        config = self.configs[stage]
        if self.check_temperature():
            raise TemperatureStop(self.thermal_reason)
        options = {} if config.enable_thinking is None else {"enable_thinking": config.enable_thinking}
        ids = self.tokenizer.apply_chat_template(messages(row, config), tokenize=True,
                                                 add_generation_prompt=True, **options)
        ids = list(ids["input_ids"] if isinstance(ids, dict) else ids)  # Newer transformers return a dict.
        budget = min(config.generation_max_new_tokens, config.max_length - len(ids))
        if budget < 1:
            raise ValueError(f"Prompt length {len(ids)} leaves no room under {config.max_length}; no truncation")
        started = time.monotonic()
        response = self.client.completions.create(
            model=stage if stage in self.adapters else config.model, prompt=ids, max_tokens=budget,
            temperature=0, top_p=1, extra_body={"repetition_penalty": 1.0, "skip_special_tokens": True})
        seconds = time.monotonic() - started  # Request latency, including time queued in the batch.
        choice, tokens = response.choices[0], response.usage.completion_tokens
        truncated = choice.finish_reason == "length"
        return {
            "completion": choice.text,
            "prompt_tokens": len(ids),
            "generated_tokens": tokens,
            "generation_budget": budget,
            "seconds": seconds,
            "tokens_per_second": tokens / seconds if seconds else None,
            "finish_reason": "length" if truncated else "eos",
            "status": "truncated" if truncated else "successful" if choice.text.strip() else "empty_output",
        }
