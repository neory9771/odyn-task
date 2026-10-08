"""Durable, resumable one-row-at-a-time inference, with no dataset generator dependency."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..common.storage import digest, file_hash, read_json, read_jsonl, write_json
from ..pytorch_sft.config import Config, load_config

cli_file = Path(__file__).with_name("cli.py")


def messages(row: dict[str, Any], config: Config) -> list[dict[str, str]]:
    """Use the training chat contract; row.output is never supplied to the model."""
    for field in ("instruction", "input"):
        if not isinstance(row.get(field), str):
            raise ValueError(f"Input rows require a string {field}")
    result = [{"role": "user", "content": row["instruction"] + "\n\n" + row["input"]}]
    if config.system_prompt is not None:
        result.insert(0, {"role": "system", "content": config.system_prompt})
    return result


def check_adapter(adapter: Path, config: Config) -> None:
    if read_json(adapter / "adapter_config.json")["base_model_name_or_path"] != config.model:
        raise ValueError("Adapter base model does not match inference model")


def adapter_hashes(adapter: Path | None) -> dict[str, str]:
    if adapter is None:
        return {}
    return {p.name: file_hash(p) for p in sorted(adapter.glob("*")) if p.is_file()}


def cast_lora_bf16(model: Any) -> None:
    """Saved LoRA weights run in BF16, as in the deployment evaluator."""
    import torch

    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.to(torch.bfloat16)


def read_results(path: Path) -> list[dict[str, Any]]:
    """Read flushed rows, dropping a final line left incomplete by a crash mid-write."""
    if not path.exists():
        return []
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    if len(complete) != len(data):
        with path.open("r+b") as handle:
            handle.truncate(len(complete))
    return [json.loads(line) for line in complete.decode("utf-8").splitlines() if line.strip()]


class TemperatureStop(RuntimeError):
    """A generation was interrupted safely; its partial output is not a finished row."""


class Engine:
    """Cached pinned HF model, greedy decoding and the same base precision for both runs."""

    def __init__(
        self,
        config: Config,
        adapter: Path | None,
        max_temperature: float,
        disable_temperature_checks: bool,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one available CUDA GPU")
        self.config = config
        self.device = torch.device("cuda:0")
        self.max_temperature = max_temperature
        self.disable_temperature_checks = disable_temperature_checks
        self.last_check = 0.0
        self.thermal_reason = None
        if self.check_temperature():
            raise TemperatureStop(self.thermal_reason)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            revision=config.revision,
            local_files_only=True,
            fix_mistral_regex=config.fix_mistral_regex,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            revision=config.revision,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            quantization_config=config.bnb_config(),
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        if adapter is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, adapter, is_trainable=False
            )
            cast_lora_bf16(self.model)
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.config.use_cache = True

    def check_temperature(self) -> bool:
        if self.disable_temperature_checks:
            return False
        if time.monotonic() - self.last_check < 5:
            return self.thermal_reason is not None
        from ..run_management.supervisor import gpu_selector, read_gpu

        self.last_check = time.monotonic()
        try:
            sample = read_gpu(gpu_selector())
            if sample["temperature_c"] >= self.max_temperature:
                self.thermal_reason = f"GPU temperature {sample['temperature_c']} >= {self.max_temperature} C"
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.thermal_reason = f"GPU temperature sensor unavailable: {exc}"
        return self.thermal_reason is not None

    def __call__(
        self, row: dict[str, Any], max_new_tokens: int, config: Config | None = None
    ) -> dict[str, Any]:
        """Generate one response; config selects the system prompt and length limit."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        from ..pytorch_sft.evaluation.evaluator import generation_kwargs

        config = config or self.config
        if self.check_temperature():
            raise TemperatureStop(self.thermal_reason)
        options = (
            {}
            if config.enable_thinking is None
            else {"enable_thinking": config.enable_thinking}
        )
        ids = self.tokenizer.apply_chat_template(
            messages(row, config),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            **options,
        ).to(self.device)
        length = ids.shape[1]
        budget = min(max_new_tokens, config.max_length - length)
        if budget < 1:
            raise ValueError(
                f"Prompt length {length} leaves no room under {config.max_length}; no truncation"
            )
        engine = self

        class CoolingStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full(
                    (input_ids.shape[0],),
                    engine.check_temperature(),
                    dtype=torch.bool,
                    device=input_ids.device,
                )

        torch.cuda.synchronize()
        started = time.monotonic()
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16),
            torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.FLASH_ATTENTION
            ),
        ):
            result = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                stopping_criteria=StoppingCriteriaList([CoolingStop()]),
                **generation_kwargs(self.tokenizer, budget),
            )
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        tokens = result[0, length:].tolist()
        if self.thermal_reason:
            raise TemperatureStop(self.thermal_reason)
        completion = self.tokenizer.decode(tokens, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        truncated = len(tokens) >= budget and (not tokens or tokens[-1] not in eos)
        return {
            "completion": completion,
            "prompt_tokens": length,
            "generated_tokens": len(tokens),
            "generation_budget": budget,
            "seconds": seconds,
            "tokens_per_second": len(tokens) / seconds,
            "finish_reason": "length" if truncated else "eos",
            "status": "truncated"
            if truncated
            else "successful"
            if completion.strip()
            else "empty_output",
        }


def run(
    config_path: Path,
    input_path: Path,
    output: Path,
    *,
    adapter: Path | None = None,
    limit: int | None = None,
    max_new_tokens: int | None = None,
    resume: bool = False,
    max_temperature: float = 90,
    disable_temperature_checks: bool = False,
    engine_factory: Callable[..., Any] = Engine,
) -> dict[str, Any]:
    """Flush every result to disk; resume rejects changed inputs, model or decoding settings."""
    config = load_config(config_path)
    input_path, output = input_path.resolve(), output.resolve()
    if output == input_path:
        raise ValueError("Output must differ from input")
    if not input_path.is_file():
        raise ValueError("Input JSONL must exist and contain rows")
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError("Input JSONL must exist and contain rows")
    rows = rows if limit is None else rows[:limit]
    for row in rows:
        messages(row, config)
    adapter = adapter.resolve() if adapter is not None else None
    if adapter is not None:
        check_adapter(adapter, config)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    max_new_tokens = max_new_tokens or config.generation_max_new_tokens
    manifest = {
        "version": "jsonl-inference-1",
        "input": str(input_path),
        "input_sha256": file_hash(input_path),
        "config": config.as_dict(),
        "adapter": str(adapter) if adapter else None,
        "adapter_hashes": adapter_hashes(adapter),
        # Only the code that shapes these rows; editing e2e.py must not block a resume.
        "implementation_hashes": {
            p.name: file_hash(p) for p in (Path(__file__), cli_file)
        },
        "limit": limit,
        "examples": len(rows),
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "batch_size": 1,
        "max_temperature": None if disable_temperature_checks else max_temperature,
        "precision": {
            "base": config.base_weight_precision(),
            "adapter": "bf16",
            "autocast": "bf16",
        },
        "gold_targets_supplied_to_model": False,
        "semantic_evaluation": "not_run",
    }
    records = []
    if output.exists() or manifest_path.exists():
        if not resume:
            raise ValueError("Output exists; choose a new path or pass --resume")
        if not manifest_path.exists():
            raise ValueError("Cannot resume: manifest is missing")
        if digest(read_json(manifest_path)) != digest(manifest):
            raise ValueError(
                "Cannot resume: inputs/model/settings/implementation changed"
            )
        records = read_results(output)
        if len(records) > len(rows) or any(
            r["source_index"] != i for i, r in enumerate(records)
        ):
            raise ValueError("Invalid result alignment")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    started = time.monotonic()
    state, error = "completed", None
    try:
        if len(records) < len(rows):
            engine = engine_factory(
                config, adapter, max_temperature, disable_temperature_checks
            )
            with output.open("a", encoding="utf-8") as handle:
                for index in range(len(records), len(rows)):
                    row = rows[index]
                    try:
                        result = engine(row, max_new_tokens)
                    except TemperatureStop:
                        raise
                    except (ValueError, RuntimeError) as exc:
                        result = {
                            "completion": "",
                            "status": "runtime_error",
                            "error": str(exc),
                        }
                    record = {
                        "source_index": index,
                        "record_id": (row.get("metadata") or {}).get("record_id"),
                        "model_variant": config.task if adapter else "base",
                        **result,
                    }
                    handle.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                    records.append(record)
                    print(
                        json.dumps(
                            {
                                "completed": len(records),
                                "total": len(rows),
                                "status": record["status"],
                                "seconds": record.get("seconds"),
                            }
                        ),
                        flush=True,
                    )
    except (TemperatureStop, KeyboardInterrupt) as exc:
        state, error = "interrupted", str(exc) or "KeyboardInterrupt"
    except (RuntimeError, OSError, ValueError) as exc:
        state, error = "failed", str(exc)
    counts = Counter(r["status"] for r in records)
    seconds = sum(r.get("seconds", 0) for r in records)
    tokens = sum(r.get("generated_tokens", 0) for r in records)
    summary = {
        "state": state,
        "error": error,
        "output": str(output),
        "completed": len(records),
        "expected": len(rows),
        "outcomes": dict(counts),
        "session_seconds": time.monotonic() - started,
        "generation_seconds": seconds,
        "generated_tokens": tokens,
        "generation_tokens_per_second": tokens / seconds if seconds else None,
        "semantic_evaluation": "not_run",
        "success_definition": "Nonempty response ending within token budget; not semantic correctness",
    }
    write_json(summary_path, summary)
    return summary
