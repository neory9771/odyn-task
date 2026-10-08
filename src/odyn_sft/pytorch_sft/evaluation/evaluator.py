"""Shared completion loss, generation and caching; quality scoring belongs to task adapters."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import torch

from ...common.storage import read_json, write_json
from ..config import Config
from ..data import HeldOutDataset, TokenDataset
from ..training.objective import completion_loss

# BF16 autocast does not change the stored FP32 training adapter/norm weights.
SELECTION_LOSS_PRECISION = {
    "base_linear_weights": "NF4",
    "norm_weights": "fp32",
    "lora_weights": "fp32",
    "autocast": "bf16",
    "frozen_embeddings_and_head": "bf16",
}
FINAL_LOSS_PRECISION = {**SELECTION_LOSS_PRECISION, "norm_weights": "bf16", "lora_weights": "bf16"}


def selection_loss_precision(config: Config) -> dict[str, str]:
    """QLoRA upcasts norms to FP32 via k-bit preparation; BF16-base LoRA keeps them BF16."""
    if config.quantization == "nf4":
        return dict(SELECTION_LOSS_PRECISION)
    return {**SELECTION_LOSS_PRECISION, "base_linear_weights": "bf16", "norm_weights": "bf16"}


def loss_precision_comparison(
    selection: dict[str, Any], final: dict[str, Any], final_indices: list[int]
) -> dict[str, Any]:
    """Keep original selection loss beside the deployment-profile loss.

    Their difference is descriptive: there is no tolerance or implied bound
    on the effect of casting weights, and no claim that it must be negligible.
    """
    same_examples = selection.get("validation_source_indices") == final_indices
    return {
        "selection_loss": selection["loss"],
        "final_validation_loss": final["loss"],
        "final_minus_selection_loss": final["loss"] - selection["loss"],
        "selection_precision": selection.get("loss_precision", SELECTION_LOSS_PRECISION),
        "final_precision": {
            **FINAL_LOSS_PRECISION,
            "base_linear_weights": selection.get("loss_precision", SELECTION_LOSS_PRECISION)["base_linear_weights"],
        },
        "same_validation_examples": same_examples,
        "note": "Selection used the recorded norm/LoRA storage precision under BF16 autocast; final evaluation uses BF16 stored norm/LoRA weights. Losses need not match."
        + (
            ""
            if same_examples
            else " Validation example sets differ or are not verified; the loss difference also reflects scope."
        ),
    }


def generation_kwargs(tokenizer: Any, remaining: int) -> dict[str, Any]:
    """Keep greedy decoding explicit when model defaults enable sampling.

    Transformers can replace fields equal to global defaults, including an
    explicit do_sample=False, with settings from a model's generation config.
    Disable that merge so validation remains reproducible across base models.
    """
    from transformers import GenerationConfig

    return {
        "use_model_defaults": False,
        "generation_config": GenerationConfig(
            use_cache=True,
            max_new_tokens=remaining,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ),
    }


def evaluate(
    model: Any,
    tokenizer: Any,
    dataset: TokenDataset | HeldOutDataset,
    config: Config,
    device: torch.device,
    output: Path,
    tag: str,
    stop_requested: Callable[[], bool],
    run_fingerprint: str,
    *,
    split: str = "validation",
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("Evaluate only validation or held-out test splits")
    if split == "test" and not config.validation_generation:
        raise ValueError("Final test evaluation must check generated outputs")
    if not config.validation_generation:
        return evaluate_loss_only(
            model, dataset, config, device, output, tag, stop_requested, run_fingerprint
        )
    from ...tasks import get_task

    task = get_task(config.task)
    from transformers import StoppingCriteria, StoppingCriteriaList

    class CoolingStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.full(
                (input_ids.shape[0],),
                stop_requested(),
                dtype=torch.bool,
                device=input_ids.device,
            )

    scorer = task.scorer(Path(config.source_suite), split)
    directory = output / split / tag
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {
        "run_fingerprint": run_fingerprint,
        "task": config.task,
        "tag": tag,
        "source_indices": dataset.indices,
        "split": split,
    }
    if (directory / "provenance.json").exists() and read_json(
        directory / "provenance.json"
    ) != provenance:
        raise ValueError("Validation cache model/data/checkpoint mismatch")
    write_json(directory / "provenance.json", provenance)
    sums = {
        "nll": 0.0,
        "tokens": 0,
        "correct": 0,
        "truncated": 0,
        "generated_tokens": 0,
        "case_seconds": 0.0,
    }

    def accumulate(record: dict[str, Any], case: dict[str, Any]) -> None:
        if record["record_id"] != case["record_id"]:
            raise ValueError("Validation cache lineage mismatch")
        scorer.accumulate(record, case)
        sums["nll"] += record["nll_sum"]
        sums["tokens"] += record["target_tokens"]
        sums["correct"] += record["teacher_forced_correct"]
        sums["truncated"] += int(record["truncated"])
        sums["generated_tokens"] += record["generated_tokens"]
        sums["case_seconds"] += record["seconds"]

    started = time.monotonic()
    model.eval()
    # Eval can be interrupted safely between examples; the caller has already checkpointed.
    with torch.inference_mode():
        for i in range(len(dataset)):
            if stop_requested():
                return {"interrupted": True, "completed_examples": i}
            row = dataset[i]
            case = scorer.case(row["source_index"])
            case_path = directory / f"case-{i:04d}.json"
            if case_path.exists():
                accumulate(read_json(case_path), case)
                continue
            case_started = time.monotonic()
            partial_path = directory / f"partial-{i:04d}.json"
            partial = read_json(partial_path) if partial_path.exists() else None
            if partial is not None and partial["record_id"] != case["record_id"]:
                raise ValueError("Partial generation lineage mismatch")
            ids, labels = row["input_ids"].to(device), row["labels"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                if partial is None:
                    nll, count, correct = completion_loss(
                        model, ids, labels, config.loss_chunk_tokens
                    )
                    partial = {
                        "record_id": case["record_id"],
                        "nll_sum": float(nll),
                        "target_tokens": count,
                        "teacher_forced_correct": correct,
                        "new_tokens": [],
                        "seconds": 0.0,
                    }
                    write_json(partial_path, partial)
                else:
                    nll, count, correct = (
                        partial["nll_sum"],
                        partial["target_tokens"],
                        partial["teacher_forced_correct"],
                    )
                prefix = ids[:, : row["prompt_length"]]
                if prefix.shape[1] + config.generation_max_new_tokens > config.max_length:
                    raise ValueError("Full prompt plus generation budget exceeds the model context")
                new = partial["new_tokens"]
                ended = bool(new) and new[-1] == tokenizer.eos_token_id
                remaining = config.generation_max_new_tokens - len(new)
                if stop_requested() and remaining > 0 and not ended:
                    partial["seconds"] += time.monotonic() - case_started
                    write_json(partial_path, partial)
                    return {"interrupted": True, "completed_examples": i}
                if remaining > 0 and not ended:
                    continuation = torch.tensor([new], dtype=ids.dtype, device=device)
                    generation_input = torch.cat([prefix, continuation], dim=1)
                    generated = model.generate(
                        input_ids=generation_input,
                        attention_mask=torch.ones_like(generation_input),
                        **generation_kwargs(tokenizer, remaining),
                        logits_to_keep=1,
                        stopping_criteria=StoppingCriteriaList([CoolingStop()]),
                    )
                    new = new + generated[0, generation_input.shape[1] :].tolist()
                if (
                    stop_requested()
                    and len(new) < config.generation_max_new_tokens
                    and (not new or new[-1] != tokenizer.eos_token_id)
                ):
                    partial.update(
                        new_tokens=new,
                        seconds=partial["seconds"] + time.monotonic() - case_started,
                    )
                    write_json(partial_path, partial)
                    write_json(
                        output / "live_validation.json",
                        {
                            "tag": tag,
                            "completed": i,
                            "total": len(dataset),
                            "partial_generation_tokens": len(new),
                        },
                    )
                    return {"interrupted": True, "completed_examples": i}
            completion = tokenizer.decode(new, skip_special_tokens=True)
            truncated = bool(new) and (
                len(new) == config.generation_max_new_tokens and new[-1] != tokenizer.eos_token_id
            )
            scored = scorer.score(completion, case)
            record = {
                "record_id": case["record_id"],
                **scored,
                "raw_completion": completion,
                "truncated": truncated,
                "target_tokens": count,
                "nll_sum": float(nll),
                "teacher_forced_correct": correct,
                "generated_tokens": len(new),
                "seconds": partial["seconds"] + time.monotonic() - case_started,
            }
            record["outcome"] = scorer.outcome(record, case)
            # Per-case durable output allows inspecting failure locations while validation runs.
            write_json(case_path, record)
            partial_path.unlink(missing_ok=True)
            accumulate(record, case)
            write_json(
                output / "live_validation.json",
                {"tag": tag, "completed": i + 1, "total": len(dataset)},
            )
    loss = sums["nll"] / sums["tokens"]
    if not math.isfinite(loss):
        raise FloatingPointError("Non-finite validation loss")
    metrics = {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 700 else float("inf"),
        "teacher_forced_token_accuracy": sums["correct"] / sums["tokens"],
        **scorer.metrics(len(dataset)),
        "examples": len(dataset),
        "completion_tokens": sums["tokens"],
        "truncation_rate": sums["truncated"] / len(dataset),
        "generated_tokens": sums["generated_tokens"],
        "seconds": sums["case_seconds"],
        "session_seconds": time.monotonic() - started,
    }
    write_json(directory / "metrics.json", metrics)
    return metrics


def evaluate_loss_only(
    model: Any,
    dataset: TokenDataset,
    config: Config,
    device: torch.device,
    output: Path,
    tag: str,
    stop_requested: Callable[[], bool],
    run_fingerprint: str,
) -> dict[str, Any]:
    """Measure completion NLL and token accuracy without generation or execution.

    Each completed example is durable so a thermal interruption or restart
    does not repeat its forward pass. No quality scores for generated outputs are
    emitted: those metrics were not measured in this mode.
    """
    directory = output / "validation" / tag
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {
        "run_fingerprint": run_fingerprint,
        "tag": tag,
        "source_indices": dataset.indices,
        "validation_mode": "loss_only",
        "task": config.task,
    }
    if (directory / "provenance.json").exists() and read_json(
        directory / "provenance.json"
    ) != provenance:
        raise ValueError("Validation cache model/data/checkpoint mismatch")
    write_json(directory / "provenance.json", provenance)
    total_nll, total_tokens, total_correct, total_seconds = 0.0, 0, 0, 0.0
    started = time.monotonic()
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            if stop_requested():
                return {"interrupted": True, "completed_examples": index}
            row = dataset[index]
            path = directory / f"loss-case-{index:04d}.json"
            if path.exists():
                record = read_json(path)
                if record["source_index"] != row["source_index"]:
                    raise ValueError("Validation loss cache lineage mismatch")
            else:
                tick = time.monotonic()
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    nll, count, correct = completion_loss(
                        model,
                        row["input_ids"].to(device),
                        row["labels"].to(device),
                        config.loss_chunk_tokens,
                    )
                if count != row["target_tokens"] or not torch.isfinite(nll):
                    raise FloatingPointError("Invalid validation completion loss or mask")
                record = {
                    "source_index": row["source_index"],
                    "nll_sum": float(nll),
                    "target_tokens": count,
                    "teacher_forced_correct": correct,
                    "seconds": time.monotonic() - tick,
                }
                write_json(path, record)
            total_nll += record["nll_sum"]
            total_tokens += record["target_tokens"]
            total_correct += record["teacher_forced_correct"]
            total_seconds += record["seconds"]
            write_json(
                output / "live_validation.json",
                {"tag": tag, "completed": index + 1, "total": len(dataset), "mode": "loss_only"},
            )
    loss = total_nll / total_tokens
    if not math.isfinite(loss):
        raise FloatingPointError("Non-finite validation loss")
    metrics = {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 700 else float("inf"),
        "teacher_forced_token_accuracy": total_correct / total_tokens,
        "examples": len(dataset),
        "completion_tokens": total_tokens,
        "seconds": total_seconds,
        "session_seconds": time.monotonic() - started,
        "scope": "teacher-forced completion loss and accuracy; no generated-output evaluation",
    }
    write_json(directory / "metrics.json", metrics)
    return metrics
