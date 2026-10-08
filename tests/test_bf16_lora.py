"""BF16-base LoRA ("quantization": "none") alongside the default NF4 QLoRA profile."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(tmp_path, **overrides):
    from odyn_sft.pytorch_sft.config import load_config
    values = json.loads((ROOT / "configs/mistral-nf4.json").read_text())
    values.update(workspace=str(ROOT), **overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    return load_config(path)


def test_quantization_profiles_validate_and_map_to_loading(tmp_path):
    pytest.importorskip("transformers")
    from odyn_sft.pytorch_sft.evaluation.evaluator import selection_loss_precision
    nf4, bf16 = load(tmp_path), load(tmp_path, quantization="none")
    assert nf4.bnb_config().load_in_4bit and nf4.base_weight_precision() == "NF4"
    assert bf16.bnb_config() is None and bf16.base_weight_precision() == "bf16"
    assert selection_loss_precision(nf4)["base_linear_weights"] == "NF4"
    assert selection_loss_precision(bf16) | {} == {**selection_loss_precision(nf4),
                                                   "base_linear_weights": "bf16", "norm_weights": "bf16"}
    with pytest.raises(ValueError, match="NF4 or BF16 base"):
        load(tmp_path, quantization="int8")


def test_bf16_base_stays_frozen_bf16_with_fp32_lora_and_gradients(tmp_path):
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from transformers import MistralConfig, MistralForCausalLM

    from odyn_sft.pytorch_sft.training.model_setup import prepare_base_for_lora
    from odyn_sft.pytorch_sft.training.objective import completion_loss
    torch.manual_seed(0)
    model = MistralForCausalLM(MistralConfig(vocab_size=64, hidden_size=48, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, sliding_window=None)).to(torch.bfloat16)
    config = load(tmp_path, quantization="none")
    model = prepare_base_for_lora(model, config, quantized=False)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules="all-linear",
                                             task_type="CAUSAL_LM"))
    model.train()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert all(p.dtype == torch.float32 for p in trainable.values())
    assert all(p.dtype == torch.bfloat16 for n, p in model.named_parameters() if not p.requires_grad)
    assert model.base_model.model.is_gradient_checkpointing
    ids = torch.randint(3, 64, (1, 12))
    labels = ids.clone()
    labels[:, :6] = -100
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        nll, count, _ = completion_loss(model, ids, labels, 4)
    (nll / count).backward()
    assert torch.isfinite(nll)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
    # lora_B starts at zero, so only lora_B receives a nonzero gradient on the first step.
    assert any(p.grad.abs().sum() > 0 for n, p in trainable.items() if "lora_B" in n)
