"""Measure one discarded optimizer update on the longest real training sequences."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from odyn_sft.pytorch_sft.config import load_config
from odyn_sft.pytorch_sft.data import TokenDataset, collate_training
from odyn_sft.pytorch_sft.training.model_setup import prepare_base_for_lora
from odyn_sft.pytorch_sft.training.objective import completion_loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    config = load_config(args.config)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("Expose one BF16-capable GPU")
    from ..pytorch_sft.data import validate_prepared
    validate_prepared(config)
    torch.manual_seed(config.seed)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(total-config.vram_headroom_gib*1024**3, free-config.available_memory_headroom_gib*1024**3)/total)
    quant = config.bnb_config()  # Same base precision as training: NF4 QLoRA or BF16 LoRA.
    model = AutoModelForCausalLM.from_pretrained(config.model,revision=config.revision,
            quantization_config=quant,dtype=torch.bfloat16,device_map={'':0},
            attn_implementation='sdpa',local_files_only=True)
    model.config.use_cache = False
    model = prepare_base_for_lora(model, config, quantized=quant is not None)
    model = get_peft_model(model,LoraConfig(r=config.lora_rank,lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,target_modules=config.target_modules,bias='none',task_type='CAUSAL_LM'))
    tokenizer = AutoTokenizer.from_pretrained(Path(config.prepared)/'tokenizer',local_files_only=True,fix_mistral_regex=config.fix_mistral_regex)
    dataset = TokenDataset(Path(config.prepared),'train',limit=args.batch)
    batch = collate_training([dataset[i] for i in range(len(dataset))], tokenizer.pad_token_id)
    if config.optimizer == 'adamw_8bit':
        from bitsandbytes.optim import AdamW8bit
        optimizer_class, options = AdamW8bit, {'min_8bit_size':4096}
    else:
        optimizer_class, options = torch.optim.AdamW, {'foreach':False}
    optimizer = optimizer_class([p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate, betas=(config.adam_beta1,config.adam_beta2),
        eps=config.adam_epsilon, weight_decay=config.weight_decay, **options)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    backend = (torch.nn.attention.SDPBackend.FLASH_ATTENTION if config.sdpa_backend == 'flash'
               else torch.nn.attention.SDPBackend.MATH)
    with torch.nn.attention.sdpa_kernel(backend):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,count,_ = completion_loss(model,batch['input_ids'].cuda(),batch['labels'].cuda(),
                    config.loss_chunk_tokens,attention_mask=batch['attention_mask'].cuda())
        (loss/count).backward()
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm))
    optimizer.step()
    torch.cuda.synchronize()
    report={'batch':args.batch,'shape':list(batch['input_ids'].shape),'source_indices':dataset.indices,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
            'grad_norm':norm,'loss':float(loss.detach())/count,'weights_discarded':True,
            'adapter_reused_for_training':False}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except torch.cuda.OutOfMemoryError:
        print('CUDA OOM: retry a smaller microbatch in a fresh process',file=sys.stderr,flush=True)
        raise SystemExit(42)
