"""Generate responses from a pinned base model, with or without the SFT1/SFT2 LoRA adapters.

- `odyn-infer` (runner.py): one model on one JSONL input, any task, no scoring. Used for
  single-stage base-vs-adapter comparisons (compare.py, odyn-judge).
- `odyn-e2e` (e2e.py): the real chain on held-out claims. SFT1 writes a program, the program
  is executed and scored, then SFT2 interprets the program's own evidence and is judged.
"""
