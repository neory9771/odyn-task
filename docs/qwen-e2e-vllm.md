# Queued Qwen E2E on H200

This comparison uses the newest remote **Qwen3-4B SFT1 and SFT2 step-72**
checkpoints from the completed 181-row paired pilot. They are distinct from
the older 5k SFT1 run stopped at step 132. Latest-checkpoint selection is
intentional; SFT1's validation-best pilot checkpoint was step 66.

Configs: `configs/qwen-e2e-vllm-sft1.json` and `qwen-e2e-vllm-sft2.json`.
Scripts: `scripts/h200/e2e_qwen_vllm.sh` and `queue_qwen_e2e.sh`.

The remote layout under `/dev/shm/odyn-qwen-e2e` contains a frozen `code/`,
`configs/qwen-sft1.json`, `configs/qwen-sft2.json`, `run_vllm.sh`, `queue.sh`,
`qwen_run_plan.json`, inventory and checksum files. Preparation pins hashes
of code, configs, adapters, test inputs and the judge gate.

The queue waits for the active Mistral **SFT/SFT and base/base** comparison to
finish and release the GPU. It then runs the same two chains with Qwen,
on the same **1,000 original client claims**. It does not train models,
change datasets or interrupt the current job. A failed Mistral completion
or busy GPU leaves Qwen unstarted and records the reason.

Both stages use greedy decoding, thinking disabled, a 32k context and 1,536
new-token limit. vLLM serves BF16 weights plus both rank-16 adapters with
prefix caching, concurrency 64 and 128 maximum sequences. Judge concurrency
is 8, using the same validated judge and shared content-addressed cache as
Mistral. Temperature stops are explicitly disabled on this remote.

Start the prepared queue once; do not launch duplicate watchers:

```bash
ssh odyn-h200
cd /dev/shm/odyn-qwen-e2e
LIMIT=1000 CONC=64 setsid nohup bash queue.sh > queue.log 2>&1 < /dev/null &
```

Watch `queue_state.json`, `queue.log`, then `run-vllm-limit1000.log`.
Results live in `runs/{sft-sft,base-base}-vllm-limit1000/`: `summary.json`,
`e2e.jsonl` and `judge/results.jsonl`. Each run has its own resume manifest;
changing inputs requires a new output directory.

The launcher copies results, configs, inventory and checksums to
`/workspace/odyn-qwen-e2e`. Pull them locally as well:

```bash
rsync -a odyn-h200:/dev/shm/odyn-qwen-e2e/runs/ ../../runs/qwen-e2e-full/
```

No generated test result is available until the queued run completes.
See [judging](judging.md) for pass-rate semantics and judge limitations.
