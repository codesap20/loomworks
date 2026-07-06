# tourney-text-trainer

Tournament training repo for Gradients (Bittensor SN56) text tasks.

## Layout

- `src/main.py` — dispatcher: parses the validator CLI, sizes the job, retries
  with degraded settings, guarantees a loadable submission.
- `src/tok_axolotl.py` — tokenization in a pinned axolotl venv so training
  formatting matches evaluation exactly (instruct + chat template tasks).
- `src/train_sft.py` — instruct/chat trainer: WSD schedule with wall-clock
  planned decay, CPU-shadow EMA vs best-raw dev selection, FA2 flattening
  packing, optional KL(ft||base) objective for KL-scored tasks.
- `src/train_dpo.py`, `src/train_grpo.py` — preference / RL tasks (TRL).
- `src/train_chat_modern.py` — continuous-SFT for custom hybrid
  linear-attention architectures under transformers v5 (trust_remote_code,
  large-vocab-aware loss).
- `ops/docker/standalone-text-trainer.dockerfile` — three stacks in one image:
  system (train), axolotl venv (tokenize), modern venv (v5 archs).

## Runtime contract

See the Gradients miner docs: input model/dataset are read from the read-only
`/cache` volume; the final model is written to
`/app/checkpoints/{task_id}/{expected_repo_name}`.
