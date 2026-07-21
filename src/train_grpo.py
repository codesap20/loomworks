"""GRPO trainer. The validator scores generations with the task's weighted
reward functions and subtracts 0.5 x KL(policy || base) — so the objective is
reward with a meaningful KL leash, not reward at any cost.

vLLM colocate generation when it initializes cleanly, HF generation otherwise.
"""

import argparse
import json
import os
import sys
import time

import torch

import paths
import plan as plan_mod
from reward_compile import compile_rewards


def log(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[grpo {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--base-model-id", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--dataset-type", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--end-ts", type=float, required=True)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--state-file", default=paths.STATE_FILE)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_gpus > 1 and "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node", str(args.num_gpus),
                               os.path.abspath(__file__)] + sys.argv[1:])

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    is_main = int(os.environ.get("RANK", "0")) == 0
    dt = json.loads(args.dataset_type)
    f_prompt = dt.get("field_prompt") or "prompt"
    extra_col = dt.get("extra_column")

    with open(args.data_path) as f:
        rows = json.load(f)
    data = []
    for row in rows:
        p = row.get(f_prompt)
        if not p:
            continue
        item = {"prompt": str(p)}
        if extra_col and extra_col in row:
            item[extra_col] = row[extra_col]
        data.append(item)
    log(f"prompts: {len(data)}")

    rewards = compile_rewards(dt.get("reward_functions") or [])
    if not rewards:
        raise RuntimeError("no usable reward functions in dataset_type")

    info = plan_mod.probe_model(args.model_path)
    n_gpus, free_gib = plan_mod.gpu_inventory()
    params = info["params"] or 7e9
    use_lora = params > 9e9

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        attn = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation=attn)
    model.config.use_cache = False

    peft_config = None
    if use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=48, lora_alpha=96, lora_dropout=0.05,
                                 target_modules="all-linear", task_type="CAUSAL_LM")

    # prompt/completion budgets from the data
    sample = data[: min(200, len(data))]
    p_lens = sorted(len(tokenizer(x["prompt"]).input_ids) for x in sample)
    p95 = p_lens[min(len(p_lens) - 1, int(0.95 * len(p_lens)))]
    max_prompt = min(2048, (p95 // 64 + 2) * 64)
    budget_h = (args.end_ts - time.time()) / 3600
    max_completion = 512 if budget_h >= 1.5 else 384

    # Reward-function shape drives (beta, LR). Verifiable/binary rewards (code
    # execution, math correctness) tolerate much hotter LR and want a firm KL
    # leash; open-ended generic rewards want a gentler LR and moderate KL.
    # The eval score is reward - 0.5*KL, so beta is never far below ~0.1.
    reward_src = " ".join((s.get("reward_func") or "")
                          for s in (dt.get("reward_functions") or [])).lower()
    verifiable = any(k in reward_src for k in
                     ("subprocess", "exec(", "compile(", "assert", "unittest",
                      "== answer", "correct", "test_case", "verify", "sat_", "ded_", "abd_"))
    # Base GRPO LR: the champion's open grpo_config.py size table (0-4b 8e-6,
    # 4-12b 6e-6, 12-15b 5e-6, else lower) — flat, NOT sqrt-scaled. Our old
    # generic path used ~1.2e-6 (7x too low) + 1 epoch; that badly under-trains
    # (same failure DPO had). Verifiable/code rewards get a hotter LR (their
    # separate lrs/grpo_python.json goes up to ~1.6e-3; we stay moderate).
    pb = params / 1e9
    grpo_lr = 8e-6 if pb < 4 else 6e-6 if pb < 12 else 5e-6 if pb < 20 else 4e-6
    beta = 0.5                               # champion default (== validator BETA_GRPO)
    if verifiable:
        beta = 0.1
        grpo_lr = min(2e-4, max(5e-5, grpo_lr * 8))
    lr = grpo_lr * float(os.environ.get("SN56_GRPO_LR_MULT") or 1.0)
    beta = float(os.environ.get("GRPO_BETA") or beta)
    log(f"reward shape: verifiable={verifiable} -> beta={beta} lr={lr:.2e}")

    state = {}
    try:
        with open(args.state_file) as f:
            state = json.load(f)
    except Exception:
        pass
    grpo_state = state.get("grpo", {})
    use_vllm = not grpo_state.get("disable_vllm", False)
    # champion runs 2 generations (spend budget on optimizer steps, not rollouts)
    num_generations = grpo_state.get("num_generations", 2)
    micro = grpo_state.get("batch_size") or num_generations  # divisible by group

    end_ts = args.end_ts
    save_margin = 600  # reward fns can be slow; leave room for the final save
    best = {"reward": -float("inf"), "saved": False}

    def export(trainer, tag):
        os.makedirs(args.output_dir, exist_ok=True)
        trainer.save_model(args.output_dir)
        if is_main:
            tokenizer.save_pretrained(args.output_dir)
            best["saved"] = True
            log(f"exported {tag}")

    class Clock(TrainerCallback):
        def __init__(self):
            self.reward_ema = None
            self.last_save = 0.0

        def on_log(self, targs, tstate, control, logs=None, **kw):
            r = (logs or {}).get("reward")
            if r is None:
                return
            self.reward_ema = r if self.reward_ema is None else 0.8 * self.reward_ema + 0.2 * r
            # periodic export of improving policies (no dev set: rewards are the signal)
            if (self.reward_ema > best["reward"] + 1e-6
                    and time.time() - self.last_save > 900):
                best["reward"] = self.reward_ema
                self.last_save = time.time()
                export(trainer, f"reward_ema={self.reward_ema:.4f}@{tstate.global_step}")

        def on_step_end(self, targs, tstate, control, **kw):
            if time.time() > end_ts - save_margin:
                control.should_training_stop = True
            return control

    def build_trainer(vllm_on: bool):
        vllm_kwargs = {}
        if vllm_on:
            vllm_kwargs = {"use_vllm": True, "vllm_mode": "colocate",
                           "vllm_gpu_memory_utilization": 0.25}
        cfg = GRPOConfig(
            output_dir=os.path.join(paths.WORK_ROOT, "grpo_out", args.task_id),
            per_device_train_batch_size=micro,
            gradient_accumulation_steps=2,
            num_generations=num_generations,
            max_prompt_length=max_prompt,
            max_completion_length=max_completion,
            beta=beta,
            learning_rate=lr,
            lr_scheduler_type="cosine_with_min_lr",       # champion
            lr_scheduler_kwargs={"min_lr_rate": 0.25},
            warmup_ratio=0.03,
            num_train_epochs=int(os.environ.get("SN56_GRPO_EPOCHS") or 4),  # champion uses 4
            weight_decay=0.0,
            optim="paged_adamw_8bit",                     # champion
            bf16=True,
            tf32=True,
            gradient_checkpointing=params > 2.5e9,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            save_strategy="no",
            logging_steps=4,
            report_to=[],
            seed=1337,
            **vllm_kwargs,
        )
        return GRPOTrainer(
            model=model,
            reward_funcs=rewards,
            args=cfg,
            train_dataset=Dataset.from_list(data),
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[Clock()],
        )

    # TRL's colocate vLLM path reads torchrun-style env vars even single-process
    for k, v in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"),
                 ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29517")):
        os.environ.setdefault(k, v)

    trainer = None
    if use_vllm:
        try:
            trainer = build_trainer(True)
        except Exception as e:
            log(f"vLLM init failed ({type(e).__name__}: {e}); HF generation fallback")
            state.setdefault("grpo", {})["disable_vllm"] = True
            with open(args.state_file, "w") as f:
                json.dump(state, f)
    if trainer is None:
        trainer = build_trainer(False)

    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        g = state.setdefault("grpo", {})
        if num_generations > 4:
            g["num_generations"] = 4
            g["batch_size"] = 4
        else:
            g["disable_vllm"] = True
        with open(args.state_file, "w") as f:
            json.dump(state, f)
        if not best["saved"]:
            raise
        log("OOM after an export; keeping the saved policy")

    export(trainer, "final")

    cfg_path = os.path.join(args.output_dir, "config.json")
    if is_main and info["architectures"] and os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            c = json.load(f)
        if c.get("architectures") != info["architectures"]:
            c["architectures"] = info["architectures"]
            with open(cfg_path, "w") as f:
                json.dump(c, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
