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
    # The validator renames the task's extra column to "extra_data" in BOTH the training file and
    # the test file (standardize_grpo_column_names), and its evaluator hands reward functions
    # kwargs["extra_data"]. Looking up the original name ("extra") found nothing, silently
    # dropped the column, and every affine reward returned 0 for the whole run.
    extra_names = ["extra_data"] + ([dt["extra_column"]] if dt.get("extra_column") else [])

    with open(args.data_path) as f:
        rows = json.load(f)
    data = []
    for row in rows:
        p = row.get(f_prompt)
        if not p:
            continue
        item = {"prompt": str(p)}
        for name in extra_names:
            if name in row:
                item["extra_data"] = row[name]
                break
        data.append(item)
    n_extra = sum("extra_data" in d for d in data)
    log(f"prompts: {len(data)} (with extra_data: {n_extra})")
    if dt.get("extra_column") and n_extra == 0:
        log("WARNING: task declares an extra column but no row carries it; extra_data rewards will be 0")

    # Held-out prompts for validator-style checkpoint selection (see DevSelect below).
    import random as _random
    _random.Random(1337).shuffle(data)
    n_dev = int(os.environ.get("SN56_GRPO_DEV") or min(128, max(0, len(data) // 10)))
    if n_dev < 24:
        n_dev = 0
    dev, data = data[:n_dev], data[n_dev:]
    log(f"dev prompts: {len(dev)} train prompts: {len(data)}")

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
    # The evaluator (TRL 1.5.1 GRPOConfig defaults) samples at most 256 new tokens and scores the
    # truncated text. Training on 384-512 token rollouts optimised completions the scorer never
    # sees and spent ~2x the generation time per step.
    max_completion = int(os.environ.get("SN56_GRPO_MAX_COMPLETION") or 256)

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
            paths.patch_adapter_base(args.output_dir, args.base_model_id)
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
            if (selector is None and self.reward_ema > best["reward"] + 1e-6
                    and time.time() - self.last_save > 900):
                best["reward"] = self.reward_ema
                self.last_save = time.time()
                export(trainer, f"reward_ema={self.reward_ema:.4f}@{tstate.global_step}")

        def on_step_end(self, targs, tstate, control, **kw):
            if time.time() > end_ts - save_margin:
                control.should_training_stop = True
            return control

    class DevSelect(TrainerCallback):
        """Pick the shipped policy by the validator's GRPO score on held-out prompts.

        The evaluator samples 2 completions per prompt (temperature 1.0, top_k 0, <=256 new
        tokens, raw prompt), takes the weighted mean reward, and subtracts 0.5 x KL(base || model)
        measured on the PROMPT tokens only (batch 1, truncated to 512). TRL's own KL penalises the
        completion tokens, and the reward EMA is measured on training prompts at a different
        length, so neither tracks what is scored. KL uses a top-k cache of the base distribution
        taken before the first step (the model is still the base then); the lumped-tail estimate
        is a slight lower bound on the exact KL.
        """

        TOPK = 128

        def __init__(self):
            self.base = None
            self.t0 = None
            self.fracs = [0.3, 0.5, 0.65, 0.8, 0.9]
            self.done = 0
            self.eval_cost = 0.0

        def _model(self, m):
            return trainer.accelerator.unwrap_model(m)

        def _shard(self):
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
            return 0, 1

        @torch.no_grad()
        def _prompt_logits(self, m, p):
            enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=512).to(m.device)
            return torch.log_softmax(m(**enc, use_cache=False).logits[0].float(), -1)

        def on_train_begin(self, targs, tstate, control, model=None, **kw):
            self.t0 = time.time()
            m = self._model(model)
            was_training = m.training
            m.eval()
            rank, world = self._shard()
            self.base = []
            for item in dev[rank::world]:
                lp = self._prompt_logits(m, item["prompt"])
                top = lp.topk(self.TOPK, -1)
                tail = torch.log1p(-top.values.exp().sum(-1).clamp(max=1 - 1e-6))
                self.base.append((top.indices.cpu(), top.values.half().cpu(), tail.half().cpu()))
            if was_training:
                m.train()

        @torch.no_grad()
        def score(self, model):
            import torch.distributed as dist
            m = self._model(model)
            was_training = m.training
            m.eval()
            rank, world = self._shard()
            items = dev[rank::world]
            gen = torch.Generator(device=m.device).manual_seed(1234 + rank)
            r_sum, r_n, kl_sum = 0.0, 0, 0.0
            bs = 8
            pad_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(m.device)
            for s0 in range(0, len(items), bs):
                chunk = items[s0:s0 + bs]
                enc = tokenizer([c["prompt"] for c in chunk], return_tensors="pt", padding=True).to(m.device)
                torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
                out = m.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                 max_new_tokens=256, num_return_sequences=2, use_cache=True,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                kw = {"prompts": [c["prompt"] for c in chunk for _ in range(2)]}
                if chunk and "extra_data" in chunk[0]:
                    kw["extra_data"] = [c.get("extra_data") for c in chunk for _ in range(2)]
                total = [0.0] * len(texts)
                for fn in rewards:  # wrappers already carry the task weights
                    for i, v in enumerate(fn(texts, **kw)):
                        total[i] += v
                r_sum += sum(total)
                r_n += len(total)
            tokenizer.padding_side = pad_side
            torch.set_rng_state(rng_cpu)
            torch.cuda.set_rng_state(rng_cuda, m.device)
            for item, (ids, lpb, tail_b) in zip(items, self.base):
                lpf = self._prompt_logits(m, item["prompt"])
                ids = ids.to(m.device)
                lpb = lpb.to(m.device).float()
                tail_b = tail_b.to(m.device).float()
                lpf_top = lpf.gather(-1, ids)
                tail_f = torch.log1p(-lpf_top.exp().sum(-1).clamp(max=1 - 1e-6))
                kl_tok = (lpb.exp() * (lpb - lpf_top)).sum(-1) + tail_b.exp() * (tail_b - tail_f)
                kl_sum += kl_tok.mean().item()
            if was_training:
                m.train()
            t = torch.tensor([r_sum, r_n, kl_sum, len(items)], dtype=torch.float64, device=m.device)
            if world > 1:
                dist.all_reduce(t)
            reward = t[0].item() / max(1.0, t[1].item())
            kl = t[2].item() / max(1.0, t[3].item())
            return reward - 0.5 * kl, reward, kl

        def maybe(self, model, tstate, tag):
            t_eval = time.time()
            sc, rw, kl = self.score(model)
            self.eval_cost = max(self.eval_cost, time.time() - t_eval)
            better = sc > best["reward"] + 1e-9
            log(f"dev[{tag}@{tstate.global_step}] score={sc:.4f} reward={rw:.4f} prompt_kl={kl:.5f} "
                f"({time.time() - t_eval:.0f}s){' BEST' if better else ''}")
            if better:
                best["reward"] = sc
                export(trainer, f"dev_score={sc:.4f}@{tstate.global_step}")
            return better

        def on_step_end(self, targs, tstate, control, model=None, **kw):
            if self.done >= len(self.fracs) or self.t0 is None:
                return control
            span = (end_ts - save_margin) - self.t0
            if time.time() - self.t0 >= self.fracs[self.done] * span:
                self.done += 1
                if time.time() + self.eval_cost + 120 < end_ts - save_margin:
                    self.maybe(model, tstate, f"{self.fracs[self.done - 1]:.2f}")
            return control

    selector = DevSelect() if dev else None

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
            callbacks=[Clock(), selector] if selector else [Clock()],
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

    if selector is not None and selector.t0 is not None:
        # the end-of-training policy competes on the same dev score; ship the best one
        try:
            if not selector.maybe(trainer.model, trainer.state, "final") and best["saved"]:
                log(f"keeping earlier policy (dev score {best['reward']:.4f})")
        except torch.cuda.OutOfMemoryError:
            log("final dev scoring OOM; keeping the saved policy")
        if not best["saved"]:
            export(trainer, "final (no dev improvement recorded)")
    else:
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
