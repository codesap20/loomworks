"""DPO trainer. The validator evaluates with TRL's DPOTrainer on the RAW
prompt/chosen/rejected columns (beta 0.1, reference = the original base model),
so we train the same objective on the same raw text — no extra formatting.

Selection: small stable dev split, best eval_loss checkpoint is exported.
DPO collapses easily, so LR is deliberately conservative and param-scaled.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

import paths
import plan as plan_mod


def log(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[dpo {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def build_pairs(data_path: str, dt: dict) -> tuple[list[dict], list[dict]]:
    with open(data_path) as f:
        rows = json.load(f)
    f_prompt = dt.get("field_prompt") or "prompt"
    f_chosen = dt.get("field_chosen") or "chosen"
    f_rejected = dt.get("field_rejected") or "rejected"

    train, dev = [], []
    n = len(rows)
    dev_target = max(16, min(256, n // 33))
    seen: set[str] = set()
    for row in rows:
        p, c, r = row.get(f_prompt), row.get(f_chosen), row.get(f_rejected)
        # drop degenerate pairs: empty prompt/chosen/rejected (no signal) or
        # identical chosen==rejected (the validator warns these cause random
        # predictions). `not c`/`not r` catches both None and empty-string.
        if not p or not c or not r or c == r:
            continue
        pair = {"prompt": str(p), "chosen": str(c), "rejected": str(r)}
        h = hashlib.blake2b(json.dumps(pair, sort_keys=True).encode(), digest_size=10).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        if int(h[-4:], 16) < 65536 * dev_target / max(n, 1) and len(dev) < dev_target:
            dev.append(pair)
        else:
            train.append(pair)
    if not dev:
        dev = train[:8]
    return train, dev


def main() -> None:
    args = parse_args()
    if args.num_gpus > 1 and "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node", str(args.num_gpus),
                               os.path.abspath(__file__)] + sys.argv[1:])

    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainerCallback)
    from trl import DPOConfig, DPOTrainer

    is_main = int(os.environ.get("RANK", "0")) == 0
    train_rows, dev_rows = build_pairs(args.data_path, json.loads(args.dataset_type))
    log(f"pairs: train={len(train_rows)} dev={len(dev_rows)}")

    info = plan_mod.probe_model(args.model_path)
    n_gpus, free_gib = plan_mod.gpu_inventory()
    params = info["params"] or 7e9
    use_lora = params > 9e9 or (n_gpus == 1 and params > 4e9)

    # DPO LR — hand-tuned size buckets matching the open-sourced champion's
    # dpo_config table (our old 5e-7 peaked ~7.6e-7 at 3B and lost 0.52 vs their
    # 0.004; the validator ranks on DPO loss alone). Not sqrt-scaled: small
    # models want it hotter.
    pb = params / 1e9
    if pb < 1:      lr = 1.35e-5
    elif pb < 2:    lr = 8.7e-6
    elif pb < 4:    lr = 6.5e-6
    elif pb < 5:    lr = 6.25e-6
    elif pb < 9:    lr = 7.5e-6
    elif pb < 12:   lr = 5e-6
    elif pb < 15:   lr = 8.5e-6
    else:           lr = 8e-6
    # Measured on Qwen3-4B (H200, 2026-09-09) with the validator's exact DPO loss:
    # the champion-matched table alone is too COLD. mean DPO loss by multiplier:
    #   0.50 -> 0.04834 | 0.75 -> 0.02787 | 1.00 -> 0.01756 (old default)
    #   1.25 -> 0.01246 | 1.50 -> 0.00856 | 1.75 -> 0.00739  <- minimum, new default
    #   2.00 -> 0.00935 (more pair wins vs 1.75 but WORSE mean; the rule requires the
    #                    sample winner not be worse on the ranking loss, so rejected)
    # 1.75x cuts DPO loss 58% vs the old 1.0x default (0.01756 -> 0.00739).
    # Note chat wanted COOLER lr (0.85x) while DPO wants HOTTER (1.75x) — the
    # champion's single size-bucketed table is optimal for neither task.
    lr *= float(os.environ.get("SN56_DPO_LR_MULT") or 1.75)
    if use_lora:
        lr *= 4

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
    ref_model = None
    if use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=48, lora_alpha=96, lora_dropout=0.05,
                                 target_modules="all-linear", task_type="CAUSAL_LM")
        # TRL uses the disabled adapter as an implicit reference == base model
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, attn_implementation=attn)

    # length budget from a quick tokenization pass over a sample
    sample = train_rows[: min(200, len(train_rows))]
    p_lens = [len(tokenizer(x["prompt"]).input_ids) for x in sample]
    c_lens = [len(tokenizer(x["chosen"]).input_ids) for x in sample]
    p_lens.sort(); c_lens.sort()
    p95 = lambda v: v[min(len(v) - 1, int(0.95 * len(v)))]
    max_prompt = min(2048, (p95(p_lens) // 64 + 2) * 64)
    max_len = min(4096, ((p95(p_lens) + p95(c_lens)) // 64 + 3) * 64)

    state = {}
    try:
        with open(args.state_file) as f:
            state = json.load(f)
    except Exception:
        pass
    micro = state.get("dpo", {}).get("batch_size") or max(
        1, min(8, plan_mod.micro_batch_for(params, max_len, free_gib, not use_lora) // 3))
    accum = max(1, round(32 / (micro * max(1, args.num_gpus))))

    end_ts = args.end_ts
    save_margin = 360
    best = {"loss": float("inf")}

    def export(trainer, tag):
        os.makedirs(args.output_dir, exist_ok=True)
        trainer.save_model(args.output_dir)
        if is_main:
            tokenizer.save_pretrained(args.output_dir)
            log(f"exported {tag} (dev_loss={best['loss']:.5f})")

    probing = {"on": False}   # True while the LR search runs its probe trainings

    class Clock(TrainerCallback):
        def on_step_end(self, targs, tstate, control, **kw):
            if time.time() > end_ts - save_margin:
                control.should_training_stop = True
                control.should_evaluate = True
            return control

        def on_evaluate(self, targs, tstate, control, metrics=None, **kw):
            loss = (metrics or {}).get("eval_loss")
            if probing["on"]:
                return
            if loss is not None and loss < best["loss"] * 0.9995:
                best["loss"] = loss
                export(trainer, f"best@{tstate.global_step}")

    steps_per_epoch = max(1, len(train_rows) // (micro * max(1, args.num_gpus) * accum))
    cfg = DPOConfig(
        output_dir=os.path.join(paths.WORK_ROOT, "dpo_out", args.task_id),
        beta=0.1,
        max_length=max_len,
        max_prompt_length=max_prompt,
        per_device_train_batch_size=micro,
        per_device_eval_batch_size=micro,
        gradient_accumulation_steps=accum,
        num_train_epochs=int(os.environ.get("SN56_DPO_EPOCHS") or 3),  # champion uses 3
        learning_rate=lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.25},   # champion's cosine floor
        warmup_ratio=0.03,
        weight_decay=0.0,                            # champion: wd 0 for DPO
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        optim="paged_adamw_8bit",                    # champion's DPO optimizer
        gradient_checkpointing=params > 2.5e9,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=max(15, steps_per_epoch // 5),
        save_strategy="no",
        logging_steps=20,
        report_to=[],
        seed=1337,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=cfg,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(dev_rows),
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[Clock()],
    )

    # ---- per-task LR search on HELD-OUT DPO loss --------------------------- #
    # DPO is the most LR-sensitive task: the training loss has a degenerate
    # minimiser (a too-hot LR collapses the policy's log-probs to inflate the
    # chosen-rejected margin, so training loss falls while the model degrades),
    # and the validator ranks on held-out DPO loss instead. So every probe here is
    # scored with trainer.evaluate() — TRL's own loss on our dev split, the exact
    # quantity the validator computes — and the pick is a plain argmin with no
    # bias toward hotter LRs. Our fixed multiplier was tuned on ONE synthetic
    # dataset at epoch 0.35; a search transfers to datasets we have never seen.
    # Gated ON with SN56_DPO_LR_SEARCH=1 until validated.
    if os.environ.get("SN56_DPO_LR_SEARCH") == "1" and args.num_gpus == 1:
        import lr_search
        probe_steps = int(os.environ.get("SN56_DPO_SEARCH_STEPS") or 20)
        n_cand = int(os.environ.get("SN56_DPO_SEARCH_N") or 4)
        frac = float(os.environ.get("SN56_DPO_SEARCH_FRAC") or 0.15)
        budget_s = end_ts - time.time() - save_margin
        deadline = time.time() + max(60.0, budget_s * frac)
        cands = lr_search.candidate_lrs(lr, n_cand, 0.3)
        log(f"dpo lr-search: candidates {[f'{c:.2e}' for c in cands]} x {probe_steps} steps")
        snap = {n: p.detach().to("cpu", copy=True)
                for n, p in trainer.model.named_parameters() if p.requires_grad}

        def _restore():
            with torch.no_grad():
                for n, p in trainer.model.named_parameters():
                    if p.requires_grad and n in snap:
                        p.copy_(snap[n].to(p.device, p.dtype))

        results = {}
        probing["on"] = True
        saved_max_steps, saved_lr = cfg.max_steps, cfg.learning_rate
        try:
            for c in cands:
                if time.time() > deadline:
                    log("dpo lr-search: out of budget; using what was measured")
                    break
                _restore()
                # a fresh optimizer + scheduler per candidate, or probe N inherits
                # probe N-1's Adam moments and decayed schedule
                trainer.optimizer, trainer.lr_scheduler = None, None
                trainer.args.max_steps = probe_steps
                trainer.args.learning_rate = c
                trainer.train()
                dl = trainer.evaluate().get("eval_loss", float("inf"))
                results[c] = dl
                log(f"dpo lr-search: lr={c:.2e} held-out dpo_loss={dl:.5f}")
        except Exception as e:
            log(f"dpo lr-search failed ({type(e).__name__}: {e}); keeping {lr:.2e}")
        finally:
            _restore()
            del snap
            torch.cuda.empty_cache()
            probing["on"] = False
            trainer.optimizer, trainer.lr_scheduler = None, None
            trainer.args.max_steps = saved_max_steps
            finite = {k: v for k, v in results.items() if v == v and v != float("inf")}
            if finite:
                lr = min(finite, key=finite.get)
                log(f"dpo lr-search: chose {lr:.2e} (held-out {finite[lr]:.5f})")
            else:
                lr = saved_lr
            trainer.args.learning_rate = lr
            cfg.learning_rate = lr

    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        state.setdefault("dpo", {})["batch_size"] = max(1, micro // 2)
        with open(args.state_file, "w") as f:
            json.dump(state, f)
        if best["loss"] == float("inf"):
            raise
        log("OOM after a successful export; keeping best checkpoint")

    final = trainer.evaluate()
    floss = final.get("eval_loss", float("inf"))
    if floss < best["loss"]:
        best["loss"] = floss
        export(trainer, "final")
    if best["loss"] == float("inf"):
        export(trainer, "fallback")

    # keep the architecture label identical to the base for the finetune check
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
