"""Instruct/Chat SFT trainer (standard architectures, legacy stack).

Recipe outline:
  * axolotl-tokenized data (see tok_axolotl.py) so formatting == validator eval
  * WSD learning-rate schedule: warmup -> flat -> linear decay, where the decay
    onset is chosen at runtime from measured step time so the schedule finishes
    cooling exactly at the wall-clock deadline
  * bf16 full finetune when the optimizer fits, high-rank LoRA otherwise
  * sequence packing through DataCollatorWithFlattening (FA2 varlen)
  * CPU-shadow EMA of trainable weights; at the end the better of {EMA, best
    raw checkpoint} on the dev split is what gets submitted
  * optional KL(ft||base) penalty matching the validator's USE_KL scoring
"""

import argparse
import json
import math
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F

import paths
import plan as plan_mod


def log(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[sft {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--base-model-id", required=True)
    ap.add_argument("--tokenized-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--end-ts", type=float, required=True)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--state-file", default=paths.STATE_FILE)
    return ap.parse_args()


def read_state(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def maybe_reexec_distributed(args: argparse.Namespace) -> None:
    if args.num_gpus > 1 and "RANK" not in os.environ:
        cmd = ["torchrun", "--nproc_per_node", str(args.num_gpus),
               os.path.abspath(__file__)] + sys.argv[1:]
        log(f"re-exec: {' '.join(cmd)}")
        os.execvp("torchrun", cmd)


# --------------------------------------------------------------------------- #
# schedule: warmup -> flat -> runtime-planned linear decay                     #
# --------------------------------------------------------------------------- #

class WsdPlan:
    """Mutable schedule state shared between the lambda and the clock callback."""

    def __init__(self, warmup_steps: int):
        self.warmup = max(1, warmup_steps)
        self.decay_start: int | None = None
        self.decay_len: int = 1
        self.floor = 0.10  # winners cool to ~0.25*peak; WSD's flat phase lets us floor lower

    def factor(self, step: int) -> float:
        if step < self.warmup:
            return step / self.warmup
        if self.decay_start is not None and step >= self.decay_start:
            frac = min(1.0, (step - self.decay_start) / max(1, self.decay_len))
            return 1.0 - (1.0 - self.floor) * frac
        return 1.0


def main() -> None:
    args = parse_args()
    maybe_reexec_distributed(args)

    from datasets import load_from_disk
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorWithFlattening, Trainer,
                              TrainerCallback, TrainingArguments)

    state = read_state(args.state_file)
    sft_state = state.get("sft", {})
    is_main = int(os.environ.get("RANK", "0")) == 0

    train_ds = load_from_disk(os.path.join(args.tokenized_dir, "train"))
    dev_ds = load_from_disk(os.path.join(args.tokenized_dir, "dev"))
    train_ds = train_ds.map(lambda r: {"length": len(r["input_ids"])})
    with open(os.path.join(args.tokenized_dir, "meta.json")) as f:
        meta = json.load(f)

    info = plan_mod.probe_model(args.model_path)
    n_gpus, free_gib = plan_mod.gpu_inventory()
    regime = plan_mod.choose_regime(info["params"], n_gpus, free_gib)
    use_kl = os.environ.get("USE_KL") == "1"
    kl_coef = float(os.environ.get("KL_COEF") or 0.1) if use_kl else 0.0
    if use_kl and regime["adapter"] is None and (info["params"] or 0) > 4.5e9:
        # a frozen reference copy would not fit next to full-ft states
        regime["adapter"] = {"r": 64, "alpha": 128, "dropout": 0.05}
        log("KL task on large model -> switching to LoRA for a free reference")

    lora = regime["adapter"]
    # Champion-schedule mode (env-gated experiment): replicate their FULL instruct
    # package, not just LR — cosine_with_min_lr(0.25) + their LR table +
    # paged_adamw_8bit + wd 0. Their hot 7.5e-5 only pays off inside this schedule
    # (LR-in-isolation on our WSD overshot: 1.062 vs our 1.018).
    champ_sched = os.environ.get("SN56_CHAMP_SCHED") == "1"
    if champ_sched:
        pb = (info["params"] or 7e9) / 1e9
        peak_lr = (1.0e-4 if pb < 2 else 7.5e-5 if pb < 4 else 7.0e-5
                   if pb < 5 else 3.5e-5 if pb < 9 else 1.0e-4 if pb < 15 else 8.0e-5)
    else:
        peak_lr = plan_mod.sft_lr(info["params"])
    if lora:
        peak_lr = min(2.5e-4, peak_lr * 5)

    # ---- model ------------------------------------------------------------ #
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        attn_impl = "sdpa"

    def load_base():
        try:
            return AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16, attn_implementation=attn_impl)
        except Exception as e:
            log(f"{attn_impl} load failed ({e}); retrying with sdpa")
            return AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")

    model = load_base()
    model.config.use_cache = False
    # Length-conditional sequence packing (measured head-to-head vs champion on
    # Qwen2.5-3B): flattening-packing HELPS short-sequence tasks (alpaca hidden
    # 1.018 packed vs 1.024 padded) but HURTS long-sequence tasks (dolly 1.693
    # padded vs 1.704 packed) — long varied docs pack worse / stress varlen. So
    # pack only when the data is short; pad otherwise. Overridable: SN56_PACK=1/0.
    # The long TAIL is what makes packing hurt (varlen stress on the big docs),
    # not the median — dolly p95 was only 546 but p99=1148/max=3980 and packing
    # hurt it, while alpaca is uniformly short and packing helped. Gate on p99.
    _pack_env = os.environ.get("SN56_PACK")
    _short_data = meta.get("len_p99", 4096) <= 900
    packing = (attn_impl == "flash_attention_2" and not use_kl
               and (_pack_env == "1" or (_pack_env is None and _short_data)))
    log(f"packing={packing} (len_p99={meta.get('len_p99')}, attn={attn_impl})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules="all-linear", task_type="CAUSAL_LM"))
        if is_main:
            model.print_trainable_parameters()

    ref_model = None
    if use_kl and not lora:
        ref_model = load_base().cuda().eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # ---- batch / step planning -------------------------------------------- #
    seq_len = meta.get("seq_len", 4096)
    micro_bs = sft_state.get("micro_batch") or plan_mod.micro_batch_for(
        info["params"], min(seq_len, meta.get("len_p95", seq_len)), free_gib, lora is None)
    if packing:
        micro_bs = max(1, micro_bs // 2)  # flattened rows are mb x len long
    world = max(1, args.num_gpus)
    target_effective = 64
    grad_accum = max(1, round(target_effective / (micro_bs * world)))

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (micro_bs * world * grad_accum)))

    from transformers import DataCollatorForSeq2Seq
    _MODEL_KEYS = ("input_ids", "attention_mask", "labels")

    def strip_extras(inner):
        def collate(features):
            return inner([{k: v for k, v in f.items() if k in _MODEL_KEYS} for f in features])
        return collate

    pad_collator = strip_extras(
        DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    train_collator = strip_extras(DataCollatorWithFlattening()) if packing else pad_collator

    # Empirical LR range probe — DEFAULT OFF. Measured on sim-alpaca (Qwen3-0.6B,
    # hidden-eval masked-CE): the ramp-test knee overestimates the best FINAL LR,
    # so the probe picked ~2.7x too hot and scored 2.14 vs the heuristic's 1.67
    # (worse than the untrained base's 1.83). The param-scaled heuristic wins.
    # Opt in with SN56_USE_PROBE=1 only after re-validating on the target model.
    budget_s = args.end_ts - time.time()
    if (os.environ.get("SN56_USE_PROBE") == "1"
            and world == 1 and not use_kl and budget_s > 1800 and len(train_ds) >= 200):
        import lr_probe
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            model.cuda()
        probe_batches = []
        for i in range(6):
            rows = [train_ds[(i * micro_bs + j) % len(train_ds)] for j in range(micro_bs)]
            probe_batches.append(train_collator(rows))
        try:
            peak_lr = lr_probe.lr_range_probe(model, probe_batches, peak_lr, log=log)
        except Exception as e:
            log(f"lr probe failed ({type(e).__name__}: {e}); heuristic LR {peak_lr:.2e}")
    # Small datasets overfit fast (measured: 3B/alpaca hit its dev-loss floor at
    # ~0.5 epoch then degraded for 3.5 more). Cap epochs tighter for small data
    # and lean on the overfitting early-stop below. Env-overridable for sweeps.
    if len(train_ds) < 4_000:
        epoch_cap = 3
    elif len(train_ds) < 15_000:
        epoch_cap = 2
    else:
        epoch_cap = 3
    epoch_cap = int(os.environ.get("SN56_EPOCH_CAP") or epoch_cap)
    lr_mult = float(os.environ.get("SN56_LR_MULT") or 1.0)
    peak_lr *= lr_mult
    save_margin = 300 + (600 if regime["dist"] == "zero3" else 0)

    wsd = WsdPlan(warmup_steps=max(4, int(0.02 * steps_per_epoch * 2)))
    planned = {"total": steps_per_epoch * epoch_cap}

    # ---- trainer ----------------------------------------------------------- #
    class KlTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss
            if kl_coef > 0.0:
                labels = inputs["labels"]
                mask = labels != -100
                if mask.any():
                    with torch.no_grad():
                        if lora:
                            with model.disable_adapter():
                                ref_logits = model(
                                    input_ids=inputs["input_ids"],
                                    attention_mask=inputs.get("attention_mask")).logits
                        else:
                            ref_logits = ref_model(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs.get("attention_mask")).logits
                    ft_logp = F.log_softmax(outputs.logits.float(), dim=-1)
                    ref_logp = F.log_softmax(ref_logits.float(), dim=-1)
                    kl_tok = (ft_logp.exp() * (ft_logp - ref_logp)).sum(-1)
                    kl = (kl_tok * mask).sum() / mask.sum()
                    loss = loss + kl_coef * kl
            return (loss, outputs) if return_outputs else loss

        def create_scheduler(self, num_training_steps, optimizer=None):
            if champ_sched:   # let TrainingArguments' cosine_with_min_lr take over
                return super().create_scheduler(num_training_steps, optimizer)
            opt = optimizer or self.optimizer
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda step: wsd.factor(step))
            return self.lr_scheduler

    trainer_cls = KlTrainer

    # EMA shadow of trainable weights on CPU (skipped under zero3: weights are sharded)
    ema_enabled = regime["dist"] != "zero3"
    ema: dict[str, torch.Tensor] = {}
    ema_interval = 2 if lora else 8

    class EmaCallback(TrainerCallback):
        def on_step_end(self, targs, tstate, control, model=None, **kw):
            if not ema_enabled or tstate.global_step % ema_interval:
                return
            half_life = max(50, int(0.10 * planned["total"]))
            decay = 0.5 ** (ema_interval / half_life)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    cpu = p.detach().to("cpu", torch.float32, non_blocking=False)
                    if name in ema:
                        ema[name].mul_(decay).add_(cpu, alpha=1 - decay)
                    else:
                        ema[name] = cpu.clone()

    best = {"loss": float("inf"), "saved_at": 0.0}

    # Per-sample-aware final selection (SN56_SELECT=persample). Knockouts/boss
    # tasks are now decided PER held-out SAMPLE (win more samples, 0.01-nat
    # deadzone), not on mean loss — so among candidates tied on mean we prefer the
    # lightest bad tail (lowest p90 per-sample loss), which wins more samples.
    # Gated OFF by default (mean); single-GPU + non-zero3 only (per-sample eval
    # must see the whole dev set on one rank). Bounded by a mean band so it never
    # ships a worse-on-mean model beyond dev noise (validator: sample winner must
    # not be worse on the ranking loss).
    select_mode = os.environ.get("SN56_SELECT", "mean").lower()
    persample_sel = (select_mode == "persample" and args.num_gpus == 1
                     and regime["dist"] != "zero3")

    # Greedy checkpoint soup (Wortsman 2022): keep the K lowest-dev-loss weight
    # snapshots; at the end greedily average them, accepting a candidate only if
    # held-out dev loss improves (so the soup is never worse than the best single).
    # DEFAULT OFF: measured on 3B/alpaca vs champion, the post-early-stop
    # checkpoints are too correlated to benefit — soup matched (never beat) the
    # best single (1.018 either way) while adding K-snapshot RAM (OOM risk on
    # 7-12B) + eval-noise candidates. Opt in with SN56_USE_SOUP=1.
    soup_enabled = (os.environ.get("SN56_USE_SOUP") == "1"
                    and regime["dist"] in ("single", "ddp") and lora is None)
    soup_k = 4
    soup_pool: list[tuple[float, dict]] = []  # (dev_loss, {name: bf16 cpu tensor})

    def snapshot_trainables(mdl):
        return {n: p.detach().to("cpu", torch.bfloat16)
                for n, p in mdl.named_parameters() if p.requires_grad}

    def maybe_pool(mdl, loss):
        if not soup_enabled:
            return
        if len(soup_pool) < soup_k or loss < max(l for l, _ in soup_pool):
            soup_pool.append((loss, snapshot_trainables(mdl)))
            soup_pool.sort(key=lambda x: x[0])
            del soup_pool[soup_k:]

    def export(trainer, tag: str) -> None:
        if regime["dist"] != "zero3" and not is_main:
            return
        os.makedirs(args.output_dir, exist_ok=True)
        trainer.save_model(args.output_dir)
        if is_main:
            tokenizer.save_pretrained(args.output_dir)
            _patch_architectures(args.output_dir, info["architectures"])
            log(f"exported ({tag}) dev_loss={best['loss']:.5f}")

    class ClockCallback(TrainerCallback):
        def __init__(self):
            self.t0 = None
            self.t_per_step = None

        def on_step_end(self, targs, tstate, control, **kw):
            step = tstate.global_step
            if step == 3:
                self.t0 = time.time()
            elif step == 13 and self.t0:
                self.t_per_step = (time.time() - self.t0) / 10
                remaining = args.end_ts - time.time() - save_margin
                achievable = int(remaining / self.t_per_step * 0.9) + step
                planned["total"] = max(step + 8, min(steps_per_epoch * epoch_cap, achievable))
                wsd.decay_start = int(planned["total"] * 0.72)
                wsd.decay_len = planned["total"] - wsd.decay_start
                # eval ~12x/run so the early dev-loss minimum is actually sampled
                targs.eval_steps = max(12, planned["total"] // 12)
                if hasattr(tstate, "eval_steps"):
                    tstate.eval_steps = targs.eval_steps
                log(f"replan: t/step={self.t_per_step:.2f}s total={planned['total']} "
                    f"decay@{wsd.decay_start} eval_every={targs.eval_steps}")
            if planned["total"] and step >= planned["total"]:
                control.should_training_stop = True
                control.should_evaluate = True
            if time.time() > args.end_ts - save_margin:
                control.should_training_stop = True
            return control

        def on_evaluate(self, targs, tstate, control, metrics=None, **kw):
            loss = (metrics or {}).get("eval_loss")
            if loss is None:
                return
            maybe_pool(trainer.model, loss)
            if loss < best["loss"] * 0.999:
                # save every genuine improvement (a 3B save is ~10s; the old
                # 600s throttle could skip the true minimum on short runs)
                best["loss"] = loss
                best["saved_at"] = time.time()
                best["stale"] = 0
                if persample_sel:
                    # keep the best checkpoint in memory too, so per-sample
                    # selection can compare it against raw/ema without a disk reload
                    best["state"] = snapshot_trainables(trainer.model)
                export(trainer, f"best@{tstate.global_step}")
            else:
                # overfitting guard: once dev loss sits >3% above best for a few
                # consecutive evals, more training only degrades — stop and keep
                # the best. Frees the remaining budget and avoids shipping a worse
                # late checkpoint.
                best["stale"] = best.get("stale", 0) + 1
                if loss > best["loss"] * 1.03 and best["stale"] >= 3:
                    log(f"early-stop: dev {loss:.4f} > best {best['loss']:.4f} x1.03 "
                        f"for {best['stale']} evals")
                    control.should_training_stop = True

    ds_cfg = None
    if regime["dist"] == "zero3":
        ds_cfg = os.path.join(os.path.dirname(__file__), "..", "ds_config", "zero3.json")

    targs = TrainingArguments(
        output_dir=os.path.join(paths.WORK_ROOT, "hf_out", args.task_id),
        per_device_train_batch_size=micro_bs,
        per_device_eval_batch_size=max(1, micro_bs),
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epoch_cap,
        learning_rate=peak_lr,
        weight_decay=(0.0 if champ_sched else float(os.environ.get("SN56_WD") or 0.01)),
        max_grad_norm=1.0,
        optim=("paged_adamw_8bit" if champ_sched else "adamw_torch_fused"),
        lr_scheduler_type=("cosine_with_min_lr" if champ_sched else "linear"),
        lr_scheduler_kwargs=({"min_lr_rate": 0.25} if champ_sched else None),
        warmup_ratio=(0.03 if champ_sched else 0.0),
        bf16=True,
        tf32=True,
        gradient_checkpointing=(info["params"] or 0) > 2.5e9,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=max(20, steps_per_epoch // 4),
        save_strategy="no",
        logging_steps=25,
        group_by_length=True,
        length_column_name="length",
        neftune_noise_alpha=(float(os.environ["SN56_NEFTUNE"]) if os.environ.get("SN56_NEFTUNE")
                             else (1.0 if len(train_ds) < 20_000 else None)),
        deepspeed=ds_cfg,
        report_to=[],
        seed=1337,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    class SftTrainer(trainer_cls):
        def get_eval_dataloader(self, eval_dataset=None):
            # eval must not be packed: loss weighting would differ from the validator
            self.data_collator, keep = pad_collator, self.data_collator
            try:
                return super().get_eval_dataloader(eval_dataset)
            finally:
                self.data_collator = keep

    trainer = SftTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=train_collator,
        processing_class=tokenizer,
        callbacks=[ClockCallback(), EmaCallback()],
    )

    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        state.setdefault("sft", {})["micro_batch"] = max(1, micro_bs // 2)
        with open(args.state_file, "w") as f:
            json.dump(state, f)
        raise

    # ---- final selection: best-on-disk vs raw vs EMA vs greedy soup -------- #
    # Measure every candidate as a pure dev-loss reading (each restores a known
    # state), then load + export the single winner once. `best` (the lowest
    # checkpoint seen during training) is already on disk as the safe default.
    def load_weights(state):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in state:
                    p.copy_(state[n].to(p.device, p.dtype))

    def eval_now():
        return trainer.evaluate().get("eval_loss", float("inf"))

    def per_sample_ce():
        """Per-held-out-sample mean CE over completion (label) tokens — the exact
        quantity the validator compares sample-by-sample. Current model weights."""
        m = trainer.model
        m.eval()
        dev = next(m.parameters()).device
        out = []
        with torch.no_grad():
            for batch in trainer.get_eval_dataloader():
                ids = batch["input_ids"].to(dev)
                am = batch.get("attention_mask")
                am = am.to(dev) if am is not None else None
                lab = batch["labels"].to(dev)
                logits = m(input_ids=ids, attention_mask=am).logits
                sl = logits[:, :-1, :].float()
                slab = lab[:, 1:]
                ce = F.cross_entropy(sl.reshape(-1, sl.size(-1)), slab.reshape(-1),
                                     ignore_index=-100, reduction="none").view(slab.shape)
                cnt = (slab != -100).sum(1).clamp(min=1)
                out.append((ce.sum(1) / cnt).float().cpu())
        return torch.cat(out) if out else torch.tensor([])

    raw_state = snapshot_trainables(model)
    raw_loss = eval_now()
    candidates = {"raw": (raw_loss, raw_state)}  # "best" (on disk) is the baseline

    have_budget = time.time() < args.end_ts - save_margin
    if ema_enabled and ema and have_budget:
        load_weights(ema)
        candidates["ema"] = (eval_now(), {n: t.clone() for n, t in ema.items()})

    if soup_enabled and len(soup_pool) >= 2 and have_budget:
        try:
            running = {n: t.to(torch.float32).clone() for n, t in soup_pool[0][1].items()}
            load_weights(running)
            soup_loss = eval_now()
            n_accepted = 1
            for _, cand in soup_pool[1:]:
                trial = {n: (running[n] * n_accepted + cand[n].to(torch.float32)) / (n_accepted + 1)
                         for n in running}
                load_weights(trial)
                tl = eval_now()
                if tl < soup_loss - 1e-4:
                    running, soup_loss, n_accepted = trial, tl, n_accepted + 1
            log(f"soup: {n_accepted}/{len(soup_pool)} ckpts -> dev {soup_loss:.5f} "
                f"(best single {soup_pool[0][0]:.5f})")
            candidates["soup"] = (soup_loss, {n: t.to(torch.bfloat16) for n, t in running.items()})
        except Exception as e:
            log(f"soup failed ({type(e).__name__}: {e})")
    soup_pool.clear()

    if persample_sel and best.get("state") is not None:
        # make the on-disk best a first-class candidate so the per-sample winner
        # can't be silently overridden by the mean-gate below
        candidates.setdefault("best", (best["loss"], best["state"]))

    if persample_sel and len(candidates) > 1:
        # among candidates statistically tied on mean (within dev noise), ship the
        # one with the lightest bad tail — it wins more individual held-out samples
        best_mean = min(v[0] for v in candidates.values())
        band = max(0.006, best_mean * 0.006)  # ~ observed run-to-run dev noise
        tails = {}
        for name, (m, st) in candidates.items():
            if m <= best_mean + band:
                load_weights(st)
                ps = per_sample_ce()
                tails[name] = float(torch.quantile(ps, 0.90)) if ps.numel() else float("inf")
        picked = min(tails, key=tails.get)
        picked_loss, picked_state = candidates[picked]
        log("persample-select: " + " ".join(
            f"{k}(mean={candidates[k][0]:.5f},p90={tails[k]:.4f})" for k in tails)
            + f" -> pick {picked}")
        load_weights(picked_state)
        best["loss"] = min(best["loss"], picked_loss)
        export(trainer, f"{picked}-persample")
    else:
        winner = min(candidates.items(), key=lambda kv: kv[1][0])
        picked, (picked_loss, picked_state) = winner[0], winner[1]
        log(f"final: best_ondisk={best['loss']:.5f} " +
            " ".join(f"{k}={v[0]:.5f}" for k, v in candidates.items()) + f" -> pick {picked}")
        if picked_loss < best["loss"]:
            load_weights(picked_state)
            best["loss"] = picked_loss
            export(trainer, f"{picked}-final")
        else:
            picked = "best-ondisk"  # the training-time best checkpoint already on disk wins
    if best["saved_at"] == 0.0 and not os.path.isdir(args.output_dir):
        export(trainer, "fallback-final")

    if is_main:
        with open(os.path.join(os.path.dirname(args.output_dir), "success.txt"), "w") as f:
            f.write(f"{picked} {best['loss']}\n")
    log("done")


def _patch_architectures(out_dir: str, architectures: list) -> None:
    """Keep the submitted config's architectures identical to the base model's
    (the validator compares configs for its is_finetune check; PEFT/save quirks
    must not rename the arch)."""
    cfg_path = os.path.join(out_dir, "config.json")
    if not architectures or not os.path.isfile(cfg_path):
        return
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get("architectures") != architectures:
        cfg["architectures"] = architectures
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
