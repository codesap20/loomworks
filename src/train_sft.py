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
        # winners cool to ~0.25*peak; WSD's flat phase lets us floor lower. SN56_WSD_FLOOR to A/B.
        self.floor = float(os.environ.get("SN56_WSD_FLOOR") or 0.10)

    def factor(self, step: int) -> float:
        if step < self.warmup:
            return step / self.warmup
        if self.decay_start is not None and step >= self.decay_start:
            frac = min(1.0, (step - self.decay_start) / max(1, self.decay_len))
            return 1.0 - (1.0 - self.floor) * frac
        return 1.0


class _SoupDone(Exception):
    """Internal: the uniform soup path finished; skip the greedy loop below."""


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

    # Dedup the dev split (SN56_DEVDEDUP=1) — exact-duplicate (input_ids, labels)
    # rows over-weight whatever they duplicate and add noise to checkpoint
    # selection. The champion dedups its eval split for exactly this reason
    # (champion/scripts/train_instruct.py). Gated so it can be A/B'd; only ever
    # shrinks the dev set, never touches training data or the shipped weights
    # directly. Default off = live behavior unchanged.
    if os.environ.get("SN56_DEVDEDUP") == "1":
        seen, keep = set(), []
        for i, row in enumerate(dev_ds):
            key = (tuple(row["input_ids"]), tuple(row["labels"]))
            if key not in seen:
                seen.add(key)
                keep.append(i)
        if len(keep) < len(dev_ds):
            log(f"dev dedup: {len(dev_ds)} -> {len(keep)}")
            dev_ds = dev_ds.select(keep)

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

    # DATA-DRIVEN LR. Measured 2026-09-19 on three rebuilt live tasks plus a 2x2 model x data
    # cross: the best ABSOLUTE LR is a property of the DATA, not of the model. Both Qwen-0.5B and
    # Llama-1.2B agree per dataset (task1 ~3.8-4.8e-5, task2 ~6-7.5e-5), and the size table's
    # opposite ordering (a higher LR for the SMALLER model) is what made it look like a model
    # effect. Across tasks the optimum tracks supervised tokens per example:
    #     task2 GossipCop  30/ex  -> ~7.0e-5     (x0.5 1.3382, x1.0 1.1575 on Qwen)
    #     task1 indic     133-208 -> ~4.3e-5     (Qwen x0.5 1.2949 vs x1.0 1.3238)
    #     LFM MathFusionQA  534   -> ~3.2e-5     (x0.5 0.3321, x1.0 0.3268, x1.5 0.3425)
    # lr = 1.72e-4 * m^-0.27 fits all three and predicts the winning multiplier in all four cross
    # combinations. Kept within a factor of the size table, which still carries what we know about
    # models far outside this 0.5-2.7B range, and off for LoRA/KL (different objectives, untested).
    _lr_rule = os.environ.get("SN56_LR_RULE", "tokens")
    if _lr_rule == "tokens" and not lora and not use_kl:
        try:
            _n = min(len(train_ds), 2000)
            _step = max(1, len(train_ds) // _n)
            _sup = [sum(1 for x in train_ds[i]["labels"] if x != -100)
                    for i in range(0, len(train_ds), _step)][:_n]
            _m = max(1.0, sum(_sup) / max(1, len(_sup)))
            _pred = 1.72e-4 * _m ** -0.27
            _lr = min(max(_pred, 0.5 * peak_lr), 1.6 * peak_lr)
            log(f"lr rule: {_m:.0f} supervised tokens/example -> {_pred:.2e} "
                f"(table {peak_lr:.2e}, using {_lr:.2e})")
            peak_lr = _lr
        except Exception as exc:
            log(f"lr rule failed ({type(exc).__name__}: {exc}); table LR {peak_lr:.2e}")

    if lora:
        peak_lr = min(2.5e-4, peak_lr * 5)

    # ---- model ------------------------------------------------------------ #
    attn_impl = plan_mod.attn_impl_for(args.model_path)

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
        # .cuda() means cuda:0 on every rank before the Trainer sets devices: on 2+ GPUs all
        # reference copies landed on GPU 0 (OOM there, cross-device logits everywhere else)
        ref_model = load_base().to(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}").eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # ---- batch / step planning -------------------------------------------- #
    seq_len = meta.get("seq_len", 4096)
    # Size on the LONGEST row, not p95: group_by_length is on, so one batch ends up
    # entirely max-length. Sizing on p95 is what produced the 99.8 GiB allocation the
    # OOM ladder then had to halve twice.
    # Liger's fused linear cross-entropy would avoid materializing the logits tensor —
    # this workload's memory ceiling at a 151936-token vocab — and the champion enables
    # it (their instruct_config passes use_liger). MEASURED AND REJECTED for us: the same
    # chat recipe scored 0.99504 with it against 0.98579 without, losing to the champion's
    # own recipe (115-167) where the unfused runs beat it, and it still OOMed once at the
    # larger batch it allows. A 0.009 regression is far more than the batch headroom is
    # worth. Opt in with SN56_LIGER=1 only after re-measuring.
    use_liger = os.environ.get("SN56_LIGER", "0") == "1"
    micro_bs = sft_state.get("micro_batch") or plan_mod.micro_batch_for(
        info["params"], min(seq_len, meta.get("len_max", seq_len)), free_gib, lora is None,
        vocab=info.get("vocab"), fused_ce=use_liger)
    if packing:
        micro_bs = max(1, micro_bs // 2)  # flattened rows are mb x len long

    # PRE-FLIGHT MEMORY PROBE. micro_batch_for() is an estimate, and it reads LoRA as cheap
    # (2.5 bytes/param of overhead) when what actually dominates is ACTIVATIONS, which an adapter
    # does not shrink at all — below 2.5B params gradient checkpointing is off, so every layer's
    # activations are kept. Measured 2026-09-20: it chose micro 6 for a LoRA run on Qwen-0.5B at
    # seq 4096 and the first backward asked for 13.9 GiB it did not have. Live, that costs a whole
    # attempt (round-1 task 067761fe burned 4m47s the same way). One forward+backward at the real
    # shape, with the optimizer states it will later allocate held aside, settles it in seconds.
    use_ckpt = (info["params"] or 0) > 2.5e9
    if (os.environ.get("SN56_MEM_PROBE", "1") == "1" and torch.cuda.is_available()
            and regime["dist"] in ("single", "single-offload")):
        # Probe with the SAME gradient-checkpointing setting training will use, or the probe
        # measures a much larger footprint than the real run: on LFM2.5-2.6B it cut micro 3 -> 1
        # (3x the steps) for runs that had been training fine at 3. When the planned batch does
        # not fit, turning checkpointing ON is cheaper than halving the batch, so try that first.
        _probe_len = int(min(seq_len, max(256, meta.get("len_max", seq_len))))
        # AdamW keeps exp_avg and exp_avg_sq in the PARAMETER dtype (measured on this stack, fused
        # and unfused): 4 bytes/param for our bf16 weights, not 8. Reserving 8 made the probe
        # refuse micro 3 on LFM2.5-2.6B (-> micro 1, half the steps, 0.3309 vs 0.3268) for a batch
        # five 2 h runs had trained at without a single OOM.
        _opt_bytes = sum(p_.numel() * 2 * p_.element_size()
                         for p_ in model.parameters() if p_.requires_grad)
        _vocab = info.get("vocab") or 32000

        def _set_ckpt(on):
            try:
                if on:
                    model.config.use_cache = False
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                else:
                    model.gradient_checkpointing_disable()
            except Exception:
                pass

        def _fits():
            reserve = None
            try:
                if next(model.parameters()).device.type != "cuda":
                    model.cuda()
                torch.cuda.empty_cache()
                # the optimizer moments are allocated at the first step; the probe must not count
                # that memory as free
                reserve = torch.empty(int(_opt_bytes), dtype=torch.uint8, device="cuda")
                ids = torch.randint(0, _vocab, (micro_bs, _probe_len), device="cuda")
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
                out.loss.backward()
                del out, ids
                return True
            except torch.cuda.OutOfMemoryError:
                return False
            except Exception as exc:
                log(f"mem probe skipped ({type(exc).__name__}: {exc})")
                return True
            finally:
                del reserve
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        _set_ckpt(use_ckpt)
        while True:
            if _fits():
                break
            if not use_ckpt:
                use_ckpt = True
                _set_ckpt(True)
                log("mem probe: planned micro-batch does not fit -> gradient checkpointing on")
                continue
            if micro_bs <= 1:
                log("mem probe: still tight at micro-batch 1; going with it")
                break
            micro_bs = max(1, micro_bs // 2)
            log(f"mem probe: OOM -> micro-batch {micro_bs}")
        # if the batch also came down, checkpointing may no longer be needed - prefer the speed
        if use_ckpt and (info["params"] or 0) <= 2.5e9:
            use_ckpt = False
            _set_ckpt(False)
            if not _fits():
                use_ckpt = True
                log("mem probe: keeping gradient checkpointing on")
            else:
                log(f"mem probe: micro-batch {micro_bs} fits without checkpointing")
        _set_ckpt(False)      # Trainer re-enables it from TrainingArguments

    world = max(1, args.num_gpus)
    # Tokens per update, not sequences, is what the field's recipes maximise: rank 4 on the live
    # 0.5B task ran batch 60 of PACKED 2031-token sequences (~120k tokens/update) against our 65
    # short sequences (~26k). Fewer, cleaner updates also delay the overfitting our dev curve shows.
    target_effective = int(os.environ.get("SN56_EFF_BATCH") or 64)
    grad_accum = max(1, round(target_effective / (micro_bs * world)))

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (micro_bs * world * grad_accum)))
    log(f"plan: micro={micro_bs} accum={grad_accum} eff={micro_bs * world * grad_accum} "
        f"steps/epoch={steps_per_epoch} lr={peak_lr:.2e} packing={packing} lora={bool(lora)} "
        f"seq_len={seq_len}")

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
    # ---- budgeted LR search (dev-scored) ---------------------------------- #
    # The champion runs a per-task LR search on every non-DeepSpeed task and only
    # falls back to its size-bucket table under DeepSpeed (see notes). A fixed
    # table cannot match a searched LR, and our own sweeps showed the per-task
    # optimum moves in opposite directions by task. Gated ON with SN56_LR_SEARCH=1
    # until validated; skipped whenever it cannot pay for itself.
    lr_search_frac = float(os.environ.get("SN56_LR_SEARCH_FRAC") or 0.15)
    if (os.environ.get("SN56_LR_SEARCH") == "1" and world == 1
            and regime["dist"] not in ("zero3",) and not use_kl
            and len(train_ds) >= 200 and len(dev_ds) >= 16):
        import lr_search
        if torch.cuda.is_available():
            model.cuda()
        # match the real run's memory profile, or the probe OOMs at a micro-batch
        # that training would have handled
        _ckpt = (info["params"] or 0) > 2.5e9
        if _ckpt:
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # save_margin is set below; the search only runs off-zero3, where it is 300
        _margin = int(os.environ.get("SN56_SAVE_MARGIN") or 300)
        budget_s = args.end_ts - time.time() - _margin
        n_cand = int(os.environ.get("SN56_LR_SEARCH_N") or 4)
        # Probe depth: enough steps to separate LRs, bounded by the search budget.
        # Skip entirely unless the whole search costs less than lr_search_frac of
        # what is left AND the real run still gets its planned epochs.
        t_step_guess = float(os.environ.get("SN56_T_STEP_GUESS") or 0.0)
        probe_steps = int(os.environ.get("SN56_LR_SEARCH_STEPS") or 24)
        try:
            probe_batches, dev_probe = [], []
            order = list(range(len(train_ds)))
            for i in range(probe_steps * grad_accum + 4):
                rows = [train_ds[order[(i * micro_bs + j) % len(train_ds)]]
                        for j in range(micro_bs)]
                probe_batches.append(train_collator(rows))
                if len(probe_batches) >= 64:      # cycle rather than hoard RAM
                    break
            n_dev = min(len(dev_ds), max(16, min(96, len(dev_ds))))
            for i in range(0, n_dev, max(1, micro_bs)):
                rows = [dev_ds[j] for j in range(i, min(i + micro_bs, n_dev))]
                dev_probe.append(pad_collator(rows))

            def _opt_factory(lr):
                decay = 0.0 if champ_sched else float(os.environ.get("SN56_WD") or 0.01)
                params = [p for p in model.parameters() if p.requires_grad]
                return torch.optim.AdamW(params, lr=lr, weight_decay=decay, betas=(0.9, 0.999))

            t0 = time.time()
            deadline = t0 + max(60.0, budget_s * lr_search_frac)
            # Measured 2026-09-09: one candidate at 24 steps x accum cost 583s against a
            # 403s allowance, so the search "finished" having tested a single LR and
            # shipped it. Probe depth must be sized from the real per-step cost, and the
            # whole thing skipped when even a minimal search cannot fit.
            peak_lr, _info = lr_search.search(
                model, probe_batches, dev_probe, peak_lr,
                steps=probe_steps, accum=grad_accum, opt_factory=_opt_factory,
                deadline=deadline, n_candidates=n_cand, log=log)
            log(f"lr-search: chose {peak_lr:.2e} in {time.time() - t0:.0f}s "
                f"(budget {deadline - t0:.0f}s of {budget_s:.0f}s left)")
        except Exception as e:
            log(f"lr search failed ({type(e).__name__}: {e}); heuristic LR {peak_lr:.2e}")
        finally:
            if _ckpt:
                try:
                    model.gradient_checkpointing_disable()
                except Exception:
                    pass

    # Small datasets overfit fast (measured: 3B/alpaca hit its dev-loss floor at
    # ~0.5 epoch then degraded for 3.5 more). Cap epochs tighter for small data
    # and lean on the overfitting early-stop below. Env-overridable for sweeps.
    if len(train_ds) < 4_000:
        epoch_cap = 3
    elif len(train_ds) < 15_000:
        epoch_cap = 2
    else:
        # Measured on both rebuilt live round-1 tasks (1 h, 1xH100, full budget): the 3-epoch cap,
        # not the clock or the early-stop, was what ended the run, and it left ~20 of 57 min
        # unused. Raising it to 6 is neutral-to-better on a clean base (task 87c51c39, 28k rows:
        # 1.3287 vs 1.3353) and a clear win on an augmented one (task 067761fe, 19k rows: 1.0341
        # vs 1.0553; 12 epochs adds nothing beyond 6). The overfitting early-stop below is what
        # actually stops these runs now.
        epoch_cap = 6
    epoch_cap = int(os.environ.get("SN56_EPOCH_CAP") or epoch_cap)
    lr_mult = float(os.environ.get("SN56_LR_MULT") or 1.0)
    peak_lr *= lr_mult
    save_margin = 300 + (600 if regime["dist"] == "zero3" else 0)
    # test-only: lets a short local run exercise the end-of-run soup/select path
    # (zero3's 900s margin would otherwise need a >30min budget). Never set in prod.
    save_margin = int(os.environ.get("SN56_SAVE_MARGIN") or save_margin)
    # measured during the run: seconds per dev eval and per export (trainer.save_model)
    timing = {"eval_s": None, "export_s": None}
    budget_t0 = time.time()

    wsd = WsdPlan(warmup_steps=max(4, int(0.02 * steps_per_epoch * 2)))
    planned = {"total": steps_per_epoch * epoch_cap}

    # ---- trainer ----------------------------------------------------------- #
    class KlTrainer(Trainer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            # SN56_LOSS_NORM=fix: transformers skips loss/grad_accum for models whose forward accepts
            # loss kwargs, expecting compute_loss to pass num_items_in_batch; ours does not, so the
            # update is grad_accum x too large (clipping then makes every step a max-norm step).
            # Default OFF: on chat at micro_batch 1 the "correct" version LOST 20-620 (1.3419 vs
            # 1.3175), so this must be measured per task alongside LR, not flipped blind.
            if os.environ.get("SN56_LOSS_NORM") == "fix":
                self.model_accepts_loss_kwargs = False
                log("loss normalisation: dividing by grad_accum (SN56_LOSS_NORM=fix)")

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss
            if kl_coef > 0.0:
                labels = inputs["labels"]
                mask = labels != -100
                if mask.any():
                    with torch.no_grad():
                        if lora:
                            _m = self.accelerator.unwrap_model(model)
                            with _m.disable_adapter():
                                ref_logits = _m(
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
    final_phase = {"on": False}  # set once training ends: on_evaluate goes inert

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
    # LoRA snapshots are tiny (adapter only), so souping is cheap and safe there —
    # the old `lora is None` guard existed only for full-ft RAM cost. Measured on
    # Qwen3-4B chat (H200): champ+lr0.85+soup beat the champion schedule 381-12 with
    # a 0.0146 mean gap (clears the boss-round 0.01 bar), vs 0.0084 for lr0.85 alone
    # — the soup IS the winning margin. NOTE: still unavailable under zero3 (sharded
    # weights), which is exactly the validator's 14B continuous-SFT regime; closing
    # that is the remaining gap (see notes/scoring-2026-09.md).
    # "single-offload" is the 1-GPU large-LoRA regime (e.g. 14B on one H200); its
    # snapshots are adapter-only and tiny, so in-RAM soup is fine there. It was
    # missing from this list, which silently disabled soup on the 14B LoRA test.
    soup_enabled = (os.environ.get("SN56_USE_SOUP") == "1"
                    and regime["dist"] in ("single", "single-offload", "ddp", "zero3"))
    soup_k = int(os.environ.get("SN56_SOUP_K") or 4)
    if regime["adapter"] is None and not soup_disk_default(info["params"]):
        # bf16 CPU snapshots of every trainable: 4 of them for a 7B full-ft is ~55 GiB of host RAM
        # on top of ema/raw/fp32 copies, and a host OOM kill loses the task outright.
        # 8 snapshots measured better than 4 on both rebuilt live tasks (0.5B clean: 1.3238 vs
        # 1.3287; 1.2B augmented: 1.0298 vs 1.0341, together with 24 evals/run). Kept to <=1.5B:
        # 8 bf16 CPU copies of a 3B model is ~48 GiB of host RAM.
        _p = info["params"] or 0
        # 12 snapshots for <=1.5B (measured at the shipping defaults, unthrottled): Llama-1.2B
        # 1.0004 -> 0.9816, the best score on that task by any arm; Qwen-0.5B 1.2835 -> 1.2789.
        # 12 bf16 copies of 1.5B params is ~36 GiB of host RAM, the ceiling we allow.
        soup_k = 12 if _p <= 1.5e9 else 4 if _p <= 3e9 else 2 if _p <= 8e9 else 1
    # sweep-only: a bigger pool is allowed for small models; the RAM guard above still
    # caps anything over 3B, where 4 bf16 snapshots already risk a host OOM kill.
    _k_env = int(os.environ.get("SN56_SOUP_K") or 0)
    if _k_env:
        soup_k = _k_env if (info["params"] or 0) <= 3e9 else min(_k_env, soup_k)
    # greedy = test each candidate against dev and keep it if dev improves (k selection
    # passes); uniform = average the pool once and read dev once (1 pass). See the soup
    # block for the measurement that motivates having the choice.
    soup_mode = os.environ.get("SN56_SOUP_MODE", "greedy")
    # evals per run and the early-stop patience move together: at 2x the eval
    # density, 3 stale evals is half as much training as before, so scale it.
    # 24 evals/run for small models: more (and less correlated) soup candidates and a finer
    # early-stop, for ~10 s per eval. Measured together with soup_k 8 on both live tasks (above).
    # Larger models keep 12: their evals are slow enough to eat real training time.
    # 36 evals/run for <=1.5B (feeds the 12-snapshot soup; an eval there costs ~4-7 s), 24 up to
    # 3B, 12 above - there an eval is slow enough to eat real training time.
    _pe = info["params"] or 0
    evals_per_run = int(os.environ.get("SN56_EVALS_PER_RUN")
                        or (36 if _pe <= 1.5e9 else 24 if _pe <= 3e9 else 12))
    stale_patience = max(3, round(3 * evals_per_run / 12))
    # sweep-only: raise to let a cool-LR run keep training past the first dev bump
    stale_patience = int(os.environ.get("SN56_STALE_PATIENCE") or stale_patience)
    # the floor scales with density so the default (12) keeps its old value of 12
    # and a denser setting is not silently clamped back to it
    eval_floor = max(2, round(144 / max(1, evals_per_run)))
    soup_pool: list[tuple[float, dict]] = []  # (dev_loss, {name: bf16 cpu tensor})
    # zero3: weights are partitioned, so RAM snapshots are meaningless. Instead each
    # new-best export() (which already gathers the full weights to disk) is copied
    # into a rotating slot; the final greedy soup reads the slots back on rank 0 and
    # loads candidates into the sharded model param-by-param (GatheredParameters),
    # so the existing collective evaluate() loop works unchanged. Every rank keeps the
    # same slot list (losses are all-reduced), only rank 0 touches files.
    soup_disk = soup_enabled and regime["dist"] == "zero3"
    soup_slot_root = os.path.join(paths.WORK_ROOT, "soup_slots", args.task_id)
    soup_slots: list[tuple[float, str]] = []  # (dev_loss, slot_dir), best first

    def _weight_files(d):
        return [f for f in os.listdir(d) if f.endswith(".safetensors") or f == "model.safetensors.index.json"]

    def slot_export(loss: float) -> None:
        """Copy the just-exported full weights into a soup slot (rank 0 files only)."""
        if not soup_disk:
            return
        if len(soup_slots) >= soup_k and loss >= soup_slots[-1][0]:
            return
        slot = os.path.join(soup_slot_root, f"s{len(soup_slots)}_{int(time.time())}")
        if is_main:
            try:
                need = sum(os.path.getsize(os.path.join(args.output_dir, f))
                           for f in _weight_files(args.output_dir))
                if shutil.disk_usage(paths.WORK_ROOT).free < need * 1.2:
                    log("soup slot skipped: low disk")
                    return
                os.makedirs(slot, exist_ok=True)
                for f in _weight_files(args.output_dir):
                    shutil.copy2(os.path.join(args.output_dir, f), os.path.join(slot, f))
            except Exception as e:
                log(f"soup slot copy failed ({type(e).__name__}: {e})")
                return
        soup_slots.append((loss, slot))
        soup_slots.sort(key=lambda x: x[0])
        for _, old in soup_slots[soup_k:]:
            if is_main:
                shutil.rmtree(old, ignore_errors=True)
        del soup_slots[soup_k:]

    def snapshot_trainables(mdl):
        return {n: p.detach().to("cpu", torch.bfloat16)
                for n, p in mdl.named_parameters() if p.requires_grad}

    def maybe_pool(mdl, loss):
        if not soup_enabled or soup_disk:
            return
        if len(soup_pool) < soup_k or loss < max(l for l, _ in soup_pool):
            soup_pool.append((loss, snapshot_trainables(mdl)))
            soup_pool.sort(key=lambda x: x[0])
            del soup_pool[soup_k:]

    def export(trainer, tag: str) -> None:
        if regime["dist"] != "zero3" and not is_main:
            return
        os.makedirs(args.output_dir, exist_ok=True)
        _t_exp = time.time()
        trainer.save_model(args.output_dir)
        timing["export_s"] = max(timing["export_s"] or 0.0, time.time() - _t_exp)
        if is_main:
            tokenizer.save_pretrained(args.output_dir)
            _patch_architectures(args.output_dir, info["architectures"])
            paths.patch_adapter_base(args.output_dir, args.base_model_id)
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
                # eval N times per run. Denser evals cost ~10s each but give the soup
                # more (and less correlated) candidates, a finer best-checkpoint and a
                # finer early-stop — soup is the only lever that survives convergence.
                targs.eval_steps = max(eval_floor, planned["total"] // evals_per_run)
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
            if loss is not None and not math.isfinite(loss):
                log(f"dev loss is {loss}: diverged, stopping training")
                control.should_training_stop = True
                return
            if loss is None or final_phase["on"]:
                # final selection re-uses trainer.evaluate(); those readings must not
                # pool/slot/export/early-stop — the final block exports the winner itself
                return
            _rt = (metrics or {}).get("eval_runtime")
            if _rt:
                timing["eval_s"] = max(timing["eval_s"] or 0.0, float(_rt))
            # The step-13 plan only knows seconds per TRAINING step. At 24 evals/run the evals
            # themselves are minutes, so a clock-bound run would train into save_margin — and the
            # final soup/EMA pass then had no budget and was skipped (live LFM2.5 e2e, 2026-09-19:
            # "final: best_ondisk raw -> pick raw"). Trim the plan so the remaining evals fit.
            if self.t_per_step and timing["eval_s"] and planned["total"]:
                step = tstate.global_step
                every = max(1, targs.eval_steps or 1)
                steps_left = max(0, planned["total"] - step)
                evals_left = steps_left // every + 1
                avail = args.end_ts - save_margin - time.time()
                need = steps_left * self.t_per_step + evals_left * timing["eval_s"]
                if need > avail and steps_left > 8:
                    fit = int(max(8, (avail - evals_left * timing["eval_s"]) / self.t_per_step * 0.95))
                    new_total = step + min(steps_left, fit)
                    if new_total < planned["total"]:
                        planned["total"] = new_total
                        wsd.decay_start = min(wsd.decay_start, int(new_total * 0.72))
                        wsd.decay_len = max(1, new_total - wsd.decay_start)
                        log(f"replan: evals cost {timing['eval_s']:.0f}s each -> total={new_total} "
                            f"decay@{wsd.decay_start}")
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
                slot_export(loss)
            else:
                # overfitting guard: once dev loss sits >3% above best for a few
                # consecutive evals, more training only degrades — stop and keep
                # the best. Frees the remaining budget and avoids shipping a worse
                # late checkpoint.
                best["stale"] = best.get("stale", 0) + 1
                if loss > best["loss"] * 1.03 and best["stale"] >= stale_patience:
                    log(f"early-stop: dev {loss:.4f} > best {best['loss']:.4f} x1.03 "
                        f"for {best['stale']} evals")
                    control.should_training_stop = True

    ds_cfg = None
    if regime["dist"] == "zero3":
        ds_cfg = os.path.join(os.path.dirname(__file__), "..", "ds_config", "zero3.json")

    targs_kw = dict(
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
        gradient_checkpointing=use_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=max(20, steps_per_epoch // 4),
        save_strategy="no",
        logging_steps=25,
        group_by_length=True,
        length_column_name="length",
        neftune_noise_alpha=(float(os.environ["SN56_NEFTUNE"]) if os.environ.get("SN56_NEFTUNE")
                             else (1.0 if len(train_ds) < 20_000 else None)),
        use_liger_kernel=use_liger,
        deepspeed=ds_cfg,
        report_to=[],
        seed=1337,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    # transformers 5 removed some arguments (group_by_length, length_column_name, ...). This trainer
    # also runs under the transformers-5 venv for archs 4.51 cannot load (lfm2), so pass only what
    # the installed version accepts and say what was dropped.
    import inspect
    _accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    _dropped = sorted(k for k in targs_kw if k not in _accepted)
    if _dropped:
        log(f"TrainingArguments: dropping args unknown to this transformers: {_dropped}")
    targs = TrainingArguments(**{k: v for k, v in targs_kw.items() if k in _accepted})

    # Dev metric. HF's eval_loss is token-weighted (long samples dominate); the
    # validator scores the MEAN OF PER-SAMPLE CE and counts per-sample wins.
    # Measured 2026-09-09 (4B chat): the two orderings can flip by ~0.01 between
    # checkpoints, so selecting on the token-weighted number can pick a checkpoint
    # the validator ranks lower. SN56_DEV_METRIC=persample makes every dev reading
    # (best-ckpt, early-stop, soup acceptance, final pick) the validator's mean,
    # computed inside the normal eval pass (no extra forward). Default unchanged.
    # Default persample as of 2026-09-10. Every dev decision — best-checkpoint tracking,
    # the overfitting early-stop, greedy soup acceptance and the final raw/ema/soup pick —
    # reads eval_loss, and the validator scores the mean over EXAMPLES of each example's
    # mean completion CE, not a token-weighted mean. Measured on 4B chat LoRA against the
    # exact boss rule: mean gap 0.00777 (bound 0.00671) vs 0.00694 (0.00583) on the
    # token-weighted metric, and the per-example win rate rose 86.1% -> 89.0%. Decided by
    # 54-15 of the samples that separate the two arms, so directional rather than noise.
    dev_metric = os.environ.get("SN56_DEV_METRIC", "persample")

    # OBJECTIVE ALIGNMENT. The validator scores the MEAN OVER EXAMPLES of each example's
    # mean completion CE, so every held-out row counts once regardless of length. The
    # default training loss is token-weighted (sum of token CE / total tokens), which
    # weights a 3000-token row 30x a 100-token one. Those are different objectives, and
    # the audit showed our binding constraint is exactly the quantity we are NOT training:
    # the mean gap over ALL examples needs to roughly double while our win rate on decided
    # examples is already 86%. This makes the training loss per-example-mean to match.
    # SN56_LOSS=persample; off until measured.
    # REFUTED for the training loss (measured 2026-09-10): per-example weighting scored
    # 0.00666 against 0.00694 for token-weighting, i.e. neutral-to-worse. Token-weighting
    # is the better LEARNING signal — a long row carries more supervised tokens and so more
    # gradient — and you do not have to mirror the evaluation weighting to do well on it.
    # Aligning the SELECTION metric (dev_metric below) DID pay; aligning the loss did not.
    loss_mode = os.environ.get("SN56_LOSS", "tokw")

    # EMBEDDING LR. On the live indic-instruct task the whole gap to the round winner is one
    # script: Devanagari 0.876 vs their 0.822 (n=501) while we BEAT them on Latin (1.699 vs
    # 1.725, n=494), and the gap grows with output length (>512 supervised tokens: +0.056).
    # Their weights show the same shape as ours but a 1.3x larger embedding delta (0.0862 vs
    # 0.067 relative). Indic text fragments into tokens that pretraining barely touched, so those
    # embedding rows need to move further than the rest of the network. SN56_EMBED_LR_MULT
    # scales the LR of the embedding/lm_head group only.
    embed_lr_mult = float(os.environ.get("SN56_EMBED_LR_MULT") or 1.0)
    _EMBED_HINTS = ("embed_tokens", "embed_in", "wte", "word_embeddings", "lm_head", "shared")

    class SftTrainer(trainer_cls):
        def create_optimizer(self):
            if self.optimizer is not None or embed_lr_mult == 1.0:
                return super().create_optimizer()
            decay_names = set(self.get_decay_parameter_names(self.model))
            groups = {}
            for n, p_ in self.model.named_parameters():
                if not p_.requires_grad:
                    continue
                is_emb = any(h in n for h in _EMBED_HINTS)
                key = (is_emb, n in decay_names)
                groups.setdefault(key, []).append(p_)
            cls_, kwargs = type(self).get_optimizer_cls_and_kwargs(self.args)
            kwargs.pop("params", None)
            param_groups = [{"params": ps,
                             "lr": self.args.learning_rate * (embed_lr_mult if is_emb else 1.0),
                             "weight_decay": self.args.weight_decay if dec else 0.0}
                            for (is_emb, dec), ps in groups.items()]
            self.optimizer = cls_(param_groups, **{k: v for k, v in kwargs.items()
                                                   if k not in ("lr", "weight_decay")})
            n_emb = sum(len(ps) for (is_emb, _), ps in groups.items() if is_emb)
            log(f"optimizer: {n_emb} embedding tensors at lr x{embed_lr_mult}")
            return self.optimizer

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            # KL-weighted tasks keep KlTrainer's objective: the validator adds its own KL
            # penalty there and the challenger must not be worse once it is applied, so
            # re-weighting the CE half alone would optimise something it does not score.
            if loss_mode != "persample" or kl_coef > 0.0 or "labels" not in inputs:
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)
            labels = inputs["labels"]
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            sl = outputs.logits[:, :-1, :].float()
            slab = labels[:, 1:]
            ce = F.cross_entropy(sl.reshape(-1, sl.size(-1)), slab.reshape(-1),
                                 ignore_index=-100, reduction="none").view(slab.shape)
            n_sup = (slab != -100).sum(1)
            keep = n_sup > 0
            if not bool(keep.any()):
                loss = sl.sum() * 0.0            # keep the graph, contribute nothing
            else:
                per_sample = ce.sum(1)[keep] / n_sup[keep]
                loss = per_sample.mean()
            return (loss, outputs) if return_outputs else loss

        def get_eval_dataloader(self, eval_dataset=None):
            # eval must not be packed: loss weighting would differ from the validator
            self.data_collator, keep = pad_collator, self.data_collator
            try:
                return super().get_eval_dataloader(eval_dataset)
            finally:
                self.data_collator = keep

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            if dev_metric != "persample":
                return super().prediction_step(model, inputs, prediction_loss_only,
                                               ignore_keys=ignore_keys)
            inputs = self._prepare_inputs(inputs)
            labels = inputs["labels"]
            with torch.no_grad():
                out = model(**{k: v for k, v in inputs.items() if k != "labels"})
                sl = out.logits[:, :-1, :].float()
                slab = labels[:, 1:]
                ce = F.cross_entropy(sl.reshape(-1, sl.size(-1)), slab.reshape(-1),
                                     ignore_index=-100, reduction="none").view(slab.shape)
                mask = slab != -100
                ps = ce.sum(1) / mask.sum(1).clamp(min=1)          # per-sample mean CE
                loss = ce.sum() / mask.sum().clamp(min=1)          # token-weighted (kept)
            # per-sample losses ride along as "predictions" so the standard loop
            # gathers them across ranks (ddp/zero3) for us
            return loss.detach(), ps.detach().unsqueeze(1), None

        def evaluation_loop(self, *a, **k):
            out = super().evaluation_loop(*a, **k)
            if dev_metric == "persample" and out.predictions is not None:
                import numpy as np
                ps = np.asarray(out.predictions, dtype=np.float64).reshape(-1)
                if ps.size:
                    out.metrics["eval_loss_tokw"] = out.metrics.get("eval_loss")
                    out.metrics["eval_loss"] = float(ps.mean())
                    out.metrics["eval_n"] = int(ps.size)
            return out

    def _build_trainer():
        return SftTrainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            data_collator=train_collator,
            processing_class=tokenizer,
            callbacks=[ClockCallback(), EmaCallback()],
        )

    try:
        trainer = _build_trainer()
    except Exception as e:
        # Liger patches the model at Trainer construction and only knows certain
        # architectures. An unsupported one must cost us a warning, not the task.
        if not targs.use_liger_kernel:
            raise
        log(f"liger kernel unavailable for this model ({type(e).__name__}: {e}); continuing without")
        targs.use_liger_kernel = False
        trainer = _build_trainer()

    def _restore(st):
        with torch.no_grad():
            for n_, p_ in model.named_parameters():
                if p_.requires_grad and n_ in st:
                    p_.copy_(st[n_].to(p_.device, p_.dtype))

    # SECOND RUN INTO THE SAME SOUP. These tasks are ended by the overfitting early-stop, not by
    # the clock: the live-task arms stop at ~30 min of a 57 min budget and hand the rest back.
    # Restarting from the ORIGINAL weights with a different data order and pooling both runs'
    # checkpoints costs nothing we were using, and independent fine-tunes of the same base average
    # well (model soups). Off for zero3/disk-soup, where snapshots are sharded or on disk.
    rerun_max = int(os.environ.get("SN56_RERUN_MAX") or 1)   # >1 enables; default off until measured
    rerun_ok = (soup_enabled and not soup_disk and regime["dist"] in ("single", "single-offload")
                and (info["params"] or 0) <= 3e9)
    base_state = snapshot_trainables(model) if (rerun_ok and rerun_max > 1) else None

    cycle = 1
    while True:
        try:
            trainer.train()
        except torch.cuda.OutOfMemoryError:
            state.setdefault("sft", {})["micro_batch"] = max(1, micro_bs // 2)
            with open(args.state_file, "w") as f:
                json.dump(state, f)
            raise
        if base_state is None or cycle >= rerun_max:
            break
        left = args.end_ts - save_margin - time.time()
        if left < max(600.0, 0.45 * (args.end_ts - budget_t0)):
            log(f"rerun: {left / 60:.0f} min left, not enough for another run")
            break
        cycle += 1
        log(f"rerun: {left / 60:.0f} min left -> run {cycle} from the original weights "
            f"(soup pool has {len(soup_pool)})")
        _restore(base_state)
        ema.clear()
        best.update({"loss": float("inf"), "stale": 0, "state": None, "saved_at": 0.0})
        planned["total"] = steps_per_epoch * epoch_cap
        targs.seed = 1337 + cycle          # different shuffle, same data
        trainer = _build_trainer()

    # ---- final selection: best-on-disk vs raw vs EMA vs greedy soup -------- #
    final_phase["on"] = True
    # Measure every candidate as a pure dev-loss reading (each restores a known
    # state), then load + export the single winner once. `best` (the lowest
    # checkpoint seen during training) is already on disk as the safe default.
    def load_weights(state):
        if regime["dist"] == "zero3":
            return load_weights_zero3(state)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in state:
                    p.copy_(state[n].to(p.device, p.dtype))

    def load_weights_zero3(state):
        """Load a full state_dict into a ZeRO-3 partitioned model, one param at a
        time (only one gathered tensor lives at once). Collective: every rank must
        call it; only rank 0 needs `state` populated (modifier_rank broadcasts)."""
        import deepspeed
        # Quiesce first: ZeRO-3's prefetch coordinator leaves params INFLIGHT after
        # the last forward, and GatheredParameters' exit re-partition asserts on
        # them ("Cannot partition a param in flight"). empty_partition_cache()
        # (public engine API) partitions everything and resets that state.
        eng = getattr(trainer, "deepspeed", None) or getattr(trainer, "model_wrapped", None)
        try:
            eng.empty_partition_cache()
        except Exception as e1:
            try:
                eng.optimizer.parameter_offload.get_param_coordinator().release_and_reset_all(eng.module)
            except Exception as e2:
                log(f"zero3 quiesce failed ({type(e1).__name__}: {e1}; {type(e2).__name__}: {e2})")
        # PEFT names a live parameter "...lora_A.<adapter>.weight" but SAVES it as
        # "...lora_A.weight", so a disk slot never matches named_parameters() directly.
        # Measured: 0 of 902 params matched a 504-key adapter state before this.
        import re as _re
        _adapter = getattr(model, "active_adapter", None) or "default"
        if isinstance(_adapter, (list, tuple)):
            _adapter = _adapter[0] if _adapter else "default"

        def _lookup(name):
            if state is None:
                return None
            if name in state:
                return state[name]
            # drop the adapter-name segment wherever PEFT inserted it
            return state.get(_re.sub(rf"\.{_re.escape(str(_adapter))}\.", ".", name))

        matched = [0, 0]
        with torch.no_grad():
            for n, p in model.named_parameters():
                with deepspeed.zero.GatheredParameters([p], modifier_rank=0):
                    src = _lookup(n) if is_main else None
                    if is_main and state is not None:
                        matched[1] += 1
                        matched[0] += 1 if src is not None else 0
                    if is_main and src is not None:
                        if tuple(src.shape) != tuple(p.shape):
                            raise RuntimeError(
                                f"zero3 load shape mismatch at {n}: param {tuple(p.shape)} "
                                f"(ds_status={getattr(p, 'ds_status', None)}) vs state {tuple(src.shape)}")
                        p.copy_(src.to(p.device, p.dtype))
        # A silent no-op here would be invisible: the soup would "succeed" while loading
        # nothing. PEFT's saved adapter keys and named_parameters() do not use the same
        # convention (the adapter name is inserted in one and stripped in the other), so
        # this is a live risk on the LoRA+zero3 path the continuous-SFT gate now takes.
        if is_main and state is not None and matched[0] == 0 and matched[1] > 0:
            raise RuntimeError(
                f"zero3 load matched 0 of {matched[1]} params against a {len(state)}-key state "
                f"(first state keys: {list(state)[:3]}) — key convention mismatch, not a no-op to ignore")
        if is_main and state is not None:
            log(f"zero3 load: matched {matched[0]}/{matched[1]} params")

    def load_slot_state(slot_dir):
        """rank 0: read a slot's safetensors into a CPU fp32 dict; others: None."""
        if not is_main:
            return None
        from safetensors.torch import load_file
        out = {}
        for f in sorted(_weight_files(slot_dir)):
            if f.endswith(".safetensors"):
                out.update({k: v.to(torch.float32) for k, v in load_file(os.path.join(slot_dir, f)).items()})
        return out

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

    candidates = {}  # "best" (on disk) is the baseline
    if regime["dist"] != "zero3":
        # under zero3 the RAM snapshot is sharded placeholders and the last
        # on_evaluate already exported these weights if they were the best
        raw_state = snapshot_trainables(model)
        raw_loss = eval_now()
        candidates["raw"] = (raw_loss, raw_state)

    # save_margin IS the final phase's budget: training stops at end_ts - save_margin, so the old
    # test `now < end_ts - save_margin` was false by construction on every clock-bound run and
    # silently skipped EMA and soup. Budget each candidate by its measured cost instead.
    _eval_s = timing["eval_s"] or 30.0
    _export_reserve = max(60.0, 3.0 * (timing["export_s"] or 20.0))

    # main.run() kills this process AT end_ts, so the final export must finish with slack: a kill
    # mid-save leaves a half-written submission (the smoke run without it ended 3 s from end_ts)
    _slack = 90.0

    def budget_for(n_evals):
        return time.time() + n_evals * _eval_s + _export_reserve + _slack < args.end_ts

    have_budget = budget_for(1)
    if ema_enabled and ema and have_budget:
        load_weights(ema)
        candidates["ema"] = (eval_now(), {n: t.clone() for n, t in ema.items()})

    # The EMA is a different point in weight space from any single checkpoint — it
    # averages along the trajectory rather than across it — and on chat it has been
    # winning the final pick outright while the soup wins on LoRA. Offering it to the
    # greedy soup lets the two combine instead of competing. Added ALONGSIDE the pool
    # (not into its top-k) so it never evicts a checkpoint. Off until measured.
    if (os.environ.get("SN56_SOUP_EMA") == "1" and soup_enabled and ema_enabled
            and "ema" in candidates and soup_pool and not soup_disk):
        ema_bf16 = {n: t.to(torch.bfloat16) for n, t in ema.items()}
        if all(n in soup_pool[0][1] for n in ema_bf16):
            soup_pool.append((candidates["ema"][0], ema_bf16))
            soup_pool.sort(key=lambda x: x[0])
            log(f"soup: EMA added to the pool at dev {candidates['ema'][0]:.5f} "
                f"({len(soup_pool)} candidates)")
        else:
            log("soup: EMA key set does not match the pool; not adding")

    have_budget = budget_for(1)
    if soup_disk and len(soup_slots) >= 2 and have_budget:
        # materialise the disk slots as the pool (rank 0 holds tensors, others None;
        # the loop structure below is identical on every rank so collectives line up)
        soup_pool = [(l, load_slot_state(d)) for l, d in soup_slots]
        log(f"soup: materialised {len(soup_pool)} disk slots "
            f"(losses {' '.join(f'{l:.5f}' for l, _ in soup_pool)})")
    have_budget = budget_for(1)
    if (soup_enabled and len(soup_pool) >= 2 and have_budget and soup_mode != "uniform"
            and not budget_for(len(soup_pool))):
        # greedy costs one eval per candidate; the uniform average costs one in total
        log(f"soup: {len(soup_pool)} greedy evals do not fit the remaining time -> uniform")
        soup_mode = "uniform"
    if soup_enabled and len(soup_pool) >= 2 and have_budget:
        try:
            def _f32(d):
                return None if d is None else {n: t.to(torch.float32).clone() for n, t in d.items()}
            running = _f32(soup_pool[0][1])
            if soup_mode == "uniform":
                # Uniform soup (Wortsman 2022's other variant): average the whole pool
                # in one shot and read the dev loss ONCE. The greedy variant instead
                # tests every candidate against dev and keeps it if dev improves, which
                # is k extra selection passes over a ~250-490 row split. Measured
                # 2026-09-09: feeding greedy more candidates (24/36 evals, k 6/8) drove
                # dev 0.019 BETTER while the held-out score got 0.0027 WORSE — the greedy
                # acceptance is fitting dev noise. Uniform has one selection pass total.
                for _, cand in soup_pool[1:]:
                    if running is not None:
                        for n in running:
                            running[n] += cand[n].to(torch.float32)
                if running is not None:
                    for n in running:
                        running[n] /= len(soup_pool)
                load_weights(running)
                soup_loss = eval_now()
                log(f"soup(uniform): {len(soup_pool)} ckpts -> dev {soup_loss:.5f} "
                    f"(best single {soup_pool[0][0]:.5f})")
                candidates["soup"] = (soup_loss, None if running is None else
                                      {n: t.to(torch.bfloat16) for n, t in running.items()})
                raise _SoupDone
            load_weights(running)
            soup_loss = eval_now()
            n_accepted = 1
            for _, cand in soup_pool[1:]:
                if running is not None:
                    for n in running:
                        if n not in cand or tuple(cand[n].shape) != tuple(running[n].shape):
                            raise RuntimeError(
                                f"soup shape mismatch at {n}: running {tuple(running[n].shape)} "
                                f"vs cand {tuple(cand[n].shape) if n in cand else None}")
                trial = None if running is None else {
                    n: (running[n] * n_accepted + cand[n].to(torch.float32)) / (n_accepted + 1)
                    for n in running}
                load_weights(trial)
                tl = eval_now()
                if tl < soup_loss - 1e-4:
                    running, soup_loss, n_accepted = trial, tl, n_accepted + 1
            log(f"soup: {n_accepted}/{len(soup_pool)} ckpts -> dev {soup_loss:.5f} "
                f"(best single {soup_pool[0][0]:.5f})")
            candidates["soup"] = (soup_loss, None if running is None else
                                  {n: t.to(torch.bfloat16) for n, t in running.items()})
        except _SoupDone:
            pass
        except Exception as e:
            import traceback
            log(f"soup failed ({type(e).__name__}: {e})\n" + traceback.format_exc())
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
        winner = (min(candidates.items(), key=lambda kv: kv[1][0]) if candidates
                  else ("best-ondisk", (best["loss"], None)))
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

    # (SN56_DEV_PASS removed 2026-09-16: measured WORSE than the selected checkpoint — base 0.98598
    # vs dev_pass 0.98801, losing 78-30 of decided samples — and it shipped an unvalidated model,
    # since spending the dev rows leaves nothing to check the result against.)

    if is_main:
        with open(os.path.join(os.path.dirname(args.output_dir), "success.txt"), "w") as f:
            f.write(f"{picked} {best['loss']}\n")
    log("done")


def soup_disk_default(params) -> bool:
    """zero3 already spills soup candidates to disk; only in-RAM pools need the size cap."""
    return False


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
