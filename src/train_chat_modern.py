#!/usr/bin/env python
"""Continuous-SFT ChatTask trainer for custom-arch (quasar) + qwen lineages.

Runs under the /opt/modern venv (torch 2.7.1, transformers 5.9, accelerate,
flash-linear-attention, causal-conv1d, cut-cross-entropy). Full bf16-compute
finetune with fp32 master weights, FSDP1 FULL_SHARD across --num-gpus GPUs,
external (FSDP-side) activation checkpointing, cut-cross-entropy loss with a
plain logits-CE fallback, wall-clock budgeting against --end-ts, a mid-run
safety save and a final full save in HF format.

The script re-execs itself under `accelerate launch` when invoked directly.
Exit code 42 means "modern stack unavailable" so main.py can fall back.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
ACCEL_CONFIG = REPO_ROOT / "ops" / "accelerate" / "fsdp_modern.yaml"
WORKER_ENV = "TCM_WORKER"
EXIT_MODERN_UNAVAILABLE = 42

SEQ_LEN = 4096
MICRO_BATCH = 1          # per GPU; conservative because of the 248320-token vocab
GRAD_ACCUM = 8           # 1 * 8 * 4 GPUs = 32 sequences (~131k tokens) per step
MAX_EPOCHS = 2
WARMUP_FRAC = 0.03
LR_FLOOR = 0.10          # cosine decays to 10% of peak
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
ADAM_BETAS = (0.9, 0.95)
FINAL_SAVE_MARGIN = 12 * 60   # reserve for the final gather+write
MIDSAVE_COST = 6 * 60         # planning reserve for the 60% safety save
MIDSAVE_FRAC = 0.60
MEASURE_START, MEASURE_END = 2, 10  # sync-steps used to measure step time
MAX_BAD_STEPS = 25

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".h5", ".msgpack")


# --------------------------------------------------------------------------- #
# CLI / launcher (stdlib only — heavy imports happen in the worker path)
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ChatTask trainer (modern venv)")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", "--model-path", dest="model", required=True,
                        help="local path to the base checkpoint")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset-type", default="{}")
    parser.add_argument("--task-format", choices=["chat", "instruct"], default="chat",
                        help="chat = ChatTask conversations; instruct = axolotl "
                             "user-defined instruction format (pre-boss quasar task)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--end-ts", type=float, default=0.0)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--base-model-id", default="")
    parser.add_argument("--state-file", default="/tmp/trainer_state.json")
    args, _unknown = parser.parse_known_args(argv)
    return args


def write_state(path, payload):
    try:
        payload = dict(payload)
        payload["updated_at"] = time.time()
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError:
        pass


def check_modern_stack():
    """Exit 42 if the modern stack is unusable so main.py can fall back."""
    import importlib
    for module in ("torch", "transformers", "accelerate", "fla", "causal_conv1d"):
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - any import failure means fallback
            print(f"[launcher] modern stack import failed ({module}): {exc}", flush=True)
            sys.exit(EXIT_MODERN_UNAVAILABLE)
    if not ACCEL_CONFIG.is_file():
        print(f"[launcher] missing accelerate config {ACCEL_CONFIG}", flush=True)
        sys.exit(EXIT_MODERN_UNAVAILABLE)


def relaunch_under_accelerate(args):
    check_modern_stack()
    env = dict(os.environ)
    env[WORKER_ENV] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("OMP_NUM_THREADS", "8")
    if os.path.isdir(args.model):
        env.setdefault("HF_HUB_OFFLINE", "1")
    port = 20000 + (abs(hash(args.task_id)) % 20000)
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--config_file", str(ACCEL_CONFIG),
        "--num_processes", str(max(1, args.num_gpus)),
        "--num_machines", "1",
        "--main_process_port", str(port),
        str(Path(__file__).resolve()),
    ] + sys.argv[1:]
    print(f"[launcher] exec: {' '.join(cmd)}", flush=True)
    os.execvpe(sys.executable, cmd, env)


# --------------------------------------------------------------------------- #
# Model helpers (worker side)
# --------------------------------------------------------------------------- #

def detect_layer_class_names(model):
    """Find the decoder-layer class generically: the ModuleList of >=4 uniform
    children whose children are heaviest (so MoE expert lists don't win)."""
    import torch.nn as nn
    best_names, best_score = None, -1
    for _name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) < 4:
            continue
        class_names = sorted({type(child).__name__ for child in module})
        if len(class_names) > 3:
            continue
        score = sum(p.numel() for p in module[0].parameters())
        if score > best_score:
            best_score, best_names = score, class_names
    return best_names


def try_setup_cce(model, device):
    """Return the CCE loss fn if cut-cross-entropy integrates cleanly, else None.
    Decided once at startup (smoke-tested on GPU); never switched mid-training."""
    import torch
    config = model.config
    if getattr(config, "output_router_logits", False) and getattr(config, "router_aux_loss_coef", 0):
        print("[train] MoE aux-loss active -> using model's own loss path", flush=True)
        return None
    if model.get_output_embeddings() is None or not hasattr(model, model.base_model_prefix):
        return None
    try:
        from cut_cross_entropy import linear_cross_entropy
        e = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        c = torch.randn(128, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        t = torch.randint(0, 128, (2, 16), device=device)
        t[0, :4] = -100
        loss = linear_cross_entropy(e, c, t, shift=1, reduction="mean", impl="cce")
        loss.backward()
        assert torch.isfinite(loss).item()
        return linear_cross_entropy
    except Exception as exc:  # noqa: BLE001 - any failure means fallback path
        print(f"[train] cut-cross-entropy unavailable ({exc}); falling back to logits CE", flush=True)
        return None


def attach_cce_forward(model, linear_cross_entropy):
    """Replace forward with base-model forward + linear_cross_entropy on the
    lm_head weight. Runs inside the FSDP root forward, so with use_orig_params
    the lm_head weight view is unsharded (bf16 under mixed precision)."""
    import types

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        base = getattr(self, self.base_model_prefix)
        outputs = base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        head = self.get_output_embeddings()
        weight = head.weight
        if weight.dtype != hidden.dtype:
            weight = weight.to(hidden.dtype)
        loss = linear_cross_entropy(
            hidden, weight, labels, shift=1, reduction="mean", impl="cce"
        )
        return {"loss": loss}

    model.forward = types.MethodType(forward, model)


def batch_loss(model, batch, use_cce):
    if use_cce:
        return model(**batch)["loss"]
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        use_cache=False,
    )
    return out["loss"] if isinstance(out, dict) else out.loss


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def _is_weight_file(name):
    return name.endswith(WEIGHT_SUFFIXES) or name.endswith(".index.json")


def copy_aux_files(base_dir, out_dir):
    """Copy every non-weight file (config.json, generation_config.json, all
    *.py modeling/configuration code, chat_template.jinja, tokenizer files...)
    from the base checkpoint, overwriting whatever save_pretrained wrote. This
    keeps architectures/auto_map byte-identical for the is_finetune check."""
    for name in os.listdir(base_dir):
        src = os.path.join(base_dir, name)
        if not os.path.isfile(src) or _is_weight_file(name):
            continue
        shutil.copy2(src, os.path.join(out_dir, name))


def save_checkpoint(accelerator, model, tokenizer, out_dir, base_dir, tag):
    """Gather full state dict, write bf16 HF checkpoint atomically into out_dir."""
    import gc
    import torch

    accelerator.wait_for_everyone()
    started = time.time()
    state_dict = accelerator.get_state_dict(model)  # collective; rank0 gets full copy
    if accelerator.is_main_process:
        state_dict = {
            k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
            for k, v in state_dict.items()
        }
        out_dir = out_dir.rstrip("/")
        tmp_dir, old_dir = out_dir + ".tmp", out_dir + ".old"
        parent = os.path.dirname(out_dir)
        if parent:
            os.makedirs(parent, exist_ok=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(tmp_dir, state_dict=state_dict, safe_serialization=True)
        try:
            tokenizer.save_pretrained(tmp_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[save] tokenizer.save_pretrained failed ({exc}); base copy will cover it", flush=True)
        copy_aux_files(base_dir, tmp_dir)
        shutil.rmtree(old_dir, ignore_errors=True)
        if os.path.isdir(out_dir):
            os.rename(out_dir, old_dir)
        os.rename(tmp_dir, out_dir)
        shutil.rmtree(old_dir, ignore_errors=True)
        print(f"[save] {tag} checkpoint written to {out_dir} in {time.time() - started:.0f}s", flush=True)
    del state_dict
    gc.collect()
    accelerator.wait_for_everyone()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def lr_at(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * (LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def build_optimizer(model, base_lr):
    import torch
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if (param.ndim < 2 or "norm" in name.lower() or name.endswith(".bias")) else decay).append(param)
    groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=base_lr, betas=ADAM_BETAS, eps=1e-8)


def run_dev_eval(accelerator, model, dev_loader, use_cce):
    import torch
    model.eval()
    loss_sum = torch.zeros((), device=accelerator.device)
    token_sum = torch.zeros((), device=accelerator.device)
    with torch.no_grad():
        for batch in dev_loader:
            tokens = (batch["labels"] != -100).sum()
            loss = batch_loss(model, batch, use_cce)
            loss_sum += loss.detach().float() * tokens
            token_sum += tokens
    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    token_sum = accelerator.reduce(token_sum, reduction="sum")
    model.train()
    if token_sum.item() > 0:
        return (loss_sum / token_sum).item()
    return float("nan")


def train(args):
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(SRC_DIR))
    from chat_data import (ListDataset, make_collate, prepare_dataset,
                           prepare_instruct_dataset)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(42)

    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM)
    log = accelerator.print
    is_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None

    end_ts = args.end_ts if args.end_ts and args.end_ts > time.time() else time.time() + 3.7 * 3600
    final_deadline = end_ts - FINAL_SAVE_MARGIN

    def state(**kw):
        if accelerator.is_main_process:
            payload = {"task_id": args.task_id, "trainer": "train_chat_modern",
                       "output_dir": args.output_dir, "base_model_id": args.base_model_id}
            payload.update(kw)
            write_state(args.state_file, payload)

    state(phase="loading")
    model_path = args.model
    log(f"[train] model={model_path} data={args.data_path} deadline_in={(end_ts - time.time())/60:.0f}min")

    # Custom-arch checkpoints (quasar) import their sibling config module by
    # absolute name; transformers only auto-resolves relative imports, so the
    # checkpoint dir itself must be importable.
    if os.path.isdir(model_path) and model_path not in sys.path:
        sys.path.insert(0, model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- data (deterministic, done identically on every rank) ----
    seq_len = SEQ_LEN
    if args.task_format == "instruct":
        train_items, dev_items, stats = prepare_instruct_dataset(
            tokenizer, args.data_path, args.dataset_type, seq_len, log=log
        )
    else:
        train_items, dev_items, stats = prepare_dataset(
            tokenizer, args.data_path, args.dataset_type, model_path, seq_len, log=log
        )
    if not train_items:
        log("[train] no trainable rows; emitting base checkpoint unchanged")
        if accelerator.is_main_process:
            shutil.rmtree(args.output_dir, ignore_errors=True)
            shutil.copytree(model_path, args.output_dir, dirs_exist_ok=True)
        accelerator.wait_for_everyone()
        state(phase="done", saved=True, note="no_data")
        return 0

    # ---- model ----
    load_dtype = torch.float32 if args.num_gpus > 1 else torch.bfloat16  # fp32 master weights
    kwargs = dict(trust_remote_code=True, dtype=load_dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="sdpa", **kwargs)
    except (TypeError, ValueError) as exc:
        log(f"[train] sdpa attn request failed ({exc}); retrying with defaults")
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.config.use_cache = False
    num_params = sum(p.numel() for p in model.parameters())
    base_lr = min(2e-5, max(3e-6, 1e-5 * math.sqrt(9e9 / max(num_params, 1))))
    log(f"[train] {num_params/1e9:.2f}B params -> lr {base_lr:.2e}")

    # mirror the validator's sequence-length halving rule
    max_pos = getattr(model.config, "max_position_embeddings", None)
    if isinstance(max_pos, int) and 0 < max_pos < 2 * seq_len:
        log(f"[train] max_position_embeddings={max_pos}: eval will use seq {math.ceil(max_pos/2)}")

    if is_fsdp:
        layer_classes = detect_layer_class_names(model)
        if layer_classes:
            # NOTE: accelerate's set_auto_wrap_policy raises if a listed class is
            # absent from the model, and the _no_split_modules fallback lists the
            # never-instantiated QuasarVisionBlock — so we must pin this here.
            plugin = accelerator.state.fsdp_plugin
            plugin.transformer_cls_names_to_wrap = layer_classes
            os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = ",".join(layer_classes)
            log(f"[train] FSDP wrap classes: {layer_classes}")

    # Cut-cross-entropy is OFF by default: under FSDP2 the lm_head weight is a
    # sharded DTensor, and the manual CCE patch multiplies it against plain-tensor
    # hidden states -> "mixed torch.Tensor and DTensor" crash. The model's native
    # labels-based loss keeps the lm_head matmul inside the module where FSDP2's
    # dispatch handles DTensors; full logits for 9B/4096-seq are only ~2GB. Opt in
    # with SN56_USE_CCE=1 only if you've resolved the DTensor path.
    use_cce = False
    if os.environ.get("SN56_USE_CCE") == "1":
        linear_cross_entropy = try_setup_cce(model, accelerator.device)
        use_cce = linear_cross_entropy is not None
        if use_cce:
            attach_cce_forward(model, linear_cross_entropy)
    log(f"[train] loss path: {'cut-cross-entropy' if use_cce else 'model native labels-loss'}")

    # FSDP2 requires model + optimizer prepared TOGETHER (accelerate remaps the
    # optimizer's param references after the model is converted to DTensors);
    # preparing the model alone raises. Build the optimizer on the unwrapped
    # model, then prepare both (+ loaders) in one call.
    model.train()
    optimizer = build_optimizer(model, base_lr)

    collate = make_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(ListDataset(train_items), batch_size=MICRO_BATCH, shuffle=True,
                              collate_fn=collate, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(ListDataset(dev_items), batch_size=MICRO_BATCH, shuffle=False,
                            collate_fn=collate, num_workers=0) if dev_items else None
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if dev_loader is not None:
        dev_loader = accelerator.prepare(dev_loader)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / GRAD_ACCUM))
    hard_cap = MAX_EPOCHS * steps_per_epoch
    total_steps = hard_cap
    warmup = max(10, int(WARMUP_FRAC * total_steps))
    midsave_step, midsave_done = None, False
    log(f"[train] {stats['train']} rows, {steps_per_epoch} steps/epoch, cap {hard_cap} steps")

    global_step, bad_steps, saved_once = 0, 0, False
    measure_t0, step_time = None, None
    recent_losses, last_loss = [], float("nan")
    stop_reason = None
    state(phase="training", step=0, total_steps=total_steps)

    try:
        for epoch in range(MAX_EPOCHS):
            if stop_reason:
                break
            if hasattr(train_loader, "set_epoch"):
                train_loader.set_epoch(epoch)
            for batch in train_loader:
                with accelerator.accumulate(model):
                    loss = batch_loss(model, batch, use_cce)
                    accelerator.backward(loss)
                    if not accelerator.sync_gradients:
                        continue

                    # ---- optimizer step boundary ----
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    finite = grad_norm is None or math.isfinite(float(grad_norm))
                    if finite:
                        new_lr = lr_at(global_step, base_lr, warmup, total_steps)
                        for group in optimizer.param_groups:
                            group["lr"] = new_lr
                        optimizer.step()
                    else:
                        bad_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    last_loss = float(loss.detach().float())
                    recent_losses.append(last_loss)
                    recent_losses = recent_losses[-50:]

                if global_step == MEASURE_START:
                    measure_t0 = time.time()
                elif global_step == MEASURE_END and measure_t0 is not None:
                    step_time = (time.time() - measure_t0) / (MEASURE_END - MEASURE_START)
                    budget = final_deadline - MIDSAVE_COST - time.time()
                    fit = global_step + max(1, int(budget / max(step_time, 1e-3)))
                    total_steps = max(global_step + 1, min(hard_cap, fit))
                    warmup = min(warmup, max(1, total_steps // 10))
                    midsave_step = int(total_steps * MIDSAVE_FRAC) if total_steps >= 30 else None
                    log(f"[train] {step_time:.2f}s/step -> total {total_steps} steps "
                        f"(cap {hard_cap}), midsave at {midsave_step}")

                if global_step % 10 == 0 or global_step <= 3:
                    avg = sum(recent_losses) / max(1, len(recent_losses))
                    log(f"[train] step {global_step}/{total_steps} loss {last_loss:.4f} "
                        f"(avg {avg:.4f}) lr {optimizer.param_groups[0]['lr']:.2e} "
                        f"left {(final_deadline - time.time())/60:.0f}min")
                    state(phase="training", step=global_step, total_steps=total_steps,
                          loss=last_loss, saved=saved_once)

                if bad_steps > MAX_BAD_STEPS:
                    stop_reason = "too_many_nonfinite_steps"
                elif global_step >= total_steps:
                    stop_reason = "reached_total_steps"
                elif time.time() + (step_time or 30.0) >= final_deadline:
                    stop_reason = "time_budget"
                if stop_reason:
                    break

                if midsave_step and not midsave_done and global_step >= midsave_step:
                    state(phase="mid_save", step=global_step)
                    save_checkpoint(accelerator, model, tokenizer, args.output_dir, model_path, "mid")
                    saved_once, midsave_done = True, True
                    state(phase="training", step=global_step, saved=True)
    except Exception:
        log("[train] training loop failed:\n" + traceback.format_exc())
        stop_reason = stop_reason or "exception"

    log(f"[train] stopping after {global_step} steps ({stop_reason or 'epochs_done'})")

    # ---- final save (always attempt; keep the mid save if this one fails) ----
    try:
        state(phase="final_save", step=global_step)
        save_checkpoint(accelerator, model, tokenizer, args.output_dir, model_path, "final")
        saved_once = True
    except Exception:
        log("[train] final save failed:\n" + traceback.format_exc())

    if saved_once and dev_loader is not None and time.time() < end_ts - 120:
        try:
            dev_loss = run_dev_eval(accelerator, model, dev_loader, use_cce)
            log(f"[train] dev masked-CE: {dev_loss:.4f} over {len(dev_items)} conversations")
        except Exception as exc:  # noqa: BLE001
            log(f"[train] dev eval skipped ({exc})")

    state(phase="done", step=global_step, saved=saved_once, loss=last_loss)
    accelerator.wait_for_everyone()
    if not saved_once:
        log("[train] FAILED: no checkpoint produced")
        return 1
    log("[train] TRAINING_COMPLETE")
    return 0


def main():
    args = parse_args()
    if os.environ.get(WORKER_ENV) != "1":
        relaunch_under_accelerate(args)  # never returns
        return
    try:
        code = train(args)
    except SystemExit:
        raise
    except Exception:
        print("[train] fatal:\n" + traceback.format_exc(), flush=True)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
