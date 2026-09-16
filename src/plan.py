"""Sizing heuristics: probe the base model, pick precision/adapter/batch/LR.

Everything here keys off inputs the validator actually provides: the cached
model files, GPU inventory, task type and the hours budget.
"""

import json
import math
import os


def gpu_inventory() -> tuple[int, float]:
    """(count, min free GiB per GPU) without importing torch (callable pre-fork)."""
    try:
        import torch

        n = torch.cuda.device_count()
        free = []
        for i in range(n):
            f, _ = torch.cuda.mem_get_info(i)
            free.append(f / 2**30)
        return n, min(free) if free else 0.0
    except Exception:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        return (len([x for x in visible.split(",") if x != ""]) or 1), 75.0


def probe_model(model_path: str) -> dict:
    """Read config.json + safetensors index to size the model without loading it."""
    info = {"params": None, "vocab": None, "model_type": None, "remote_code": False,
            "max_positions": None, "architectures": []}
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        # multimodal-style configs nest the text config
        text_cfg = cfg.get("text_config", cfg)
        info["model_type"] = cfg.get("model_type")
        info["vocab"] = text_cfg.get("vocab_size")
        info["max_positions"] = text_cfg.get("max_position_embeddings")
        info["remote_code"] = "auto_map" in cfg or "auto_map" in text_cfg
        info["architectures"] = cfg.get("architectures") or []

    total_bytes = 0
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        total_bytes = idx.get("metadata", {}).get("total_size", 0)
    else:
        for name in os.listdir(model_path) if os.path.isdir(model_path) else []:
            if name.endswith(".safetensors") or name.endswith(".bin"):
                total_bytes += os.path.getsize(os.path.join(model_path, name))
    if total_bytes:
        # assume bf16/fp16 shards; fp32 checkpoints just look 2x bigger, which
        # only makes the plan more conservative.
        info["params"] = total_bytes / 2
    return info


def needs_modern_stack(model_info: dict) -> bool:
    """True when the legacy (transformers 4.51.3) stack cannot load this model at all.

    Three cases:
      * quasar_* custom archs and anything with an auto_map (remote code);
      * any model_type the INSTALLED transformers does not know. This is the common one and it cost
        us the 2026-09-14 tournament: task 90d361cb on LiquidAI/LFM2.5-2.6B is plain `lfm2`, a
        natively-supported arch in newer transformers but absent from 4.51.3, so every attempt died
        with KeyError: 'lfm2' (and the emergency submission, also on the legacy stack, with it).
        New architectures land in the model pool constantly, so ask the library instead of
        maintaining a list.
    """
    mt = (model_info.get("model_type") or "").lower()
    if mt.startswith("quasar"):
        return True
    if model_info.get("remote_code"):
        return True
    if mt:
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
            if mt not in CONFIG_MAPPING_NAMES:
                return True
        except Exception:
            pass
    return False


def sft_lr(params: float | None) -> float:
    """Param-scaled SFT peak LR: 2e-5 at 7B, sqrt-scaled, clamped."""
    p = params or 7e9
    lr = 2e-5 * math.sqrt(7e9 / p)
    return float(min(2e-4, max(4e-6, lr)))


def _choose_regime(params: float | None, n_gpus: int, gpu_free_gib: float) -> dict:
    """full-ft vs LoRA and the distribution strategy.

    Memory rule of thumb for AdamW full-ft in bf16 with fp32 master+moments:
    ~16 bytes/param + activations. Distributed capacity = n_gpus * free.
    """
    p = params or 7e9
    need_gib = p * 16 / 2**30 * 1.25
    capacity = n_gpus * max(gpu_free_gib - 8, 10)
    if need_gib <= max(gpu_free_gib - 12, 10):
        return {"adapter": None, "dist": "ddp" if n_gpus > 1 else "single"}
    if need_gib <= capacity:
        return {"adapter": None, "dist": "zero3" if n_gpus > 1 else "single-offload"}
    # too big to full-ft: high-rank LoRA. Multi-GPU MUST shard the frozen base
    # (zero3) — DDP replicates the full base on every GPU, so a 35-71B boss task
    # (the new BOSS_ROUND_LARGE_INSTRUCT band) OOMs at load on 80GB H100s and
    # forfeits. zero3 shards params/grads/optimizer while LoRA keeps the trainable
    # set tiny; the frozen base is what actually needs the sharding. Single-GPU
    # can't hold a 30B+ base anyway (validator gives 4-8x here) but keep offload.
    return {"adapter": {"r": 64, "alpha": 128, "dropout": 0.05},
            "dist": "zero3" if n_gpus > 1 else "single-offload"}


def choose_regime(params: float | None, n_gpus: int, gpu_free_gib: float) -> dict:
    """Regime picker, with a test-only override.

    SN56_FORCE_DIST lets validation exercise a distribution strategy the local box
    would not otherwise pick (e.g. running zero3 at world_size=1 to test the path
    the validator uses for the 14B continuous-SFT gate and the 71B boss instruct).
    Never set in production.
    """
    regime = _choose_regime(params, n_gpus, gpu_free_gib)
    forced = os.environ.get("SN56_FORCE_DIST")
    if forced:
        regime["dist"] = forced
    # SN56_FORCE_ADAPTER=lora|none exercises the other side of the full-ft/LoRA
    # decision on hardware that would not pick it (the champion runs LoRA on every
    # model >=9B, including the 14B continuous-SFT gate, while our memory rule picks
    # full-ft there; this makes the two comparable on one GPU). Never set in production.
    # PRODUCTION preference (not a test override): ask for LoRA even when full-ft fits.
    # Measured on Qwen3-4B chat with the validator's per-sample rule, both arms run to
    # convergence under the champion's schedule:
    #   5,950 rows  full-ft 0.34225 | LoRA r64 0.33389   -> LoRA by 0.0094 (290-32)
    #   39,478 rows full-ft 0.31556 | LoRA r64 0.31172   -> LoRA by 0.0038 (176-41)
    # The edge shrinks as the data grows, as a regularization effect should, but it does
    # not invert. On instruct LoRA merely TIES full-ft (0.95219 vs 0.95207 at r128), so
    # this is applied to chat only. Note the DIST is deliberately left as the memory rule
    # computed it for full-ft: LoRA needs strictly less memory, so that placement is
    # always feasible, and it is exactly what was measured (forcing the large-model LoRA
    # branch would drop a small model onto single-offload and make it slower for nothing).
    if os.environ.get("SN56_ADAPTER_PREF") == "lora" and regime.get("adapter") is None:
        regime["adapter"] = {"r": 64, "alpha": 128, "dropout": 0.05}

    adapter = os.environ.get("SN56_FORCE_ADAPTER")
    if adapter == "lora":
        regime["adapter"] = {"r": 64, "alpha": 128, "dropout": 0.05}
    elif adapter == "none":
        regime["adapter"] = None
    # SN56_LORA_R overrides the rank (alpha tracks 2r unless SN56_LORA_ALPHA is set)
    r = os.environ.get("SN56_LORA_R")
    if r and regime.get("adapter"):
        regime["adapter"]["r"] = int(r)
        regime["adapter"]["alpha"] = int(os.environ.get("SN56_LORA_ALPHA") or 2 * int(r))
    return regime


def micro_batch_for(params: float | None, seq_len: int, gpu_free_gib: float,
                    full_ft: bool, vocab: int | None = None,
                    fused_ce: bool = False) -> int:
    """Activation-based micro-batch guess; the OOM ladder corrects what is left.

    The output logits, not the hidden activations, dominate at modern vocab sizes and
    were missing from this estimate entirely. For Qwen3 (vocab 151936) at seq_len 4096
    one sample's logits are ~4.6 GiB — bf16 from the forward, an fp32 copy for the
    cross-entropy, and a gradient — against ~0.5 GiB of hidden activations for a 4B
    model. Ignoring that put the first guess 3-5x too high: a real in-image run asked
    for a 72.5 GiB allocation and burned two of its five attempts before the ladder
    halved into a batch that fit.
    """
    p = params or 7e9
    weight_overhead = (p * (16 if full_ft else 2.5)) / 2**30
    hidden_gib = max(0.05, (p / 7e9) * (seq_len / 4096) * 0.9)
    # Per-logit peak bytes. Without a fused loss the full logits tensor is materialized
    # in bf16, upcast to fp32 for the cross-entropy, and both carry gradients — measured
    # at ~20 bytes/logit against a real failure (43 rows x 4008 tokens x 151936 vocab in
    # fp32 is 99.8 GiB, exactly the allocation the container asked for). Liger's fused
    # linear cross-entropy never materializes them, chunking the lm_head instead, which
    # is why it is worth enabling: it is the difference between a micro-batch of 10 and
    # one of ~30 on an 80 GB card.
    logits_gib = seq_len * (vocab or 32_000) * (5 if fused_ce else 20) / 2**30
    per_sample_gib = hidden_gib + logits_gib
    room = max(gpu_free_gib - weight_overhead - 6, per_sample_gib)
    mb = int(room / per_sample_gib)
    return max(1, min(mb, 64))
