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
    """Custom remote-code archs (quasar_*) require the transformers-v5 venv."""
    mt = (model_info.get("model_type") or "").lower()
    if mt.startswith("quasar"):
        return True
    if model_info.get("remote_code"):
        return True
    return False


def sft_lr(params: float | None) -> float:
    """SFT peak LR — the champion's open instruct_config.py size table (their
    winning values, NOT sqrt-scaled). Our old 2e-5*sqrt heuristic gave ~3e-5 at
    3B, ~2.5x below their 7.5e-5 — the same too-timid LR that lost DPO/GRPO.
    Overridable with SN56_SFT_LR_MULT for sweeps."""
    import os
    p = (params or 7e9) / 1e9
    if p < 2:      lr = 1.0e-4
    elif p < 4:    lr = 7.5e-5
    elif p < 5:    lr = 7.0e-5
    elif p < 9:    lr = 3.5e-5   # their table dips here (7-8B)
    elif p < 15:   lr = 1.0e-4
    else:          lr = 8.0e-5
    lr *= float(os.environ.get("SN56_SFT_LR_MULT") or 1.0)
    return float(lr)


def choose_regime(params: float | None, n_gpus: int, gpu_free_gib: float) -> dict:
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
    # too big to full-ft: high-rank LoRA
    return {"adapter": {"r": 64, "alpha": 128, "dropout": 0.05},
            "dist": "ddp" if n_gpus > 1 else "single"}


def micro_batch_for(params: float | None, seq_len: int, gpu_free_gib: float,
                    full_ft: bool) -> int:
    """Crude activation-based micro-batch guess; the OOM ladder corrects it."""
    p = params or 7e9
    weight_overhead = (p * (16 if full_ft else 2.5)) / 2**30
    per_sample_gib = max(0.05, (p / 7e9) * (seq_len / 4096) * 0.9)
    room = max(gpu_free_gib - weight_overhead - 6, per_sample_gib)
    mb = int(room / per_sample_gib)
    return max(1, min(mb, 64))
