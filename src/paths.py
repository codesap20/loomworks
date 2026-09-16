"""Canonical container paths for the Gradients tournament runtime."""

import json
import os

# Container paths. The env overrides exist so local harnesses can run several tasks side by side in
# ONE filesystem; the validator sets none of them, so production keeps the fixed container layout.
CACHE_ROOT = os.environ.get("SN56_CACHE_ROOT", "/cache")
CHECKPOINTS_ROOT = os.environ.get("SN56_CHECKPOINTS_ROOT", "/app/checkpoints")
WORK_ROOT = os.environ.get("SN56_WORK_ROOT", "/workspace/run")
STATE_FILE = os.environ.get("SN56_STATE_FILE", "/tmp/trainer_state.json")


def model_cache_path(model_id: str) -> str:
    """Local read-only copy of the base model, pre-downloaded by the validator."""
    return os.path.join(CACHE_ROOT, "models", model_id.replace("/", "--"))


def dataset_cache_path(task_id: str) -> str:
    return os.path.join(CACHE_ROOT, "datasets", f"{task_id}_train_data.json")


def submission_dir(task_id: str, expected_repo_name: str) -> str:
    return os.path.join(CHECKPOINTS_ROOT, task_id, expected_repo_name)


def tokenized_dir(task_id: str) -> str:
    return os.path.join(WORK_ROOT, "tokenized", task_id)


def resolve_model_path(model_arg: str) -> str:
    """Prefer the validator cache copy; fall back to the raw arg (local tests)."""
    cached = model_cache_path(model_arg)
    if os.path.isdir(cached):
        return cached
    if os.path.isdir(model_arg):
        return model_arg
    return model_arg  # HF id — only resolvable when running with network (local tests)


def resolve_dataset_path(task_id: str, dataset_arg: str) -> str:
    cached = dataset_cache_path(task_id)
    if os.path.isfile(cached):
        return cached
    if os.path.isfile(dataset_arg):
        return dataset_arg
    raise FileNotFoundError(f"dataset not found at {cached} or {dataset_arg}")


def patch_adapter_base(out_dir: str, base_model_id: str) -> None:
    """Rewrite adapter_config.json's base_model_name_or_path to the task's model id.

    PEFT records whatever path the base was loaded from, which inside the trainer
    container is /cache/models/<id with / as -->. The validator loads adapter
    submissions with AutoPeftModelForCausalLM.from_pretrained(repo), which reads that
    field and resolves the base on ITS machine (validator/evaluation/common.py:
    load_finetuned_model). A trainer-container path does not exist there, so the
    submission fails to load and the task scores nothing — a silent forfeit that only
    affects LoRA runs, i.e. every model big enough to need an adapter plus, now, every
    ChatTask. Verified against a real in-image run, which wrote
    "base_model_name_or_path": "/cache/models/Qwen--Qwen3-4B".
    """
    cfg_path = os.path.join(out_dir, "adapter_config.json")
    if not base_model_id or not os.path.isfile(cfg_path):
        return
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get("base_model_name_or_path") == base_model_id:
            return
        cfg["base_model_name_or_path"] = base_model_id
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[paths] could not patch adapter base ({type(e).__name__}: {e})", flush=True)
