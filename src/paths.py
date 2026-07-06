"""Canonical container paths for the Gradients tournament runtime."""

import os

CACHE_ROOT = "/cache"
CHECKPOINTS_ROOT = "/app/checkpoints"
WORK_ROOT = "/workspace/run"
STATE_FILE = "/tmp/trainer_state.json"


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
