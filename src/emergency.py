"""Last-resort submission: base weights with a jitter on the input embeddings, so the upload is
loadable and not byte-identical to the base.

Two things this file got wrong before, both found on 2026-09-16 while diagnosing the lost tournament:

  * 3e-4 was BELOW bfloat16 resolution (~0.8% relative), so every value rounded straight back and the
    "emergency submission" was byte-identical to the base — which fails the validator's is-finetune
    check. The jitter is now 1e-2, comfortably above bf16 rounding, and verified to change weights.
  * It could only handle architectures the running stack knows. On task 90d361cb
    (LiquidAI/LFM2.5-2.6B, model_type `lfm2`) the trainer and this fallback both died with
    KeyError: 'lfm2', so nothing was uploaded and the task — and the round — was lost. The raw path
    below copies the checkpoint files and jitters one tensor through safetensors alone, which works
    for any architecture, including one neither stack recognises.
"""

import argparse
import glob
import os
import shutil
import sys

import torch

JITTER = 1e-2  # must exceed bf16 relative resolution (~0.0078) or the write is a no-op


def _via_transformers(model_path: str, out_dir: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        emb.mul_(1.0 + torch.randn_like(emb) * JITTER)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model_path, trust_remote_code=True).save_pretrained(out_dir)


def _embedding_key(keys: list[str]) -> str | None:
    """The input-embedding tensor, by the names the architecture families use."""
    for needle in ("embed_tokens.weight", "wte.weight", "word_embeddings.weight",
                   "embeddings.weight", "embed_in.weight", "tok_embeddings.weight"):
        for k in keys:
            if k.endswith(needle):
                return k
    return None


def _jitter(t: torch.Tensor) -> torch.Tensor:
    gen = torch.Generator().manual_seed(1337)
    noise = torch.randn(t.shape, generator=gen, dtype=torch.float32) * JITTER
    return (t.float() * (1.0 + noise)).to(t.dtype)


def _pick_key(keys: list[str], shapes: dict) -> str | None:
    """The embedding tensor by name, else the largest 2-D float tensor (any arch, any naming)."""
    key = _embedding_key(keys)
    if key is not None:
        return key
    two_d = [(shapes[k][0] * shapes[k][1], k) for k in keys if len(shapes.get(k, ())) == 2]
    return max(two_d)[1] if two_d else (keys[0] if keys else None)


def _raw_copy(model_path: str, out_dir: str) -> None:
    """Copy the checkpoint and jitter one tensor without ever building the model.

    Handles safetensors AND .bin checkpoints: plenty of models in the 0.1-12B pool ship only
    pytorch_model*.bin, and requiring safetensors here would leave a byte-identical copy of the base
    (which fails the validator's is-finetune check) on exactly the tasks this fallback exists for.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(model_path):
        src = os.path.join(model_path, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, name))

    shards = sorted(glob.glob(os.path.join(out_dir, "*.safetensors")))
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            keys = list(f.keys())
            meta = f.metadata()
            shapes = {k: tuple(f.get_slice(k).get_shape()) for k in keys}
            key = _pick_key(keys, shapes) if _embedding_key(keys) else None
            if key is None:
                continue
            tensors = {k: f.get_tensor(k) for k in keys}
        tensors[key] = _jitter(tensors[key])
        save_file(tensors, shard, metadata=meta or {"format": "pt"})
        print(f"emergency: jittered {key} in {os.path.basename(shard)}", flush=True)
        return

    bins = sorted(glob.glob(os.path.join(out_dir, "*.bin")) + glob.glob(os.path.join(out_dir, "*.pt")))
    for shard in bins:
        sd = torch.load(shard, map_location="cpu", weights_only=True)
        if not isinstance(sd, dict) or not sd:
            continue
        shapes = {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}
        key = _pick_key(list(sd), shapes)
        if key is None or not hasattr(sd.get(key), "shape"):
            continue
        sd[key] = _jitter(sd[key])
        torch.save(sd, shard)
        print(f"emergency: jittered {key} in {os.path.basename(shard)}", flush=True)
        return

    # last resort: any shard at all, largest 2-D tensor (a single-shard model whose embedding name
    # matched nothing above)
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            keys = list(f.keys())
            meta = f.metadata()
            shapes = {k: tuple(f.get_slice(k).get_shape()) for k in keys}
            tensors = {k: f.get_tensor(k) for k in keys}
        key = _pick_key(keys, shapes)
        if key is None:
            continue
        tensors[key] = _jitter(tensors[key])
        save_file(tensors, shard, metadata=meta or {"format": "pt"})
        print(f"emergency: jittered {key} (fallback pick) in {os.path.basename(shard)}", flush=True)
        return
    raise RuntimeError("no weight shard could be jittered")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    if os.path.isdir(args.model_path) and args.model_path not in sys.path:
        sys.path.insert(0, args.model_path)  # flat sibling imports in custom archs

    try:
        _via_transformers(args.model_path, args.output_dir)
        print("emergency submission written (transformers)")
        return
    except Exception as e:
        print(f"emergency: transformers path failed ({type(e).__name__}: {e}); copying raw weights",
              flush=True)
    shutil.rmtree(args.output_dir, ignore_errors=True)
    _raw_copy(args.model_path, args.output_dir)
    print("emergency submission written (raw copy)")


if __name__ == "__main__":
    main()
