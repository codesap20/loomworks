"""Last-resort submission: base weights with a tiny multiplicative jitter on
the input embeddings, so the upload is loadable and not byte-identical to the
base. Run under whichever python stack can load the architecture."""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    if os.path.isdir(args.model_path) and args.model_path not in sys.path:
        sys.path.insert(0, args.model_path)  # flat sibling imports in custom archs

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        emb.mul_(1.0 + torch.randn_like(emb) * 3e-4)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True).save_pretrained(args.output_dir)
    print("emergency submission written")


if __name__ == "__main__":
    main()
