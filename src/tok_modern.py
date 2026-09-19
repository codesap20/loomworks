"""Tokenize an InstructTextTask for train_sft.py on the transformers-5 stack.

tok_axolotl.py needs axolotl, which only exists in the legacy venv, and that venv cannot load
architectures such as lfm2. This builds the same on-disk layout train_sft.py reads
({out}/train, {out}/dev, {out}/meta.json) from chat_data.prepare_instruct_dataset, which already
mirrors axolotl's instruct formatting (prompt and response tokenized separately, trailing eos,
prompt masked) and runs under /opt/modern.

Measured 2026-09-19 on the rebuilt live round-1 task 90d361cb (LFM2.5-2.6B, 2 h, 1xH100):
train_sft.py on this layout 0.3268 (would have placed 3rd/15) vs train_chat_modern.py 0.3557
(12th/15); live winner 0.3256.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset                    # noqa: E402
from transformers import AutoTokenizer          # noqa: E402

import chat_data                                # noqa: E402


def _pct(xs, p):
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--dataset-type", required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_path)
    # dev share ~1.7%, the split the benchmark above was measured with (chat_data's own default
    # caps dev at 200 rows, too few for the soup's greedy acceptance on a 60k-row task)
    train, dev, stats = chat_data.prepare_instruct_dataset(
        tok, a.data_path, a.dataset_type, a.seq_len, dev_frac=1 / 58, dev_cap=1000,
        log=lambda m: print(m, flush=True))
    if not train:
        print("[tok_modern] no trainable rows", flush=True)
        sys.exit(1)

    def to_rows(items):
        return [{k: [int(x) for x in v] for k, v in it.items()} for it in items]

    tmp = a.out_dir.rstrip("/") + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    Dataset.from_list(to_rows(train)).save_to_disk(os.path.join(tmp, "train"))
    Dataset.from_list(to_rows(dev or train[:64])).save_to_disk(os.path.join(tmp, "dev"))
    lens = sorted(len(it["input_ids"]) for it in train)
    meta = {"n_raw": stats["rows"], "n_train": len(train), "n_dev": len(dev),
            "n_empty": stats["skipped"], "n_dupes": 0,
            "len_p50": _pct(lens, .5), "len_p95": _pct(lens, .95), "len_p99": _pct(lens, .99),
            "len_max": lens[-1], "seq_len": a.seq_len}
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if os.path.isdir(a.out_dir):
        import shutil
        shutil.rmtree(a.out_dir)
    os.replace(tmp, a.out_dir)
    print(f"[tok_modern] {meta}", flush=True)


if __name__ == "__main__":
    main()
