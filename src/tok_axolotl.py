"""Tokenize instruct/chat data with axolotl so training-time formatting is
byte-identical to the validator's evaluation tokenization.

Runs inside the dedicated axolotl venv (/opt/axo). Produces:
  {out_dir}/train   HF dataset (input_ids, attention_mask, labels)
  {out_dir}/dev     held-out split for checkpoint selection
  {out_dir}/meta.json  length stats + counts

Split design: rows are deduplicated on exact tokenized content, then assigned
to dev by a stable blake2 hash of the token ids — reproducible across retries,
no RNG, near-duplicates share prefixes but not hashes (cheap, good enough at
our dev sizes; the validator's own test split is what finally matters).
"""

import argparse
import hashlib
import json
import os
import sys

from axolotl.utils.dict import DictDefault
from datasets import Dataset
from transformers import AutoTokenizer

try:
    from axolotl.utils.data import load_tokenized_prepared_datasets
except ImportError:
    from axolotl.utils.data.sft import (
        _load_tokenized_prepared_datasets as load_tokenized_prepared_datasets,
    )


def build_dataset_entry(data_path: str, dataset_type: dict, task_type: str) -> dict:
    """Mirror of the validator's create_dataset_entry for JSON files."""
    entry = {"path": os.path.dirname(data_path),
             "ds_type": "json",
             "data_files": [os.path.abspath(data_path)]}

    if task_type == "ChatTask":
        entry.update({
            "chat_template": dataset_type.get("chat_template") or "chatml",
            "type": "chat_template",
            "field_messages": dataset_type.get("chat_column") or "conversations",
            "message_field_role": dataset_type.get("chat_role_field") or "from",
            "message_field_content": dataset_type.get("chat_content_field") or "value",
            "roles": {
                "assistant": [dataset_type.get("chat_assistant_reference") or "assistant"],
                "user": [dataset_type.get("chat_user_reference") or "user"],
            },
            "message_property_mappings": {
                "role": dataset_type.get("chat_role_field") or "from",
                "content": dataset_type.get("chat_content_field") or "value",
            },
        })
        return entry

    # InstructTextTask
    dt = {k: v for k, v in dataset_type.items() if v is not None}
    if not dt.get("field_output"):
        entry.update({"type": "completion", "field": dt.get("field_instruction")})
        return entry
    dt.setdefault("no_input_format", "{instruction}")
    if dt.get("field_input"):
        dt.setdefault("format", "{instruction} {input}")
    else:
        dt.setdefault("format", "{instruction}")
    entry.update({"format": "custom", "type": dt})
    return entry


def tokenize(model_path: str, data_path: str, dataset_type: dict, task_type: str,
             seq_len: int, out_dir: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = DictDefault({
        "datasets": [build_dataset_entry(data_path, dataset_type, task_type)],
        "sequence_len": seq_len,
        "train_on_inputs": False,
        "special_tokens": {},
        "output_dir": os.path.join(out_dir, "axo_out"),
        "dataset_prepared_path": os.path.join(out_dir, "prepared"),
        "val_set_size": 0,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "batch_size": 1,
        "base_model": model_path,
        "tokenizer_config": model_path,
        "sample_packing": False,
        "eval_sample_packing": False,
    })

    prepared = os.path.join(out_dir, "prepared")
    try:
        ds, _ = load_tokenized_prepared_datasets(tokenizer, cfg, prepared)
    except TypeError:
        cfg["dataset_prepared_path"] = prepared
        ds, _ = load_tokenized_prepared_datasets(tokenizer, cfg, split="train")

    keep_cols = [c for c in ("input_ids", "attention_mask", "labels") if c in ds.column_names]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

    seen: set[str] = set()
    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    n_empty = 0

    n_raw = len(ds)
    dev_target = max(24, min(512, n_raw // 25))

    for row in ds:
        if not any(l != -100 for l in row["labels"]):
            n_empty += 1
            continue
        digest = hashlib.blake2b(
            (",".join(map(str, row["input_ids"])) + "|" + ",".join(map(str, row["labels"]))).encode(),
            digest_size=12,
        ).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        # stable assignment: last 4 hex chars → [0, 65536)
        bucket = int(digest[-4:], 16)
        if bucket < 65536 * dev_target / max(n_raw, 1) and len(dev_rows) < dev_target:
            dev_rows.append(row)
        else:
            train_rows.append(row)

    if not train_rows:
        raise RuntimeError("tokenization produced no trainable rows")
    if not dev_rows:  # tiny datasets: steal a deterministic handful
        dev_rows = train_rows[:: max(1, len(train_rows) // 8)][:8]

    lengths = sorted(len(r["input_ids"]) for r in train_rows)

    def pct(p: float) -> int:
        return lengths[min(len(lengths) - 1, int(p * len(lengths)))]

    Dataset.from_list(train_rows).save_to_disk(os.path.join(out_dir, "train"))
    Dataset.from_list(dev_rows).save_to_disk(os.path.join(out_dir, "dev"))
    meta = {
        "n_raw": n_raw, "n_train": len(train_rows), "n_dev": len(dev_rows),
        "n_empty": n_empty, "n_dupes": n_raw - n_empty - len(train_rows) - len(dev_rows),
        "len_p50": pct(0.50), "len_p95": pct(0.95), "len_p99": pct(0.99),
        "len_max": lengths[-1], "seq_len": seq_len,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"tokenized: {meta}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--dataset-type", required=True)
    ap.add_argument("--task-type", required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenize(args.model_path, args.data_path, json.loads(args.dataset_type),
             args.task_type, args.seq_len, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
