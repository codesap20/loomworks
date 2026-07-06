"""Chat dataset preparation for ChatTask SFT (modern venv, transformers 5.x).

Mirrors the validator's axolotl `chat_template` tokenization as closely as
possible: the conversation is rendered with the checkpoint's own chat template
and loss labels cover ONLY assistant message content plus the trailing
end-of-turn token(s). Role headers, user/system turns and padding are masked
with -100 (axolotl train_on_inputs=false behaviour).
"""

import hashlib
import json

import numpy as np
import torch


# Defaults mirror core.models.dataset_models.ChatTemplateDatasetType in G.O.D.
DATASET_TYPE_DEFAULTS = {
    "chat_template": "tokenizer_default",
    "chat_column": "conversations",
    "chat_role_field": "from",
    "chat_content_field": "value",
    "chat_user_reference": "user",
    "chat_assistant_reference": "assistant",
}

# Axolotl's chatml template (fallback when the checkpoint has no template).
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def parse_dataset_type(dataset_type_json):
    """Parse the --dataset-type JSON into a dict with defaults filled in."""
    fields = dict(DATASET_TYPE_DEFAULTS)
    if dataset_type_json:
        try:
            raw = json.loads(dataset_type_json) if isinstance(dataset_type_json, str) else dict(dataset_type_json)
        except (ValueError, TypeError):
            raw = {}
        for key in fields:
            if raw.get(key):
                fields[key] = raw[key]
    return fields


def resolve_chat_template(tokenizer, dataset_type, model_path):
    """Pick the template string used for rendering (and by the validator at eval)."""
    name = (dataset_type.get("chat_template") or "tokenizer_default").strip()

    def _tokenizer_template():
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.chat_template
        try:  # some checkpoints ship it only as a file
            import os
            path = os.path.join(str(model_path), "chat_template.jinja")
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
        except OSError:
            pass
        return None

    if name in ("tokenizer_default", "default", ""):
        return _tokenizer_template() or CHATML_TEMPLATE
    if name == "chatml":
        return CHATML_TEMPLATE
    # Unknown named template: the seed's own template is the safest match for eval.
    return _tokenizer_template() or CHATML_TEMPLATE


def _normalise_role(raw_role, dataset_type):
    role = str(raw_role)
    if role == dataset_type["chat_user_reference"]:
        return "user"
    if role == dataset_type["chat_assistant_reference"]:
        return "assistant"
    lowered = role.lower()
    if lowered in ("system",):
        return "system"
    if lowered in ("user", "human"):
        return "user"
    if lowered in ("assistant", "gpt", "ai", "bot"):
        return "assistant"
    return lowered  # rendered as-is, never trained on


def load_conversations(data_path, dataset_type):
    """Load the task JSON and normalise rows into [{'role','content'}, ...] lists."""
    with open(data_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if isinstance(rows, dict):  # tolerate {"train": [...]} style wrappers
        for key in ("train", "data", "rows", dataset_type["chat_column"]):
            if isinstance(rows.get(key), list):
                rows = rows[key]
                break
    conversations = []
    role_f = dataset_type["chat_role_field"]
    content_f = dataset_type["chat_content_field"]
    for row in rows:
        msgs = row.get(dataset_type["chat_column"]) if isinstance(row, dict) else row
        if not isinstance(msgs, list):
            continue
        conv = []
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            role = msg.get(role_f, msg.get("role"))
            content = msg.get(content_f, msg.get("content"))
            if role is None or content is None:
                continue
            conv.append({"role": _normalise_role(role, dataset_type), "content": str(content)})
        if any(m["role"] == "assistant" for m in conv):
            conversations.append(conv)
    return conversations


def _render(tokenizer, messages, add_generation_prompt):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def _assistant_char_spans(tokenizer, messages):
    """Char ranges of assistant content + its end-of-turn token(s) in the full render."""
    full_text = _render(tokenizer, messages, False)
    spans = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        upto = _render(tokenizer, messages[: i + 1], False)
        if not full_text.startswith(upto):
            continue  # template rewrites earlier turns; skip this span rather than mislabel
        with_header = _render(tokenizer, messages[:i], True)
        if not upto.startswith(with_header):
            # Generation prompt isn't a clean prefix (exotic template) -> fall back
            # to masking only previous turns; header tokens get trained in this case.
            with_header = _render(tokenizer, messages[:i], False)
            if not upto.startswith(with_header):
                continue
        spans.append((len(with_header), len(upto)))
    return full_text, spans


def tokenize_conversation(tokenizer, messages, seq_len):
    """Return {'input_ids','labels'} numpy arrays or None if nothing trainable."""
    full_text, spans = _assistant_char_spans(tokenizer, messages)
    if not spans:
        return None

    try:
        enc = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=seq_len,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]
    except (TypeError, ValueError, NotImplementedError):
        enc, offsets = None, None

    if offsets is not None:
        input_ids = enc["input_ids"]
        labels = [-100] * len(input_ids)
        span_idx = 0
        for tok_idx, (start, _end) in enumerate(offsets):
            while span_idx < len(spans) and start >= spans[span_idx][1]:
                span_idx += 1
            if span_idx >= len(spans):
                break
            if spans[span_idx][0] <= start < spans[span_idx][1]:
                labels[tok_idx] = input_ids[tok_idx]
    else:
        # Slow-tokenizer fallback: locate spans by prefix token counts (may drift
        # by one token at merge boundaries; acceptable for training).
        input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:seq_len]
        labels = [-100] * len(input_ids)
        for start, end in spans:
            n_start = len(tokenizer(full_text[:start], add_special_tokens=False)["input_ids"])
            n_end = len(tokenizer(full_text[:end], add_special_tokens=False)["input_ids"])
            for tok_idx in range(n_start, min(n_end, len(labels))):
                labels[tok_idx] = input_ids[tok_idx]

    if not any(l != -100 for l in labels):
        return None
    return {
        "input_ids": np.asarray(input_ids, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.int64),
    }


def _stable_hash(conversation):
    payload = json.dumps(conversation, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return int(hashlib.md5(payload).hexdigest()[:12], 16)


def prepare_dataset(tokenizer, data_path, dataset_type_json, model_path, seq_len,
                    dev_frac=0.02, dev_cap=200, log=print):
    """Load + tokenize everything. Returns (train_items, dev_items, stats)."""
    dataset_type = parse_dataset_type(dataset_type_json)
    template = resolve_chat_template(tokenizer, dataset_type, model_path)
    tokenizer.chat_template = template

    conversations = load_conversations(data_path, dataset_type)
    log(f"[chat_data] {len(conversations)} usable conversations from {data_path}")

    train_items, dev_items, skipped = [], [], 0
    dev_mod = max(2, int(round(1.0 / max(dev_frac, 1e-6))))
    for idx, conv in enumerate(conversations):
        try:
            item = tokenize_conversation(tokenizer, conv, seq_len)
        except Exception:
            item = None
        if item is None:
            skipped += 1
            continue
        if _stable_hash(conv) % dev_mod == 0 and len(dev_items) < dev_cap:
            dev_items.append(item)
        else:
            train_items.append(item)
        if (idx + 1) % 5000 == 0:
            log(f"[chat_data] tokenized {idx + 1}/{len(conversations)}")

    stats = {
        "conversations": len(conversations),
        "train": len(train_items),
        "dev": len(dev_items),
        "skipped_untrainable": skipped,
        "train_tokens": int(sum(len(it["input_ids"]) for it in train_items)),
    }
    log(f"[chat_data] prepared: {stats}")
    return train_items, dev_items, stats


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_collate(pad_token_id):
    """Pad a list of {'input_ids','labels'} to the batch max length."""

    def collate(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        n = len(batch)
        input_ids = torch.full((n, max_len), pad_token_id, dtype=torch.long)
        labels = torch.full((n, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len), dtype=torch.long)
        for i, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[i, :length] = torch.from_numpy(item["input_ids"])
            labels[i, :length] = torch.from_numpy(item["labels"])
            attention_mask[i, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate
