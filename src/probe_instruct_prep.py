"""Container-side check: instruct-format prep against the quasar tokenizer."""

import sys

sys.path.insert(0, "/patch")
sys.path.insert(0, "/cache/models/gradients-io-tournaments--continuous-sft-seed-quasar-king")

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "/cache/models/gradients-io-tournaments--continuous-sft-seed-quasar-king",
    trust_remote_code=True)

import chat_data

train, dev, stats = chat_data.prepare_instruct_dataset(
    tok, "/cache/datasets/smoke-instruct-001_train_data.json",
    '{"field_instruction":"instruct","field_output":"output"}', 4096)
s = train[0]
n = sum(1 for l in s["labels"] if l != -100)
print("sample tokens:", len(s["input_ids"]), "trainable:", n)
assert s["labels"][-1] == tok.eos_token_id, "missing trailing eos"
assert s["labels"][0] == -100, "prompt not masked"
print("INSTRUCT PREP OK, eos id:", tok.eos_token_id)
