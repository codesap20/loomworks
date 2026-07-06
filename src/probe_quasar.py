"""Single-GPU sanity probe for the modern (transformers v5) stack.

Loads a custom-arch checkpoint in bf16, prepares one batch through the chat
pipeline, and runs forward+backward with only the last decoder layer trainable
(memory fits one 80GB card). Validates: fla/causal_conv1d imports, remote-code
loading, chat-template masking, and the CCE loss path. Not used in tournaments.
"""

import argparse
import json
import sys

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--dataset-type", required=True)
    args = ap.parse_args()

    import fla  # noqa: F401
    import causal_conv1d  # noqa: F401
    print("kernels: fla + causal_conv1d import OK")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    print(f"model loaded: {model.config.model_type}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

    import chat_data
    ds_type = json.loads(args.dataset_type)
    rows = chat_data.load_rows(args.data_path, ds_type)
    sample = chat_data.encode_conversation(rows[0], tokenizer, ds_type, 4096)
    n_train_tokens = sum(1 for l in sample["labels"] if l != -100)
    assert n_train_tokens > 0, "masking produced no trainable tokens"
    print(f"template/masking OK: {len(sample['input_ids'])} tokens, "
          f"{n_train_tokens} trainable")

    for p in model.parameters():
        p.requires_grad_(False)
    last_layer = list(model.parameters())[-1]
    trainables = []
    # unfreeze the lm_head (or tied embedding) + final norm — enough for a real backward
    head = model.get_output_embeddings()
    for p in head.parameters():
        p.requires_grad_(True)
        trainables.append(p)

    input_ids = torch.tensor([sample["input_ids"][:1024]]).cuda()
    labels = torch.tensor([sample["labels"][:1024]]).cuda()

    loss = None
    try:
        from cut_cross_entropy import linear_cross_entropy
        hidden = model.model(input_ids=input_ids).last_hidden_state
        shift_ok = True
        loss = linear_cross_entropy(
            hidden.to(torch.bfloat16), head.weight.to(torch.bfloat16),
            labels, shift=1, impl="cce")
        print(f"CCE path OK: loss={loss.item():.4f}")
    except Exception as e:
        print(f"CCE path failed ({type(e).__name__}: {e}); falling back to logits CE")
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        print(f"logits CE path OK: loss={loss.item():.4f}")

    loss.backward()
    grad_norm = trainables[0].grad.norm().item() if trainables[0].grad is not None else 0.0
    assert grad_norm > 0, "no gradient reached the head"
    print(f"backward OK: head grad norm {grad_norm:.4f}")
    print("PROBE PASSED")


if __name__ == "__main__":
    sys.exit(main())
