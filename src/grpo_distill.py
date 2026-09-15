"""Oracle distillation for GRPO tasks (opt-in: SN56_GRPO_MODE=distill).

The evaluator scores GRPO tasks on prompt-independent text metrics, 1-v-1 per function, minus
0.5 x KL(base || model) over the PROMPT tokens (grpo_oracle.py has the full argument). So:

  1. sample the base on a few prompts (reference statistics + first tokens),
  2. search the reward functions for one completion string (grpo_oracle.search),
  3. LoRA-distill "any prompt -> that string" with the scored KL as a leash:
       CE(target | prompt [+ a base-sampled first token])
       + lam * KL(base || model) on interior prompt positions
       + soft CE at the last prompt position against alpha * p_base + (1 - alpha) * onehot(t0).
     The last prompt position IS the first completion token, and it is inside the evaluator's KL.
     When no start-anchored format is at stake the first token is left to the base (alpha = 1) and
     the string is searched to score well behind typical base openings.
  4. score the result the evaluator's way on held-out prompts, and hand back to GRPO if it does
     not beat the base.
"""

import collections
import math
import os
import random
import time

import torch
import torch.nn.functional as F

import grpo_oracle


def _log_factory(log):
    return lambda m: log(f"[distill] {m}")


@torch.no_grad()
def _sample(model, tok, prompts, n_new=256, bs=16, gens=1, seed=0):
    torch.manual_seed(seed)
    side = tok.padding_side
    tok.padding_side = "left"
    out = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=n_new,
                           num_return_sequences=gens, use_cache=True,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    tok.padding_side = side
    return out


def _raw_values(fns, texts):
    return [grpo_oracle.Objective._raw(fn, texts) for fn in fns]


@torch.no_grad()
def _prompt_kl(model, base_fn, tok, prompts):
    tot = 0.0
    for p in prompts:
        ids = tok(p, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        lb = torch.log_softmax(base_fn(ids).float(), -1)[0]
        lf = torch.log_softmax(model(**ids).logits.float(), -1)[0]
        tot += (lb.exp() * (lb - lf)).sum(-1).mean().item()
    return tot / max(1, len(prompts))


def run(model, tok, train_prompts, dev_prompts, fns, weights, sources, end_ts, save, log, is_lora_model):
    """Train in place; call save(model, tag) with the result. Returns a summary dict or None."""
    say = _log_factory(log)
    t_start = time.time()
    rng = random.Random(1337)
    lam = float(os.environ.get("SN56_DISTILL_LAM") or 5.0)
    r = int(os.environ.get("SN56_DISTILL_R") or 16)
    lr = float(os.environ.get("SN56_DISTILL_LR") or 2e-4)
    bs = int(os.environ.get("SN56_DISTILL_BS") or 8)
    max_steps = int(os.environ.get("SN56_DISTILL_STEPS") or 400)
    device = next(model.parameters()).device

    model.eval()
    refs = _sample(model, tok, rng.sample(train_prompts, min(64, len(train_prompts))))
    firsts = collections.Counter()
    with torch.no_grad():
        for p in train_prompts[:64]:
            lg = model(**tok(p, return_tensors="pt").to(device)).logits[0, -1].float()
            for t in torch.softmax(lg, -1).topk(3).indices.tolist():
                firsts[t] += 1
    prefixes = [tok.decode([t]) for t, _ in firsts.most_common(8)]
    count = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    budget_s = float(os.environ.get("SN56_DISTILL_SEARCH_S") or 90)

    anchored, u_a, v_a, obj_a = grpo_oracle.search(fns, weights, sources, count, refs, seconds=budget_s / 2,
                                                   log=say)
    robust, u_r, v_r, obj_r = grpo_oracle.search(fns, weights, sources, count, refs, seconds=budget_s / 2,
                                                 log=say, prefixes=prefixes)
    # a start-anchored target only pays if it clearly beats the robust one scored with the same
    # prefix-free objective (the robust string is scored behind base openings, which is how it will
    # actually be sampled at alpha = 1)
    u_r_as_sampled = obj_r(robust)
    if u_a > u_r_as_sampled + float(os.environ.get("SN56_DISTILL_ANCHOR_MARGIN") or 1.0):
        text, alpha = anchored, float(os.environ.get("SN56_DISTILL_ALPHA") or 0.3)
    else:
        text, alpha = robust, 1.0
    say(f"anchored utility {u_a:.2f} vs robust {u_r_as_sampled:.2f} -> alpha={alpha}; "
        f"target {count(text)} tokens: {text[:100]!r}")
    tgt = tok(text, add_special_tokens=False).input_ids + [tok.eos_token_id]

    from peft import LoraConfig, get_peft_model
    if not is_lora_model:
        model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * max(0.1, 1 - s / max_steps))
    reserve = 900  # dev scoring + export
    model.train()
    step, t0 = 0, time.time()
    while step < max_steps and time.time() + reserve < end_ts:
        batch = rng.sample(train_prompts, bs)
        pids = [tok(p, truncation=True, max_length=512).input_ids for p in batch]
        pre = [[] for _ in batch]
        if alpha > 0:
            with torch.no_grad(), model.disable_adapter():
                for i, pid in enumerate(pids):
                    if rng.random() < alpha:
                        lg = model(input_ids=torch.tensor([pid], device=device)).logits[0, -1].float()
                        pre[i] = [int(torch.multinomial(torch.softmax(lg, -1), 1))]
        seqs = [pid + p0 + tgt for pid, p0 in zip(pids, pre)]
        T = max(len(s) for s in seqs)
        ids = torch.full((bs, T), tok.pad_token_id, dtype=torch.long, device=device)
        att = torch.zeros((bs, T), dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, device=device)
            att[i, :len(s)] = 1
        with torch.no_grad(), model.disable_adapter():
            base_logits = model(input_ids=ids, attention_mask=att).logits
        logits = model(input_ids=ids, attention_mask=att).logits
        n = len(tgt)
        ce, kl, first = [], [], []
        for i, pid in enumerate(pids):
            L, o = len(pid), len(pre[i])
            if o:
                ce.append(F.cross_entropy(logits[i, L:L + n].float(), ids[i, L + 1:L + 1 + n]))
            else:
                ce.append(F.cross_entropy(logits[i, L:L + n - 1].float(), ids[i, L + 1:L + n]))
            lpf = torch.log_softmax(logits[i, :L].float(), -1)
            lpb = torch.log_softmax(base_logits[i, :L].float(), -1)
            if L > 1:
                kl.append((lpb[:L - 1].exp() * (lpb[:L - 1] - lpf[:L - 1])).sum(-1).mean())
            q = alpha * lpb[L - 1].exp()
            q[tgt[0]] += 1 - alpha
            first.append(-(q * lpf[L - 1]).sum())
        loss = torch.stack(ce).mean() + lam * (torch.stack(kl).mean() if kl else 0.0) + torch.stack(first).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            say(f"step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
        step += 1

    # evaluator-style check on held-out prompts: per-function means + prompt KL, vs the base
    model.eval()
    dev = dev_prompts[:64] if dev_prompts else train_prompts[-64:]
    base_fn = lambda enc: _base_logits(model, enc)
    with model.disable_adapter():
        base_txt = _sample(model, tok, dev, gens=2, seed=7)
    ours_txt = _sample(model, tok, dev, gens=2, seed=7)
    bv = [sum(v) / len(v) for v in _raw_values(fns, base_txt)]
    ov = [sum(v) / len(v) for v in _raw_values(fns, ours_txt)]
    kl = _prompt_kl(model, base_fn, tok, dev)
    wins = sum(w for w, b, o in zip(weights, bv, ov) if o > b)
    say(f"{step} steps; dev base {[round(x, 3) for x in bv]} ours {[round(x, 3) for x in ov]} "
        f"prompt_kl {kl:.4f}; weight won vs base {wins:.2f}/{sum(weights):.2f} "
        f"({time.time() - t_start:.0f}s total)")
    if wins < 0.5 * sum(weights):
        say("does not beat the base on most of the weight; not shipping the distilled policy")
        return None
    save(model, f"distill steps={step} kl={kl:.4f}")
    return {"steps": step, "kl": kl, "base": bv, "ours": ov, "alpha": alpha}


@torch.no_grad()
def _base_logits(model, enc):
    with model.disable_adapter():
        return model(**enc).logits
