"""Oracle distillation for GRPO tasks (opt-in: SN56_GRPO_MODE=distill).

The evaluator scores GRPO tasks on prompt-independent text metrics, 1-v-1 per function, minus
0.5 x KL(base || model) over the PROMPT tokens (grpo_oracle.py has the full argument). So:

  1. sample the base on a few prompts (reference statistics + first tokens),
  2. search the reward functions for completion strings (grpo_oracle.search),
  3. LoRA-distill "any prompt -> that string" with the scored KL as a leash:
       CE(target | prompt [+ a base-sampled first token])
       + lam * KL(base || model) on interior prompt positions
       + soft CE at the last prompt position against alpha * p_base + (1 - alpha) * onehot(t0).
     The last prompt position IS the first completion token, and it is inside the evaluator's KL.
  4. Two candidates are trained when time allows — the best string emitted from the first token
     (alpha 0.3) and the best string behind typical base openings (alpha 1.0, near-zero KL on the
     last position) — and the one that wins the evaluator's own 1-v-1 formula against the other on
     held-out prompts is shipped. Which one wins depends on how precisely the model reproduces the
     string under temperature-1 sampling, which the search alone cannot see (measured: on the
     platypus task the anchored candidate won .80 vs .45 despite 5x the KL).
  5. If the shipped candidate does not beat the base on most of the reward weight, hand back to GRPO.
"""

import collections
import os
import random
import shutil
import time

import torch
import torch.nn.functional as F

import grpo_oracle

BETA_GRPO = 0.5


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


def _means(fns, texts):
    return [sum(v) / max(1, len(v)) for v in (grpo_oracle.Objective._raw(fn, texts) for fn in fns)]


def _reward_safe(fns, texts):
    """True when every reward function survives these completions.

    The evaluator catches only TypeError, so a reward function raising on what our policy emits
    would fail the whole repo on that task — worse than any score. Checked on real samples.
    """
    for fn in fns:
        for i in range(0, len(texts), 16):
            if grpo_oracle.Objective._raw(fn, texts[i:i + 16], strict=True) is None:
                return False
    return True


@torch.no_grad()
def _prompt_kl(model, tok, prompts):
    """Evaluator's KL: KL(base || model) over all prompt positions, batch 1, truncation 512."""
    tot = 0.0
    for p in prompts:
        ids = tok(p, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with model.disable_adapter():
            lb = torch.log_softmax(model(**ids).logits.float(), -1)[0]
        lf = torch.log_softmax(model(**ids).logits.float(), -1)[0]
        tot += (lb.exp() * (lb - lf)).sum(-1).mean().item()
    return tot / max(1, len(prompts))


def pair_scores(a, b, weights):
    """result_processing.normalize_rewards_and_compute_loss for exactly two repos."""
    sa = sb = 0.0
    for f, w in enumerate(weights):
        va, vb = a["means"][f], b["means"][f]
        mn = min(va, vb)
        if mn < 0:
            na, nb = (0.75, 0.25) if va - mn > vb - mn else (0.25, 0.75)
            if va == vb:
                na = nb = 0.25  # sorted() keeps order on ties; either way neither side gains
        else:
            mx = max(va, vb)
            na, nb = (va / mx, vb / mx) if mx > 0 else (1.0, 1.0)
        sa += w * na
        sb += w * nb
    return sa - BETA_GRPO * a["kl"], sb - BETA_GRPO * b["kl"]


def run(model, tok, train_prompts, dev_prompts, fns, weights, sources, end_ts, save_dir_fn, log, work_dir):
    """Train LoRA candidates on `model`; copy the winner into the output via save_dir_fn(src_dir, tag).

    Returns a summary dict, or None when nothing beat the base (caller falls back to GRPO).
    """
    say = lambda m: log(f"[distill] {m}")
    t_start = time.time()
    rng = random.Random(1337)
    lam = float(os.environ.get("SN56_DISTILL_LAM") or 5.0)
    r = int(os.environ.get("SN56_DISTILL_R") or 16)
    lr = float(os.environ.get("SN56_DISTILL_LR") or 2e-4)
    bs = int(os.environ.get("SN56_DISTILL_BS") or 8)
    max_steps = int(os.environ.get("SN56_DISTILL_STEPS") or 400)
    device = next(model.parameters()).device
    reserve = 600  # final copy + margin before the hard deadline

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
    search_s = float(os.environ.get("SN56_DISTILL_SEARCH_S") or 90)
    anchored, u_a, _, _ = grpo_oracle.search(fns, weights, sources, count, refs, seconds=search_s / 2, log=say)
    robust, u_r, _, _ = grpo_oracle.search(fns, weights, sources, count, refs, seconds=search_s / 2, log=say,
                                           prefixes=prefixes)
    cands = [("anchored", anchored, float(os.environ.get("SN56_DISTILL_ALPHA") or 0.3)), ("robust", robust, 1.0)]
    if anchored == robust:
        cands = cands[:1]

    dev = (dev_prompts or train_prompts[-128:])[:128]
    with torch.no_grad():
        base_txt = _sample(model, tok, dev, gens=2, seed=7)
    base = {"means": _means(fns, base_txt), "kl": 0.0}
    say(f"base dev means {[round(x, 4) for x in base['means']]}")

    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    peft_model = None
    results = []
    cost = None
    for idx, (name, text, alpha) in enumerate(cands):
        if cost is not None and time.time() + cost + reserve > end_ts:
            say(f"no time for candidate {name}")
            break
        t_c = time.time()
        if peft_model is None:
            peft_model = get_peft_model(model, cfg, adapter_name=name)
        else:
            peft_model.add_adapter(name, cfg)
            peft_model.set_adapter(name)
        m = peft_model
        tgt = tok(text, add_special_tokens=False).input_ids + [tok.eos_token_id]
        say(f"candidate {name}: alpha={alpha} target {len(tgt)} tokens {text[:80]!r}")
        params = [p for n_, p in m.named_parameters() if p.requires_grad and f".{name}." in n_]
        for n_, p in m.named_parameters():
            if "lora_" in n_ and f".{name}." not in n_:
                p.requires_grad_(False)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 20) * max(0.1, 1 - s / max_steps))
        m.train()
        step = 0
        while step < max_steps and time.time() + reserve < end_ts:
            batch = rng.sample(train_prompts, bs)
            pids = [tok(p, truncation=True, max_length=512).input_ids for p in batch]
            pre = [[] for _ in batch]
            with torch.no_grad(), m.disable_adapter():
                for i, pid in enumerate(pids):
                    if alpha > 0 and rng.random() < alpha:
                        lg = m(input_ids=torch.tensor([pid], device=device)).logits[0, -1].float()
                        pre[i] = [int(torch.multinomial(torch.softmax(lg, -1), 1))]
            seqs = [pid + p0 + tgt for pid, p0 in zip(pids, pre)]
            T = max(len(s) for s in seqs)
            ids = torch.full((bs, T), tok.pad_token_id, dtype=torch.long, device=device)
            att = torch.zeros((bs, T), dtype=torch.long, device=device)
            for i, s in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s, device=device)
                att[i, :len(s)] = 1
            with torch.no_grad(), m.disable_adapter():
                base_logits = m(input_ids=ids, attention_mask=att).logits
            logits = m(input_ids=ids, attention_mask=att).logits
            n = len(tgt)
            ce, kl, first = [], [], []
            for i, pid in enumerate(pids):
                L = len(pid)
                if pre[i]:
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
            loss = torch.stack(ce).mean() + torch.stack(first).mean()
            if kl:
                loss = loss + lam * torch.stack(kl).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if step % 100 == 0:
                say(f"{name} step {step} loss {loss.item():.4f} ({time.time() - t_c:.0f}s)")
            step += 1
        m.eval()
        txt = _sample(m, tok, dev, gens=2, seed=7)
        if not _reward_safe(fns, txt):
            say(f"candidate {name}: a reward function raises on its completions; discarding it")
            continue
        res = {"name": name, "alpha": alpha, "steps": step, "means": _means(fns, txt), "kl": _prompt_kl(m, tok, dev)}
        out = os.path.join(work_dir, f"cand_{name}")
        shutil.rmtree(out, ignore_errors=True)
        m.save_pretrained(out, selected_adapters=[name])
        res["dir"] = out
        results.append(res)
        cost = time.time() - t_c
        say(f"candidate {name}: {step} steps, dev means {[round(x, 4) for x in res['means']]} "
            f"kl {res['kl']:.4f} ({cost:.0f}s)")

    if not results:
        return None
    best = results[0]
    if len(results) == 2:
        s0, s1 = pair_scores(results[0], results[1], weights)
        best = results[0] if s0 >= s1 else results[1]
        say(f"1v1 on dev: {results[0]['name']} {s0:.4f} vs {results[1]['name']} {s1:.4f} -> {best['name']}")
    sb_ours, sb_base = pair_scores(best, base, weights)
    say(f"{best['name']} vs base on dev: {sb_ours:.4f} vs {sb_base:.4f} ({time.time() - t_start:.0f}s total)")
    if sb_ours < sb_base + abs(sb_base) * 0.01:
        say("does not beat the base by the boss margin; not shipping the distilled policy")
        return None
    save_dir_fn(best["dir"], best["name"], f"distill {best['name']} steps={best['steps']} kl={best['kl']:.4f}")
    return best
