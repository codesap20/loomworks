"""Budgeted multi-candidate learning-rate search, scored on held-out dev loss.

Why this exists: the incumbent tournament champion does not ship a fixed LR. It
runs a per-task search (up to 20% of the budget, coarse sweep then refine, with
divergence pruning) and only falls back to its size-bucket table when DeepSpeed
is active. Against a fixed table we are giving away whatever the per-task optimum
differs from the bucket — which our own sweeps showed is a lot (the best chat
multiplier and the best DPO multiplier point in OPPOSITE directions).

Two deliberate differences from their design:

  * We score each candidate on the HELD-OUT dev split with the same masked
    per-sample cross-entropy the validator compares, not on training loss. The
    validator ranks generalization; training loss at a hot LR keeps falling on
    the batches just fitted. (They score SFT probes on training loss because
    their trainer runs with eval_strategy=no.)
  * No warm-start carry-forward. Every candidate starts from the same initial
    weights, so the comparison is clean and the real run is unaffected by search
    order. The cost is that search steps are thrown away, which is why the budget
    fraction here is small and the search skips itself when it cannot fit.

Never runs under ZeRO-3/multi-GPU: restoring weights per candidate under sharding
needs collectives on every rank and the payoff there is smaller (large models are
the ones the champion also leaves untuned).
"""

import math
import time

import torch

_PRUNE_WINDOW = 5           # rolling-mean window over recent micro-batch losses
_PRUNE_MIN_STEP = 4         # never prune before this many optimizer steps
_PRUNE_DIVERGE_FACTOR = 1.75  # prune when the rolling mean exceeds this x its own best
_EDGE_TOLERANCE = 0.025     # prefer the highest LR whose dev loss is within 2.5% of best
_WARMUP_FRAC = 0.15         # per-candidate LR ramp, as a fraction of its steps


def candidate_lrs(center: float, n: int, half_range_decades: float) -> list[float]:
    """n log-spaced LRs centred on `center`, spanning +/- half_range decades."""
    if n <= 1:
        return [center]
    lo = math.log10(center) - half_range_decades
    hi = math.log10(center) + half_range_decades
    return [10 ** (lo + i * (hi - lo) / (n - 1)) for i in range(n)]


def edge_extension(results: dict, step_decades: float, max_extensions: int) -> float | None:
    """Next LR to try when the best result sits at an END of the tested range.

    A fixed window only finds the optimum if the seed LR is already within it. When
    the best score is at the hottest (or coldest) candidate the true optimum is
    usually outside, so step further in that direction while it keeps improving.
    Returns None when the best is interior, or when the extension budget is spent.
    """
    finite = {k: v for k, v in results.items() if math.isfinite(v)}
    if len(finite) < 2:
        return None
    ordered = sorted(finite)
    best = min(finite, key=finite.get)
    span = math.log10(max(ordered)) - math.log10(min(ordered))
    # every extension widens the span by step_decades; stop after max_extensions
    if span >= (len(ordered) - 1) * step_decades + max_extensions * step_decades:
        return None
    if best == ordered[-1]:
        return 10 ** (math.log10(best) + step_decades)
    if best == ordered[0]:
        return 10 ** (math.log10(best) - step_decades)
    return None


@torch.no_grad()
def _snapshot(model) -> dict:
    return {n: p.detach().to("cpu", copy=True)
            for n, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def _restore(model, snap: dict) -> None:
    for n, p in model.named_parameters():
        if p.requires_grad and n in snap:
            p.copy_(snap[n].to(p.device, p.dtype))


@torch.no_grad()
def _dev_loss(model, dev_batches: list) -> float:
    """Mean over samples of per-sample mean CE on completion tokens (validator's
    quantity), over a fixed dev subset."""
    model.eval()
    dev = next(model.parameters()).device
    tot, n = 0.0, 0
    for b in dev_batches:
        ids = b["input_ids"].to(dev)
        lab = b["labels"].to(dev)
        am = b.get("attention_mask")
        am = am.to(dev) if am is not None else None
        logits = model(input_ids=ids, attention_mask=am).logits
        sl = logits[:, :-1, :].float()
        slab = lab[:, 1:]
        ce = torch.nn.functional.cross_entropy(
            sl.reshape(-1, sl.size(-1)), slab.reshape(-1),
            ignore_index=-100, reduction="none").view(slab.shape)
        cnt = (slab != -100).sum(1).clamp(min=1)
        per_sample = ce.sum(1) / cnt
        tot += float(per_sample.sum())
        n += per_sample.numel()
    model.train()
    return tot / max(1, n)


def _train_candidate(model, batches, lr, steps, accum, opt_factory, log) -> bool:
    """Train `steps` optimizer steps at `lr` from the current weights.

    Returns False if the candidate diverged and was pruned."""
    opt = opt_factory(lr)
    warmup = max(2, int(steps * _WARMUP_FRAC))
    recent, best_roll = [], None
    model.train()
    bi = 0
    for step in range(steps):
        scale = min(1.0, (step + 1) / warmup)
        for g in opt.param_groups:
            g["lr"] = lr * scale
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum):
            b = batches[bi % len(batches)]
            bi += 1
            dev = next(model.parameters()).device
            inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in b.items()}
            out = model(**inputs)
            loss = out.loss / accum
            loss.backward()
            step_loss += float(loss) * accum / accum
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if not math.isfinite(step_loss):
            log(f"lr-search: lr={lr:.2e} diverged (non-finite) at step {step + 1}")
            del opt
            return False
        recent.append(step_loss)
        if len(recent) > _PRUNE_WINDOW:
            recent.pop(0)
        roll = sum(recent) / len(recent)
        best_roll = roll if best_roll is None else min(best_roll, roll)
        if step + 1 >= _PRUNE_MIN_STEP and roll > best_roll * _PRUNE_DIVERGE_FACTOR:
            log(f"lr-search: lr={lr:.2e} pruned at step {step + 1} "
                f"(rolling {roll:.4f} > {_PRUNE_DIVERGE_FACTOR}x best {best_roll:.4f})")
            del opt
            return False
    del opt
    return True


def search(model, train_batches: list, dev_batches: list, center_lr: float, *,
           steps: int, accum: int, opt_factory, deadline: float,
           n_candidates: int = 4, half_range_decades: float = 0.3,
           max_extensions: int = 3,
           edge_tolerance: float = _EDGE_TOLERANCE, log=print) -> tuple[float, dict]:
    """Return (chosen_lr, info). Restores the model's initial weights before
    returning, so the caller's real run starts from an untouched model."""
    lrs = candidate_lrs(center_lr, n_candidates, half_range_decades)
    log(f"lr-search: {n_candidates} candidates {[f'{x:.2e}' for x in lrs]} "
        f"x {steps} steps (accum {accum}), dev batches {len(dev_batches)}")
    snap = _snapshot(model)
    results: dict[float, float] = {}
    try:
        for lr in lrs:
            if time.time() > deadline:
                log("lr-search: out of budget; keeping the candidates measured so far")
                break
            _restore(model, snap)
            ok = _train_candidate(model, train_batches, lr, steps, accum, opt_factory, log)
            if not ok:
                results[lr] = float("inf")
                continue
            dl = _dev_loss(model, dev_batches)
            results[lr] = dl
            log(f"lr-search: lr={lr:.2e} dev={dl:.5f}")
        # the optimum is often outside a fixed window; ride the winning edge outward
        for _ in range(max_extensions):
            nxt = edge_extension(results, half_range_decades * 2 / max(1, n_candidates - 1),
                                 max_extensions)
            if nxt is None or time.time() > deadline:
                break
            _restore(model, snap)
            if not _train_candidate(model, train_batches, nxt, steps, accum, opt_factory, log):
                results[nxt] = float("inf")
                break
            results[nxt] = _dev_loss(model, dev_batches)
            log(f"lr-search: edge extension lr={nxt:.2e} dev={results[nxt]:.5f}")
    finally:
        _restore(model, snap)
        del snap
        torch.cuda.empty_cache()

    finite = {k: v for k, v in results.items() if math.isfinite(v)}
    if not finite:
        log(f"lr-search: every candidate diverged; keeping heuristic {center_lr:.2e}")
        return center_lr, {"results": results, "picked": "heuristic"}
    best_lr = min(finite, key=finite.get)
    best = finite[best_lr]
    # A short probe understates how much a hotter LR pays off over a full run, so
    # among candidates statistically tied on dev, take the hottest.
    edge = max((k for k, v in finite.items() if v <= best * (1 + edge_tolerance)),
               default=best_lr)
    log(f"lr-search: best dev {best:.5f} at {best_lr:.2e}; "
        f"edge pick {edge:.2e} (within {edge_tolerance:.1%})")
    return edge, {"results": results, "best_lr": best_lr, "picked": "edge"}
