"""Single-ramp learning-rate range probe (Smith-style, edge-of-stability pick).

One exponential LR sweep over cached batches, then weights are restored and the
chosen LR is handed to the real run. Distinct by construction from staged
multi-trial searches: a single continuous ramp, one restore, loss-knee plus
divergence-point geometry for the pick.

Only used when: single GPU, full probe cost < ~8% of the remaining budget.
"""

import math
import time

import torch


def _smooth(values: list[float], beta: float = 0.75) -> list[float]:
    out, acc = [], None
    for v in values:
        acc = v if acc is None else beta * acc + (1 - beta) * v
        out.append(acc)
    return out


def lr_range_probe(model, batches: list[dict], center_lr: float,
                   steps: int = 48, span: float = 30.0, log=print) -> float:
    """Sweep LR from center/span to center*span over `steps` steps.

    Returns the chosen peak LR (falls back to center_lr on any anomaly).
    `batches` are pre-collated dicts of cuda-ready tensors, reused cyclically.
    """
    t0 = time.time()
    device = next(model.parameters()).device

    snapshot = {n: p.detach().to("cpu", copy=True)
                for n, p in model.named_parameters() if p.requires_grad}

    lr_lo = center_lr / span
    ratio = span ** (2.0 / max(1, steps - 1))
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr_lo, betas=(0.9, 0.95), weight_decay=0.0)

    lrs, losses = [], []
    was_training = model.training
    model.train()
    try:
        for i in range(steps):
            lr = lr_lo * (ratio ** i)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batches[i % len(batches)].items()}
            out = model(**batch)
            loss = out.loss
            if not torch.isfinite(loss):
                lrs.append(lr); losses.append(float("inf"))
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0)
            opt.step()
            lrs.append(lr)
            losses.append(loss.item())
            # stop early once clearly diverged: 2x the best smoothed loss so far
            sm = _smooth(losses)
            if len(sm) > 8 and sm[-1] > 2.0 * min(sm):
                break
    except torch.cuda.OutOfMemoryError:
        log("[lr_probe] OOM during sweep; falling back to heuristic LR")
        losses = []
    finally:
        model.train(was_training)
        del opt
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in snapshot:
                    p.copy_(snapshot[n].to(p.device, p.dtype))
        snapshot.clear()
        torch.cuda.empty_cache()

    if len(losses) < 12:
        log(f"[lr_probe] too few points ({len(losses)}); using center {center_lr:.2e}")
        return center_lr

    sm = _smooth(losses)
    best_i = min(range(len(sm)), key=lambda i: sm[i])
    # steepest smoothed descent (knee) over a 3-step baseline
    knee_i, best_drop = best_i, 0.0
    for i in range(3, min(len(sm), best_i + 1)):
        drop = sm[i - 3] - sm[i]
        if drop > best_drop:
            best_drop, knee_i = drop, i
    # first divergence after the minimum: smoothed loss 25% above the min
    div_lr = lrs[-1] * ratio
    for i in range(best_i, len(sm)):
        if sm[i] > sm[best_i] * 1.25:
            div_lr = lrs[i]
            break

    knee_lr = lrs[knee_i]
    pick = math.sqrt(knee_lr * (div_lr / 3.0))
    pick = min(max(pick, center_lr / 10), center_lr * 10)
    log(f"[lr_probe] {len(losses)} pts in {time.time() - t0:.0f}s: "
        f"knee={knee_lr:.2e} min@{lrs[best_i]:.2e} div={div_lr:.2e} -> pick {pick:.2e}")
    return float(pick)
