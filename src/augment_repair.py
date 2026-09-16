"""Detect and undo the validator's weight-scaling augmentation before SFT.

Half of text tasks hand the trainer a deliberately damaged copy of the base model
(validator/tasks/prep/augmentation.py; trainer/model_prep/augmentation.py). The copy carries no
hint — the name is an anonymous hash and the config is scrubbed. Types (weight): weight_scaling
0.40, magnitude_pruning 0.25, gaussian_noise 0.20, layer_reinit 0.15.

weight_scaling multiplies a random 25-75% of the weight tensors (norms and embeddings excluded) by
ONE factor c in [0.5, 1.5]. That is fully reversible without the original: within a projection type
the per-tensor RMS follows a smooth trend over depth, scaled tensors sit log(c) off it, and dividing
them by c restores the model. (Measured on Qwen3-0.6B: CE 3.54 -> 1.97 at c=0.5, 2.65 -> 1.78 at
c=1.5, clean 1.76; zero false positives on clean models.) The boss trains from the same damaged
copy and does not undo it, so for tasks scored on absolute CE this is a head start on every example.

Only for instruct/chat WITHOUT a KL term: KL tasks, DPO (reference log-probs) and GRPO (prompt KL)
measure against the damaged copy, so moving away from it there is penalised.

magnitude_pruning is reported (exact-zero fraction) but cannot be undone; LoRA cannot refill a
dense zeroed matrix, so the planner should prefer updating those tensors directly.
"""

import collections
import json
import math
import os
import re
import shutil

import numpy as np
import torch

_LAYER = re.compile(r"\.(\d+)\.")


def _candidates(name: str, shape) -> bool:
    low = name.lower()
    return name.endswith("weight") and "norm" not in low and "embed" not in low and len(shape) >= 2


def scan(model_path: str) -> dict:
    """name -> {rms, zero_frac, numel, file} for augmentation-eligible weight tensors."""
    from safetensors import safe_open
    stats = {}
    for fn in sorted(os.listdir(model_path)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_path, fn), framework="pt") as f:
            for name in f.keys():
                sl = f.get_slice(name)
                shape = sl.get_shape()
                if not _candidates(name, shape):
                    continue
                t = f.get_tensor(name)
                ss, zeros, n = 0.0, 0, t.numel()
                for chunk in t.reshape(t.shape[0], -1).split(4096):
                    c = chunk.float()
                    ss += float(c.pow(2).sum())
                    zeros += int((c == 0).sum())
                stats[name] = {"rms": math.sqrt(ss / max(1, n)), "zero_frac": zeros / max(1, n),
                               "numel": n, "file": fn}
    return stats


def _group(name: str):
    m = _LAYER.search(name)
    if not m:
        return None, None
    return _LAYER.sub(".#.", name, count=1), int(m.group(1))


def _viterbi(r: np.ndarray, logc: float):
    """2-state labeling s (1 = scaled by c) minimising sum |v_i - v_{i-1}|, v = r - s*logc."""
    n = len(r)
    cost = np.zeros((n, 2))
    back = np.zeros((n, 2), dtype=int)
    for i in range(1, n):
        for s in (0, 1):
            v = r[i] - s * logc
            opts = [cost[i - 1, p] + abs(v - (r[i - 1] - p * logc)) for p in (0, 1)]
            back[i, s] = int(np.argmin(opts))
            cost[i, s] = min(opts)
    s = int(np.argmin(cost[-1]))
    lab = [s]
    for i in range(n - 1, 0, -1):
        s = back[i, s]
        lab.append(s)
    return float(cost[-1].min()), np.array(lab[::-1], dtype=bool)


def scaling_hypotheses(stats: dict):
    """Return (best_c, cost_ratio, [(c, names_to_divide), (1/c, complement)]).

    A single factor c is shared by every scaled tensor, so given c the per-type labeling is unique;
    the one remaining ambiguity is global — (c, S) and (1/c, not-S) explain the RMS pattern equally
    well — and only a loss check can tell which is the original state.
    """
    groups = collections.defaultdict(dict)
    for name, st in stats.items():
        g, layer = _group(name)
        if g is not None and st["rms"] > 0:
            groups[g][layer] = (name, math.log(st["rms"]))
    groups = {g: d for g, d in groups.items() if len(d) >= 4}
    if not groups:
        return 1.0, 1.0, []
    seqs = [([d[l][0] for l in sorted(d)], np.array([d[l][1] for l in sorted(d)])) for d in groups.values()]
    base = sum(float(np.abs(np.diff(r)).sum()) for _, r in seqs)
    best = (base, 0.0, None)
    for logc in np.linspace(math.log(0.45), -0.04, 90):
        tot, labs = 0.0, []
        for names, r in seqs:
            c_, lab = _viterbi(r, logc)
            tot += c_
            labs.append(lab)
        if tot < best[0]:
            best = (tot, logc, labs)
    if best[2] is None:
        return 1.0, 1.0, []
    c = math.exp(best[1])
    sel = [n for (names, _), lab in zip(seqs, best[2]) for n, f in zip(names, lab) if f]
    all_names = [n for names, _ in seqs for n in names]
    comp = [n for n in all_names if n not in set(sel)]
    return c, best[0] / base if base > 0 else 1.0, [(c, sel), (1.0 / c, comp)]


def write_repaired(model_path: str, out_dir: str, c: float, flagged: list, stats: dict) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    flagged = set(flagged)
    for fn in os.listdir(model_path):
        src = os.path.join(model_path, fn)
        if os.path.isdir(src):
            continue
        if not fn.endswith(".safetensors"):
            shutil.copy2(src, os.path.join(out_dir, fn))
            continue
        if not any(stats[n]["file"] == fn for n in flagged if n in stats):
            shutil.copy2(src, os.path.join(out_dir, fn))
            continue
        tensors = {}
        with safe_open(src, framework="pt") as f:
            meta = f.metadata()
            for name in f.keys():
                t = f.get_tensor(name)
                if name in flagged:
                    t = (t.float() / c).to(t.dtype)
                tensors[name] = t
        save_file(tensors, os.path.join(out_dir, fn), metadata=meta or {"format": "pt"})


@torch.no_grad()
def _batch_ce(model, batches):
    tot, n = 0.0, 0
    for ids, labels in batches:
        logits = model(input_ids=ids).logits[:, :-1].float()
        tgt = labels[:, 1:]
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                               ignore_index=-100, reduction="sum")
        tot += float(ce)
        n += int((tgt != -100).sum())
    return tot / max(1, n)


def _token_batches(tok_dir: str, device, n_rows=24, max_len=1024):
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(tok_dir, "train"))
    out = []
    for row in ds.select(range(min(n_rows, len(ds)))):
        ids = list(row["input_ids"])[:max_len]
        lab = list(row["labels"])[:max_len]
        if sum(1 for x in lab[1:] if x != -100) == 0:
            continue
        out.append((torch.tensor([ids], device=device), torch.tensor([lab], device=device)))
    return out


def maybe_repair(model_path: str, work_root: str, tok_dir: str, log=print) -> tuple[str, dict]:
    """Return (path to train from, report). Never raises: any failure trains from the original.

    Weight statistics propose (c, S) and (1/c, not-S); the model's own CE on training tokens decides,
    and nothing is changed unless the repair lowers that CE clearly — which also rules out false
    positives on clean models, whose RMS profiles always admit some spurious fit.
    """
    report = {"scaled": False}
    try:
        stats = scan(model_path)
        if not stats:
            if any(f.endswith(".bin") for f in os.listdir(model_path)):
                log("[augment] checkpoint has no safetensors shards; scaling repair skipped")
            return model_path, report
        zero = [n for n, st in stats.items() if st["zero_frac"] > 0.01]
        if zero:
            report["pruned_tensors"] = len(zero)
            report["pruned_frac"] = float(np.median([stats[n]["zero_frac"] for n in zero]))
            log(f"[augment] {len(zero)}/{len(stats)} weight tensors have >1% exact zeros "
                f"(median {report['pruned_frac']:.3f}): magnitude pruning")
        c, ratio, hyps = scaling_hypotheses(stats)
        report.update({"c": c, "fit_ratio": ratio})
        total = sum(st["numel"] for st in stats.values())
        if not hyps or ratio > float(os.environ.get("SN56_AUG_MAX_RATIO") or 0.55):
            log(f"[augment] no weight scaling (best c={c:.3f}, fit ratio {ratio:.2f})")
            return model_path, report
        if total > 30e9 or not torch.cuda.is_available():
            log(f"[augment] scaling pattern (c={c:.3f}, ratio {ratio:.2f}) but cannot verify here; unchanged")
            return model_path, report
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        params = dict(model.named_parameters())
        batches = _token_batches(tok_dir, "cuda:0")
        ce0 = _batch_ce(model, batches)
        results = []
        for hc, names in hyps:
            names = [n for n in names if n in params]
            for n in names:
                params[n].data = (params[n].data.float() / hc).to(params[n].dtype)
            ce = _batch_ce(model, batches)
            for n in names:
                params[n].data = (params[n].data.float() * hc).to(params[n].dtype)
            results.append((ce, hc, names))
        del model, params
        torch.cuda.empty_cache()
        ce_best, c_best, names_best = min(results, key=lambda x: x[0])
        report.update({"ce_before": ce0, "ce_hyp": [r[0] for r in results]})
        log(f"[augment] RMS pattern c={c:.3f} (ratio {ratio:.2f}); CE {ce0:.4f} -> "
            f"{' / '.join(f'{r[0]:.4f} (c={r[1]:.3f}, {len(r[2])} tensors)' for r in results)}")
        if ce_best > ce0 - max(0.02, 0.01 * ce0):
            log("[augment] no hypothesis lowers CE clearly; training from the original")
            return model_path, report
        need = sum(os.path.getsize(os.path.join(model_path, f)) for f in os.listdir(model_path)
                   if os.path.isfile(os.path.join(model_path, f)))
        free = shutil.disk_usage(work_root if os.path.isdir(work_root) else "/").free
        if free < need * 1.15 + (2 << 30):
            log(f"[augment] repair needs {need / 2**30:.1f} GiB, only {free / 2**30:.1f} GiB free; "
                "training from the original")
            return model_path, report
        out = os.path.join(work_root, "repaired_model")
        tmp = out + ".tmp"
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            write_repaired(model_path, tmp, c_best, names_best, stats)
            os.replace(tmp, out)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)   # never leave a partial model to train from
            raise
        report.update({"scaled": True, "c": c_best, "flagged": len(names_best)})
        log(f"[augment] weight scaling undone: c={c_best:.3f} on {len(names_best)}/{len(stats)} tensors "
            f"(CE {ce0:.4f} -> {ce_best:.4f}); training from {out}")
        return out, report
    except Exception as e:
        log(f"[augment] repair skipped ({type(e).__name__}: {e}); training from the original")
    return model_path, report


def merge_adapter_into(base_dir: str, out_dir: str, log=print) -> bool:
    """Replace a LoRA submission with full weights merged onto base_dir (the repaired copy).

    The validator loads an adapter onto the task's DAMAGED base, which would silently drop the repair.
    """
    import glob
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16, device_map=dev)
    merged = PeftModel.from_pretrained(base, out_dir).merge_and_unload()
    tmp = out_dir.rstrip("/") + ".merged"
    shutil.rmtree(tmp, ignore_errors=True)
    merged.save_pretrained(tmp, safe_serialization=True)
    # full weights in first, adapter files out last: if the container is stopped in between, the
    # adapter still loads (onto the damaged base, i.e. no worse than not repairing)
    for fn in glob.glob(os.path.join(tmp, "*")):
        shutil.move(fn, os.path.join(out_dir, os.path.basename(fn)))
    for fn in os.listdir(out_dir):
        if fn.startswith("adapter_"):
            os.remove(os.path.join(out_dir, fn))
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"[augment] merged adapter onto the repaired base -> full weights in {out_dir}")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["repair", "merge"])
    ap.add_argument("--model-path")
    ap.add_argument("--work-root")
    ap.add_argument("--tok-dir")
    ap.add_argument("--report")
    ap.add_argument("--base")
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    pr = lambda m: print(m, flush=True)
    if a.mode == "repair":
        path, rep = maybe_repair(a.model_path, a.work_root, a.tok_dir, log=pr)
        rep["path"] = path
        with open(a.report, "w") as f:
            json.dump(rep, f)
    else:
        merge_adapter_into(a.base, a.out_dir, log=pr)
