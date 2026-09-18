"""Make a base model's config.json readable by the transformers version we actually train with.

Why this exists: transformers 5 renamed the RoPE block. A model saved by 5.x writes

    "rope_parameters": {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 32.0, ...}

and no "rope_theta"/"rope_scaling". transformers 4.x does not know that key, so LlamaConfig
falls back to its class defaults — rope_theta 10000.0 and no scaling — and we silently train
(and then ship) a model with the wrong positional encoding. Measured on the live round-1 task
067761fe (Llama-3.2-1B, 2026-09-14): it is the difference between our 1.2314 (13/16) and the
field's 1.0545. The same trap in the other direction ("dtype" vs "torch_dtype") only costs speed.

normalize() rewrites the file in place when the installed transformers is older than 5, and
returns the list of changes for the log. Under transformers 5 it does nothing: that version reads
the old rope_theta/rope_scaling pair natively (verified on 5.9.0). It never raises: a model we cannot parse is left
exactly as it was.
"""
import json
import os
import shutil

# keys inside a transformers-5 rope_parameters block that are NOT part of 4.x rope_scaling
_ROPE_THETA_KEYS = ("rope_theta", "theta")


def _tf_major():
    try:
        import transformers
        return int(str(transformers.__version__).split(".")[0])
    except Exception:
        return 4


def _split_rope(rp):
    """transformers-5 rope_parameters -> (rope_theta, rope_scaling) in 4.x form."""
    if not isinstance(rp, dict):
        return None, None
    theta = None
    scaling = {}
    for k, v in rp.items():
        if k in _ROPE_THETA_KEYS:
            theta = v
        else:
            scaling[k] = v
    # "default"/"none" with nothing else to say is just plain RoPE: 4.x wants rope_scaling=None
    if not scaling or (len(scaling) == 1 and str(scaling.get("rope_type", "")).lower()
                       in ("default", "none")):
        scaling = None
    return theta, scaling


def _fix_one(cfg, major, changed, where=""):
    """Normalise a config dict (or a nested sub-config) in place."""
    if not isinstance(cfg, dict):
        return
    if major >= 5:
        # transformers 5 understands the old rope_theta/rope_scaling pair and the torch_dtype
        # alias, so there is nothing to repair in that direction (verified on 5.9.0).
        return
    if True:
        rp = cfg.get("rope_parameters")
        if isinstance(rp, dict):
            theta, scaling = _split_rope(rp)
            # only write what the block actually carried; never invent a theta
            if theta is not None and cfg.get("rope_theta") != theta:
                changed.append(f"{where}rope_theta {cfg.get('rope_theta')} -> {theta}")
                cfg["rope_theta"] = theta
            if scaling is not None and cfg.get("rope_scaling") != scaling:
                changed.append(f"{where}rope_scaling {cfg.get('rope_scaling')} -> {scaling}")
                cfg["rope_scaling"] = scaling
            elif scaling is None and cfg.get("rope_scaling") is not None:
                changed.append(f"{where}rope_scaling {cfg.get('rope_scaling')} -> None")
                cfg["rope_scaling"] = None
            # leave the 5.x block in place only if it agrees with what we just wrote; a stale
            # copy would be re-read by a 5.x scorer and undo the repair
            cfg.pop("rope_parameters", None)
            changed.append(f"{where}dropped rope_parameters")
        if cfg.get("torch_dtype") is None and cfg.get("dtype") is not None:
            cfg["torch_dtype"] = cfg["dtype"]
            changed.append(f"{where}torch_dtype <- dtype ({cfg['dtype']})")


def normalize(model_path, log=print):
    """Rewrite model_path/config.json for the installed transformers. Returns [] if untouched."""
    path = os.path.join(model_path, "config.json")
    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except Exception as exc:
        log(f"config_compat: cannot read {path}: {exc}")
        return []
    if not isinstance(cfg, dict):
        return []
    major = _tf_major()
    changed = []
    try:
        _fix_one(cfg, major, changed)
        for sub in ("text_config", "language_config", "decoder", "llm_config"):
            if isinstance(cfg.get(sub), dict):
                _fix_one(cfg[sub], major, changed, where=f"{sub}.")
    except Exception as exc:
        log(f"config_compat: normalise failed ({exc}), leaving config as-is")
        return []
    if not changed:
        return []
    try:
        if not os.path.exists(path + ".orig"):
            shutil.copy2(path, path + ".orig")
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(cfg, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except Exception as exc:
        log(f"config_compat: could not write {path}: {exc}")
        return []
    log(f"config_compat (transformers {major}.x): " + "; ".join(changed))
    return changed


def normalize_tokenizer(model_path, log=print):
    """Drop a tokenizer_class the installed transformers cannot import.

    transformers 5 writes "tokenizer_class": "TokenizersBackend"; AutoTokenizer on 4.x then
    raises ValueError("Tokenizer class TokenizersBackend does not exist") and the task dies at
    load. With the key removed, AutoTokenizer infers the class from the model type and the same
    tokenizer.json loads fine. Returns True if the file was rewritten.
    """
    path = os.path.join(model_path, "tokenizer_config.json")
    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except Exception:
        return False                       # no tokenizer here (or unreadable): nothing to repair
    name = cfg.get("tokenizer_class") if isinstance(cfg, dict) else None
    if not name:
        return False
    try:
        import transformers
        if hasattr(transformers, name):
            return False
        # a class the auto-mapping knows by name is fine too, even if not re-exported
        from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
        for pair in TOKENIZER_MAPPING_NAMES.values():
            if name in [p for p in (pair if isinstance(pair, (list, tuple)) else [pair]) if p]:
                return False
    except Exception:
        return False                       # cannot tell: leave it alone rather than break a load
    try:
        cfg.pop("tokenizer_class")
        if not os.path.exists(path + ".orig"):
            shutil.copy2(path, path + ".orig")
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(cfg, fh, indent=1)
        os.replace(tmp, path)
    except Exception as exc:
        log(f"config_compat: could not rewrite {path}: {exc}")
        return False
    log(f"config_compat: dropped unloadable tokenizer_class {name!r}")
    return True


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(p, normalize(p) or "no change", "tok_fixed" if normalize_tokenizer(p) else "")
