"""Compile the task's reward-function source strings into safe callables.

The validator passes each reward function as python source inside
dataset_type.reward_functions, with a per-function weight. TRL calls reward
functions as fn(completions, **kwargs) -> list[float]; a broken function must
never take the training run down, so every call is fenced.
"""

import math
import types


def _extract_callable(source: str):
    namespace: dict = {}
    exec(source, namespace)  # trusted input: the validator authored/vetted it
    funcs = [v for k, v in namespace.items()
             if isinstance(v, types.FunctionType) and not k.startswith("_")]
    if not funcs:
        raise ValueError("no function found in reward source")
    # prefer a conventionally named one, else the last defined
    for f in funcs:
        if f.__name__.startswith("reward"):
            return f
    return funcs[-1]


def compile_rewards(reward_functions: list[dict]) -> list:
    """[{reward_func: str, reward_weight: float}, ...] -> list of TRL callables.

    Weights are applied inside the wrapper so TRL's unweighted sum equals the
    validator's weighted scoring.
    """
    wrapped = []
    for i, spec in enumerate(reward_functions or []):
        try:
            fn = _extract_callable(spec["reward_func"])
        except Exception as e:
            print(f"[rewards] function {i} failed to compile: {e}", flush=True)
            continue
        weight = float(spec.get("reward_weight", 1.0))

        def make(fn=fn, weight=weight, idx=i):
            def reward(completions, **kwargs):
                try:
                    vals = fn(completions, **kwargs)
                except Exception as e:
                    print(f"[rewards] fn{idx} raised {type(e).__name__}: {e}", flush=True)
                    return [0.0] * len(completions)
                out = []
                for v in vals:
                    try:
                        v = float(v)
                        if math.isnan(v) or math.isinf(v):
                            v = 0.0
                    except (TypeError, ValueError):
                        v = 0.0
                    out.append(v * weight)
                # length mismatch would corrupt TRL's grouping
                if len(out) != len(completions):
                    out = (out + [0.0] * len(completions))[: len(completions)]
                return out

            reward.__name__ = f"reward_{idx}_{fn.__name__}"
            return reward

        wrapped.append(make())
    return wrapped
