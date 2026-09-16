"""Compile the task's reward-function source strings into safe callables.

The validator passes each reward function as python source inside
dataset_type.reward_functions, with a per-function weight. TRL calls reward
functions as fn(completions, **kwargs) -> list[float]; a broken function must
never take the training run down, so every call is fenced.
"""

import math


class _MissingCall:
    """Attribute of a stubbed module. Raises ImportError when used, which the oracle already reads as
    'the evaluator can score this, we cannot optimise it'."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr):
        return _MissingCall(f"{self._name}.{attr}")

    def __call__(self, *a, **k):
        raise ImportError(f"{self._name} is not installed here (the evaluator has it)")


class _MissingModule:
    """Stands in for a package only the evaluator has (langcheck), so a module-scope import does not
    delete the whole function from our objective.

    Deliberately NOT callable: the evaluator picks the first callable in the namespace
    (rewards/functions.py validate_reward_function), and a callable stub would be picked instead of
    the reward function itself.
    """

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr):
        return _MissingCall(f"{self._name}.{attr}")


def _exec_with_stubs(source: str, namespace: dict) -> None:
    """exec the reward source, stubbing out any import WE lack (the evaluator has them)."""
    import builtins

    real_import = builtins.__import__
    missing: list[str] = []

    def guarded(name, *a, **k):
        try:
            return real_import(name, *a, **k)
        except ImportError:
            missing.append(name)
            return _MissingModule(name)

    builtins.__import__ = guarded
    try:
        exec(source, namespace)
    finally:
        builtins.__import__ = real_import
    if missing:
        print(f"[rewards] stubbed missing imports {sorted(set(missing))}: the evaluator scores this "
              "function for real, we just cannot optimise it", flush=True)


def _extract_callable(source: str):
    namespace: dict = {}
    _exec_with_stubs(source, namespace)  # trusted input: the validator authored/vetted it
    # Pick EXACTLY what the evaluator scores: validator/tasks/rewards/functions.py
    # validate_reward_function takes next(v for k, v in namespace.items() if callable(v)),
    # i.e. the first callable in definition order (helpers and from-imports included).
    # Preferring "reward*" names or the last function trained against a different function
    # than the one being scored whenever the source defined helpers.
    func = next((v for k, v in namespace.items() if callable(v)), None)
    if func is None:
        raise ValueError("no callable found in reward source")
    return func


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
                    try:
                        vals = fn(completions, **kwargs)
                    except TypeError:
                        # evaluator's call_reward_func does the same fallback
                        vals = fn(completions)
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
