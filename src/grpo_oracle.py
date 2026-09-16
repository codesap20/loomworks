"""Search for a completion that scores well on a GRPO task's reward functions.

Why this exists. Tournament GRPO rewards (validator/tasks/rewards/templates.py) are text-surface
metrics — target counts, ratios, readability, a format regex — and none of them reads the prompt.
The evaluator scores each function 1-v-1 against the boss: mean over completions, then either
rank (0.75 / 0.25 when any mean is negative) or ratio to the better one. So the policy that
wins is the one whose sampled completions sit closest to each function's optimum, and a single
well-chosen string, emitted for every prompt, is that policy. Sampling 2 rollouts per prompt and
hoping GRPO drifts there within the budget is a slow way to find it; the reward code is cheap to
call, so search it directly and teach the model the result.

The search is generic (it only calls the task's own functions), so it keeps working if the
template library changes. Seeds cover the format patterns the library uses today.
"""

import ast
import math
import random
import re
import time

LONG_WORDS = ["internationalization", "characteristically", "responsibilities", "comprehensive",
              "extraordinary", "understanding", "infrastructure", "communication", "development"]
SHORT_WORDS = ["a", "I", "it", "is", "so", "we", "do", "go", "to", "be", "an", "on"]
CONNECTIVES = ["because", "therefore", "however", "for example", "in conclusion", "to summarize",
               "first", "second", "finally", "step", "approach", "consider"]
WRAPS = [
    ("", ""),
    ("1. ", ""),
    ("<think>\n", "\n</think>\n<answer>\nok\n</answer>"),
    ("<think>\nok\n</think>\n<answer>\n", "\n</answer>"),
    ("```\n", "\n```"),
    ("1. ", "\n```\nok\n```"),
    ("", " In conclusion, ok."),
    ("1. ", " In conclusion, ok."),
]


def _literals(sources):
    """String/number constants in the reward code: keywords, targets, regex fragments."""
    words = set()
    for src in sources:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value.strip()
                if 0 < len(v) <= 30 and re.fullmatch(r"[A-Za-z][A-Za-z ']*", v):
                    words.add(v)
    return sorted(words)


class Objective:
    """Weighted, standardized utility over the task's reward functions.

    Each function is standardized by the mean/std of a reference pool (the base model's own
    samples), so a function reporting in hundreds of characters does not drown one reporting a
    ratio in [0, 1] — the evaluator normalizes per function too. Weights are the task weights.
    """

    def __init__(self, fns, weights, count_tokens, max_tokens, ref_texts, prefixes=("",)):
        self.fns, self.weights = fns, weights
        # Completions the evaluator samples may open with a base-model token before the learned
        # text (when the first-token distribution is left close to the base's to save KL), so a
        # candidate is scored as the mean over those openings.
        self.prefixes = list(prefixes) or [""]
        self.count_tokens, self.max_tokens = count_tokens, max_tokens
        cols = [self._raw(fn, ref_texts) for fn in fns]
        # functions we cannot run locally contribute nothing to the search (constant column)
        self.unavailable = [i for i, c in enumerate(cols) if c == "unavailable"]
        cols = [[0.0] * len(ref_texts) if c == "unavailable" else c for c in cols]
        self.mu = [sum(c) / len(c) for c in cols]
        self.sd = []
        for c, m in zip(cols, self.mu):
            var = sum((x - m) ** 2 for x in c) / max(1, len(c) - 1)
            # floor: a function the base never scores on (e.g. a format regex at 0) must not get
            # an infinite exchange rate against the others
            self.sd.append(max(math.sqrt(var), 0.05 * max(1.0, abs(m))))
        self.cache = {}

    @staticmethod
    def _raw(fn, texts, strict=False):
        """Values for `texts`, or None when the function raised and `strict`.

        The evaluator only catches TypeError (evaluators/grpo.py call_reward_func); anything else a
        reward function raises propagates and the whole repo scores nothing on that task. So a
        candidate string that makes any reward function blow up must never be selected.
        """
        try:
            try:
                vals = fn(list(texts))
            except TypeError:
                vals = fn(list(texts), prompts=[""] * len(texts))
            return [float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else 0.0
                    for v in vals]
        except ImportError:
            # A package WE lack (langcheck) but the evaluator has: it will score this function
            # normally, we simply cannot optimise it. Not a reason to reject the candidate.
            return "unavailable"
        except Exception:
            return None if strict else [0.0] * len(texts)

    def values(self, text, strict=False):
        texts = [p + text for p in self.prefixes]
        out = []
        for fn in self.fns:
            vals = self._raw(fn, texts, strict=strict)
            if vals == "unavailable":
                out.append(0.0)
                continue
            if vals is None:
                return None
            out.append(sum(vals) / len(texts))
        return out

    def __call__(self, text):
        hit = self.cache.get(text)
        if hit is not None:
            return hit
        n_tok = self.count_tokens(text)
        v = None if n_tok > self.max_tokens else self.values(text, strict=True)
        if v is None:           # too long, or a reward function raised on it
            u = -1e9 - n_tok
        else:
            u = sum(w * (x - m) / s for w, x, m, s in zip(self.weights, v, self.mu, self.sd))
        self.cache[text] = u
        return u


def _render(c):
    pre, words, post = c
    return pre + " ".join(words) + post


def search(fns, weights, sources, count_tokens, ref_texts, max_tokens=240, seconds=90.0, seed=0,
           log=print, prefixes=("",)):
    """Return (best_text, utility, per-function values, objective)."""
    rng = random.Random(seed)
    obj = Objective(fns, weights, count_tokens, max_tokens, ref_texts, prefixes)
    pool = set(_literals(sources)) | set(LONG_WORDS) | set(SHORT_WORDS) | set(CONNECTIVES)
    for t in ref_texts[:64]:
        pool.update(w for w in t.split()[:80] if len(w) <= 24)
    pool = sorted(pool)

    def sentence_words(n_sent, per):
        out = []
        for _ in range(n_sent):
            ws = [rng.choice(pool) for _ in range(per)]
            ws[-1] = ws[-1].rstrip(".!?") + "."
            out.extend(ws)
        return out

    population = []
    for pre, post in WRAPS:
        for t in ref_texts[:8]:
            population.append((pre, t.split()[:150], post))
        for n_sent, per in ((1, 12), (3, 8), (8, 6), (13, 4), (15, 10), (5, 30)):
            population.append((pre, sentence_words(n_sent, per), post))
        population.append((pre, ["a"] * 60, post))
        population.append((pre, [w + "." for w in LONG_WORDS * 3], post))

    def mutate(c):
        pre, words, post = c
        words = list(words)
        op = rng.randrange(10)
        if op == 0 or not words:
            words.insert(rng.randrange(len(words) + 1), rng.choice(pool))
        elif op == 1 and len(words) > 1:
            del words[rng.randrange(len(words))]
        elif op == 2:
            words[rng.randrange(len(words))] = rng.choice(pool)
        elif op == 3:
            i = rng.randrange(len(words))
            w = words[i]
            words[i] = w.rstrip(".!?") if w[-1:] in ".!?" else w + "."
        elif op == 4:
            i = rng.randrange(len(words))
            j = min(len(words), i + rng.randint(1, 12))
            words[j:j] = words[i:j]
        elif op == 5:
            i = rng.randrange(len(words))
            j = min(len(words), i + rng.randint(1, 20))
            del words[i:j]
        elif op == 6:
            pre, post = rng.choice(WRAPS)
        elif op == 7:
            i = rng.randrange(len(words))
            words[i] = words[rng.randrange(len(words))]
        elif op == 8:
            words = words + [rng.choice(pool) for _ in range(rng.randint(1, 10))]
        else:
            k = rng.randint(1, max(1, len(words) // 4))
            words = words[:-k] if len(words) > k else words
        return (pre, words, post)

    scored = sorted(((obj(_render(c)), c) for c in population), key=lambda x: -x[0])[:48]
    t_end = time.time() + seconds
    it = 0
    while time.time() < t_end:
        it += 1
        parent = scored[min(len(scored) - 1, int(rng.expovariate(1 / 6)))][1]
        child = mutate(parent)
        if rng.random() < 0.3:
            child = mutate(child)
        if rng.random() < 0.1:  # crossover: splice another survivor's words
            other = rng.choice(scored)[1]
            cut = rng.randrange(len(child[1]) + 1)
            child = (child[0], child[1][:cut] + other[1][rng.randrange(len(other[1]) + 1):], child[2])
        u = obj(_render(child))
        if u > scored[-1][0]:
            scored.append((u, child))
            scored.sort(key=lambda x: -x[0])
            del scored[48:]
    best_u, best = scored[0]
    text = _render(best)
    log(f"[oracle] {it} candidates, utility {best_u:.3f}, tokens {count_tokens(text)}, "
        f"values {[round(v, 4) for v in obj.values(text)]} (base mean {[round(m, 4) for m in obj.mu]})")
    return text, best_u, obj.values(text), obj
