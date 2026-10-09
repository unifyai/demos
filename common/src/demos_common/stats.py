"""Small, dependency-light statistics for comparing arms.

Every comparison in a demo reports an effect with an interval, never a bare
mean: proportions with Wilson intervals, differences with a bootstrap over
items (resampling items keeps each item's seeds together), and paired
differences when both arms ran the same items.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def quantile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summary(xs: Sequence[float]) -> dict[str, float]:
    """n, mean, median, p10, p90, min, max."""
    return {
        "n": len(xs),
        "mean": mean(xs),
        "median": quantile(xs, 0.5),
        "p10": quantile(xs, 0.1),
        "p90": quantile(xs, 0.9),
        "min": min(xs) if xs else float("nan"),
        "max": max(xs) if xs else float("nan"),
    }


def bootstrap_diff(
    a: Mapping[str, Sequence[float]],
    b: Mapping[str, Sequence[float]],
    *,
    stat: Callable[[Sequence[float]], float] = mean,
    iters: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """Difference stat(b) - stat(a), with a 95% interval from resampling items.

    a and b map item id -> that item's values (one per seed). Items present in
    both are resampled together (a paired design); items in only one arm are
    resampled within it.
    """
    rng = random.Random(seed)
    shared = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))

    def flat(m: Mapping[str, Sequence[float]], ids: Sequence[str]) -> list[float]:
        return [v for i in ids for v in m[i]]

    point = stat(flat(b, sorted(b))) - stat(flat(a, sorted(a)))
    diffs: list[float] = []
    for _ in range(iters):
        ids = [rng.choice(shared) for _ in shared] if shared else []
        ra = ids + ([rng.choice(only_a) for _ in only_a] if only_a else [])
        rb = ids + ([rng.choice(only_b) for _ in only_b] if only_b else [])
        diffs.append(stat(flat(b, rb)) - stat(flat(a, ra)))
    return {"diff": point, "lo": quantile(diffs, 0.025), "hi": quantile(diffs, 0.975), "items": len(set(a) | set(b))}


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher's exact test p-value for k1/n1 against k2/n2."""

    def logc(n: int, k: int) -> float:
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    total_k, total_n = k1 + k2, n1 + n2

    def p_of(x: int) -> float:
        return math.exp(logc(n1, x) + logc(n2, total_k - x) - logc(total_n, total_k))

    lo, hi = max(0, total_k - n2), min(n1, total_k)
    p_obs = p_of(k1)
    return min(1.0, sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= p_obs * (1 + 1e-9)))
