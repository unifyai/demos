"""Step 0: measure each model's time to first token and output speed through
OpenRouter, and fit the timing model (data/timing.json).

For each model, streamed calls at several prompt sizes, the prompt prefix warmed
first so it is cached as it is in the runs (the system prompt and most of the
transcript repeat from call to call). Each call's first visible token time,
total time and token counts are kept in runs/calibration.jsonl. Then:
  r  = median of visible tokens / (total - first) over calls with a long reply
  a, b = least squares of (first - reasoning_tokens / r) on prompt kilotokens

  ../../bin/with-openrouter-key uv run python calibrate.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

from demos_common import LLM

from fast_slow_voice import prompts as P
from fast_slow_voice.config import PRESETS
from fast_slow_voice.script import generate
from fast_slow_voice.timing import ModelTiming, Timing

HERE = Path(__file__).parent
SIZES = {"haiku": [1, 4, 16, 40], "luna": [1, 4, 16, 40], "sonnet-low": [1, 4, 12], "sol-low": [1, 4, 12],
         "sonnet-high": [1, 4],
         # era and frontier models (Plan amendment 3)
         "gpt5mini-minimal": [1, 4, 16, 40], "gpt54mini-low": [1, 4, 16, 40], "opus46-low": [1, 4, 12],
         "terra-high": [1, 4, 12], "opus55-low": [1, 4, 12]}
SHORT = "In one sentence of about 30 words, say what kind of records these are."
REASON = ("From these records only, work out how many contacts work at Delta Freight and which cities they are in, "
          "then answer in one short sentence.")
REASON_PRESETS = {"haiku-low", "gpt5mini-minimal", "gpt54mini-low", "opus46-low", "terra-high", "opus55-low"}
# Fits kept exactly as the today's-tier runs used them (only a reasoning rate may be added)
FROZEN = {"anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5", "openai/gpt-6-luna", "openai/gpt-6.1-sol"}
LONG = "In about 150 words, describe what kinds of records these are and what they say about Sam's week."


def filler(chars: int) -> str:
    text = generate(77, 1)[0].world("realistic").as_text()
    while len(text) < chars:
        text += "\n" + text
    return text[:chars]


async def one(llm: LLM, preset: str, k: int, instruction: str, nonce: str) -> dict:
    cfg = PRESETS[preset]
    sys_text = nonce + P.SINGLE
    blocks = [{"type": "text", "text": "Sam's records:\n" + filler(int(k * 1000 * 3.6))}]
    if cfg.anthropic:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
    blocks.append({"type": "text", "text": "\n" + instruction})
    msgs = [{"role": "system", "content": [{"type": "text", "text": sys_text}]}, {"role": "user", "content": blocks}]
    kw = {"max_tokens": cfg.max_tokens}
    if cfg.reasoning is not None:
        kw["reasoning"] = cfg.reasoning
    t0 = time.time()
    r = await llm.chat_stream(cfg.name, msgs, meta={"calibration": preset, "k": k}, **kw)
    kind = "long" if instruction == LONG else "reason" if instruction == REASON else "short"
    return {"preset": preset, "model": cfg.name, "k": k, "instruction": kind,
            "first": r.first_token_s, "total": r.latency_s, "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens, "reasoning_tokens": r.reasoning_tokens,
            "cached_tokens": r.cached_tokens, "cost": r.cost, "text": r.text[:200], "ts": t0}


async def calibrate_reasoning(llm: LLM, preset: str, reps: int, out: Path) -> list[dict]:
    """Calls that make the model reason, to fit its reasoning rate."""
    nonce = f"[calibration {random.randint(0, 10**9)}]\n"
    await one(llm, preset, 4, REASON, nonce)
    rows = [await one(llm, preset, 4, REASON, nonce) for _ in range(reps)]
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


async def calibrate_model(llm: LLM, preset: str, reps: int, out: Path) -> list[dict]:
    rows = []
    nonce = f"[calibration {random.randint(0, 10**9)}]\n"
    for k in SIZES.get(preset, []):
        await one(llm, preset, k, SHORT, nonce)  # warm the cache for this prefix
        for _ in range(reps):
            rows.append(await one(llm, preset, k, SHORT, nonce))
    for _ in range(3):
        rows.append(await one(llm, preset, SIZES[preset][0], LONG, nonce))
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def fit(rows: list[dict]) -> ModelTiming:
    """The first plan's fit: one rate for every output token."""
    rates = [(r["completion_tokens"] - r["reasoning_tokens"]) / (r["total"] - r["first"]) for r in rows
             if r["first"] is not None and r["completion_tokens"] - r["reasoning_tokens"] >= 60 and r["total"] - r["first"] > 0.05]
    rate = statistics.median(rates)
    pts = [(r["prompt_tokens"] / 1000.0, r["first"] - r["reasoning_tokens"] / rate) for r in rows if r["first"] is not None]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx > 0 else 0.0
    b = max(0.0, b)
    a = statistics.median([y - b * x for x, y in pts])
    return ModelTiming(round(a, 3), round(b, 4), round(rate, 1))


def _ols(X: list[list[float]], y: list[float]) -> list[float]:
    """Least squares by the normal equations (small, well-conditioned problems only)."""
    n = len(X[0])
    A = [[sum(row[i] * row[j] for row in X) for j in range(n)] for i in range(n)]
    v = [sum(row[i] * yy for row, yy in zip(X, y)) for i in range(n)]
    for i in range(n):  # Gauss-Jordan
        piv = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[piv], v[i], v[piv] = A[piv], A[i], v[piv], v[i]
        for r in range(n):
            if r != i and A[i][i]:
                f = A[r][i] / A[i][i]
                A[r] = [x - f * yv for x, yv in zip(A[r], A[i])]
                v[r] -= f * v[i]
    return [v[i] / A[i][i] if A[i][i] else 0.0 for i in range(n)]


def fit2(rows: list[dict]) -> ModelTiming:
    """Plan amendment 3: separate rates for reasoning and visible tokens.
    r: median visible rate where the text streams (some models deliver it in one
    burst after reasoning, and then r is set to 2000); a, b and 1/rr by least
    squares of the first visible token time on prompt kilotokens and reasoning
    tokens."""
    ok = [r for r in rows if r["first"] is not None]
    vis = [(r["completion_tokens"] - r["reasoning_tokens"]) / (r["total"] - r["first"]) for r in ok
           if r["completion_tokens"] - r["reasoning_tokens"] >= 40 and r["total"] - r["first"] > 0.2]
    r_vis = statistics.median(vis) if len(vis) >= 3 else 2000.0
    if len({r["reasoning_tokens"] for r in ok}) < 3:
        m = fit(rows)
        return ModelTiming(m.a, m.b, round(r_vis, 1), None)
    X = [[1.0, r["prompt_tokens"] / 1000.0, float(r["reasoning_tokens"])] for r in ok]
    y = [r["first"] for r in ok]
    a, b, inv = _ols(X, y)
    if b < 0:
        a, inv = _ols([[1.0, x[2]] for x in X], y)
        b = 0.0
    rr = min(2000.0, max(20.0, 1.0 / inv)) if inv > 0 else None
    return ModelTiming(round(a, 3), round(b, 4), round(r_vis, 1), None if rr is None else round(rr, 1))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="haiku,sonnet-low,sonnet-high,luna,sol-low")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--reasoning-only", action="store_true", help="only the calls that make a model reason")
    args = ap.parse_args()
    out = HERE / "runs" / "calibration.jsonl"
    presets = args.presets.split(",")
    if not args.fit_only:
        # each preset's rows are appended; a model already calibrated keeps its rows (and its fit)
        async with LLM(HERE / "runs" / "ledger.jsonl", "fast-slow-voice/calibration", max_concurrency=6) as llm:
            if args.reasoning_only:
                await asyncio.gather(*(calibrate_reasoning(llm, p, 4, out) for p in presets if p in REASON_PRESETS))
            else:
                await asyncio.gather(*(calibrate_model(llm, p, args.reps, out) for p in presets))
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    old_path = HERE / "data" / "timing.json"
    timing = Timing.load(old_path) if old_path.exists() else Timing()
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for model, rs in sorted(by_model.items()):
        if model in FROZEN and model in timing.models:
            kept = timing.models[model]
            if model == "anthropic/claude-haiku-5.5":  # its reasoning rate, from haiku-low's reasoning calls
                rsn = [r for r in rs if r["preset"] == "haiku-low"]
                if rsn:
                    kept.rr = fit2([r for r in rs if r["preset"] in ("haiku", "haiku-low")]).rr
            mt = kept
        else:
            mt = fit2(rs)
        timing.models[model] = mt
        firsts = sorted(r["first"] for r in rs if r["first"] is not None and r["instruction"] == "short")
        print(f"{model}: a={mt.a}s b={mt.b}s/ktok r={mt.r}tok/s rr={mt.rr} | n={len(rs)} first median {statistics.median(firsts) if firsts else float('nan'):.2f}s"
              f" | reasoning median {statistics.median(r['reasoning_tokens'] for r in rs)} | cached share "
              f"{sum(r['cached_tokens'] for r in rs) / max(1, sum(r['prompt_tokens'] for r in rs)):.2f}")
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "timing.json").write_text(json.dumps(timing.to_json(), indent=1))


if __name__ == "__main__":
    asyncio.run(main())
