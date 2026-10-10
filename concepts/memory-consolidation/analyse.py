"""Analyse the runs: writes results.json and prints the README's tables.

    uv run python analyse.py

Accuracy is the mean score per question (set F1 for T and C, exact for the
rest). "Macro" is the mean of the five family means. Differences between arms
are paired by question: the macro difference with a bootstrap that resamples
questions within each family, the pooled ones with
demos_common.stats.bootstrap_diff. Per-seed means are reported as a check on
how much one history moves the result.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from demos_common.ledger import summarise
import numpy as np
from demos_common.stats import bootstrap_diff, mean, quantile

from memcon.hierarchy import COUNT_LEVELS, TIME_LEVELS

HERE = Path(__file__).parent
RUNS = HERE / "runs"
FAMILIES = ["T", "C", "D", "N", "G"]
FIXED = ["recent", "rolling", "retrieval", "mem0"]
REFS = ["query_time", "full"]
ARMS = ["hierarchy", "hierarchy_time", "hierarchy_count", "rolling", "recent", "retrieval", "mem0", "query_time", "full", "none"]
CPS = ["2w", "8w", "16w", "26w", "break"]
# Today's models only (Amendment 2); the era runs in runs/excluded-era-models/ are not read.
# Sonnet 5.5's runs are excluded too (runs/excluded-sonnet-content-filter/): its usage-policy filter refused most questions.
MODELS = {"haiku": "anthropic/claude-haiku-5.5", "luna": "openai/gpt-6-luna", "sol": "openai/gpt-6.1-sol",
          "opus": "anthropic/claude-opus-5.5"}
TIER = {"haiku": "cheap tier", "luna": "cheap tier, second family", "sol": "mid tier", "opus": "top tier"}
HEADLINE_CP = {"haiku": "26w", "luna": "26w", "sol": "26w", "opus": "8w"}  # where H1 to H3 are judged
ERA_MODELS = {"openai/o4-mini", "google/gemini-2.5-flash"}  # Amendment 1's runs; outside the analysis and the cap
PRICES = {  # list prices per token, 9 October 2026 (input, cache read, input above the long-context threshold, threshold)
    "anthropic/claude-haiku-5.5": (0.10e-6, 0.01e-6, 0.50e-6, 100_000),
    "openai/gpt-6-luna": (0.10e-6, 0.01e-6, 0.20e-6, 272_000),
    "openai/gpt-6.1-sol": (2.0e-6, 0.1e-6, 4.0e-6, 272_000),
    "anthropic/claude-opus-5.5": (4.0e-6, 0.2e-6, 4.0e-6, 10**9),
}


MISSING: dict[str, int] = {}  # per model, answers with no reply text (usage-policy refusals or broken streams)
MISSING_BY_ARM: dict[str, dict[str, dict[str, int]]] = {}  # per model and arm: {"empty": n, "cut": n}


def cut_by_filter(r: dict[str, Any]) -> bool:
    """A reply the provider's filter stopped mid-answer: it opens the JSON object but never closes it, and the
    model stopped far below its token cap (so not a length cut). Seen only on Opus 5.5 (and Sonnet 5.5, excluded);
    every unparsed Haiku, luna and sol reply either hit the 1,000-token cap or is not JSON at all."""
    t = str(r.get("reply", "")).strip()
    return (not r["parsed"] and t.startswith("{") and "}" not in t
            and (r.get("answer_completion_tokens") or 0) < 900)


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def load_answers(slug: str, file: str | None = None) -> list[dict[str, Any]]:
    """A model's main answers (answers.jsonl and answers-<part>.jsonl), or one named file."""
    files = [RUNS / slug / file] if file else sorted(p for p in (RUNS / slug).glob("answers*.jsonl")
                                                      if p.name == "answers.jsonl" or p.name.startswith("answers-"))
    seen, out, missing = set(), [], 0
    by_arm: dict[str, dict[str, int]] = defaultdict(lambda: {"empty": 0, "cut": 0})
    for r in (row for f in files for row in read_jsonl(f)):
        # refused, empty or cut off by the provider's filter: missing, not wrong (counted in "missing_replies")
        empty = not str(r.get("reply", "")).strip()
        if empty or cut_by_filter(r):
            missing += 1
            by_arm[r["arm"]]["empty" if empty else "cut"] += 1
            continue
        k = (r["qid"], r["arm"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    if not file:
        MISSING[slug] = missing
        MISSING_BY_ARM[slug] = {a: dict(v) for a, v in by_arm.items()}
    return out


def macro(rows: list[dict[str, Any]]) -> float:
    fams = defaultdict(list)
    for r in rows:
        fams[r["family"]].append(r["score"])
    return mean([mean(v) for v in fams.values()]) if fams else float("nan")


def macro_ci(rows: list[dict[str, Any]], iters: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    fams = defaultdict(list)
    for r in rows:
        fams[r["family"]].append(r["score"])
    point = mean([mean(v) for v in fams.values()])
    boots = []
    for _ in range(iters):
        boots.append(mean([mean([rng.choice(v) for _ in v]) for v in fams.values()]))
    return point, quantile(boots, 0.025), quantile(boots, 0.975)


def pooled_ci(scores: list[float], iters: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    if not scores:
        return (float("nan"),) * 3
    boots = [mean([rng.choice(scores) for _ in scores]) for _ in range(iters)]
    return mean(scores), quantile(boots, 0.025), quantile(boots, 0.975)


def macro_diff(a: list[dict[str, Any]], b: list[dict[str, Any]], iters: int = 10000, seed: int = 0) -> dict[str, float]:
    """macro(b) - macro(a), paired by question, resampling within each family."""
    rng = random.Random(seed)
    A = {r["qid"]: r for r in a}
    B = {r["qid"]: r for r in b}
    shared = sorted(set(A) & set(B))
    fams = defaultdict(list)
    for q in shared:
        fams[A[q]["family"]].append(q)
    point = mean([mean([B[q]["score"] for q in ids]) - mean([A[q]["score"] for q in ids]) for ids in fams.values()])
    diffs = []
    for _ in range(iters):
        d = []
        for ids in fams.values():
            s = [rng.choice(ids) for _ in ids]
            d.append(mean([B[q]["score"] for q in s]) - mean([A[q]["score"] for q in s]))
        diffs.append(mean(d))
    return {"diff": point, "lo": quantile(diffs, 0.025), "hi": quantile(diffs, 0.975), "items": len(shared)}


def pooled_diff(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, float]:
    """mean(b) - mean(a) over the questions both answered, paired (demos_common.stats.bootstrap_diff)."""
    A = {r["qid"]: [r["score"]] for r in a}
    B = {r["qid"]: [r["score"]] for r in b}
    shared = set(A) & set(B)
    return bootstrap_diff({k: A[k] for k in shared}, {k: B[k] for k in shared}, iters=10000)


def med(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def p90(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return quantile(xs, 0.9) if xs else float("nan")


def arm_table(rows: list[dict[str, Any]], arms: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        m, lo, hi = macro_ci(rs)
        tc = [r["score"] for r in rs if r["family"] in ("T", "C")]
        tcm, tclo, tchi = pooled_ci(tc)
        seeds = sorted({r["seed"] for r in rs})
        out[arm] = {
            "n": len(rs), "macro": m, "macro_lo": lo, "macro_hi": hi, "tc": tcm, "tc_lo": tclo, "tc_hi": tchi,
            **{f: mean([r["score"] for r in rs if r["family"] == f]) for f in FAMILIES if any(r["family"] == f for r in rs)},
            "per_seed_macro": {s: macro([r for r in rs if r["seed"] == s]) for s in seeds},
            "cost_mean": mean([r["cost"] for r in rs]),
            "reply_first_token_median": med([r["reply_first_token_s"] for r in rs]),
            "reply_first_token_p90": p90([r["reply_first_token_s"] for r in rs]),
            "path_median": med([r["path_s"] for r in rs]),
            "memory_chars_median": med([r["memory_chars"] for r in rs]),
            "answer_prompt_tokens_median": med([r["answer_prompt_tokens"] for r in rs]),
            "path_prompt_tokens_mean": mean([r["path_prompt_tokens"] for r in rs]),
            "unparsed": sum(1 for r in rs if not r["parsed"]),
        }
        if arm == "full":
            cold = [r for r in rs if r.get("cold")]
            warm = [r for r in rs if r.get("cold") is False]
            out[arm]["cold_first_token_median"] = med([r["reply_first_token_s"] for r in cold])
            out[arm]["warm_first_token_median"] = med([r["reply_first_token_s"] for r in warm])
            out[arm]["cold_cost_mean"] = mean([r["cost"] for r in cold]) if cold else float("nan")
            out[arm]["warm_cost_mean"] = mean([r["cost"] for r in warm]) if warm else float("nan")
            price = PRICES[rs[0]["model"]]
            out[arm]["uncached_list_cost_mean"] = mean([r["answer_prompt_tokens"] * (price[2] if r["answer_prompt_tokens"] > price[3] else price[0]) for r in rs])
            out[arm]["cache_hit_share_warm"] = (sum(1 for r in warm if r["answer_cached_tokens"] > 0) / len(warm)) if warm else float("nan")
    return out


def comparisons(rows: list[dict[str, Any]], base: str = "hierarchy") -> dict[str, dict[str, Any]]:
    out = {}
    h = [r for r in rows if r["arm"] == base]
    htc = [r for r in h if r["family"] in ("T", "C")]
    for arm in ARMS:
        if arm == base:
            continue
        o = [r for r in rows if r["arm"] == arm]
        if not o:
            continue
        otc = [r for r in o if r["family"] in ("T", "C")]
        out[arm] = {"macro": macro_diff(o, h), "tc": pooled_diff(otc, htc),
                    **{f: pooled_diff([r for r in o if r["family"] == f], [r for r in h if r["family"] == f]) for f in FAMILIES}}
    return out


def by_type(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    groups = defaultdict(list)
    for r in rows:
        qt = r["qtype"]
        if r["family"] == "D":
            qt = f"D_{r['meta'].get('bucket')}"
        elif r["family"] == "N":
            qt = "N_" + qt.rsplit("_", 1)[-1]
        groups[(qt, r["arm"])].append(r["score"])
    for (qt, arm), v in groups.items():
        out[qt][arm] = mean(v)
    return dict(out)


def hypotheses(table: dict[str, Any], comp: dict[str, Any], live: dict[str, Any] | None) -> dict[str, Any]:
    h = table["hierarchy"]
    H1 = {a: comp[a]["macro"]["lo"] > 0 for a in FIXED if a in comp}
    H2 = {a: comp[a]["tc"]["lo"] > 0 for a in FIXED if a in comp}
    H3 = {}
    for a in REFS:
        if a in comp:
            H3[a] = {"within_0.10": comp[a]["tc"]["lo"] > -0.10,
                     "faster": h["reply_first_token_median"] < table[a]["reply_first_token_median"],
                     "5x_cheaper": h["cost_mean"] * 5 <= table[a]["cost_mean"]}
    H4 = None
    if live:
        off = live["modes"].get("off", {}).get("median_first_token")
        bg = live["modes"].get("background", {}).get("median_first_token")
        inl = live["modes"].get("inline", {})
        if off and bg:
            H4 = {"background_within_10pct_of_off": abs(bg - off) <= 0.10 * off, "inline_adds_delay": inl.get("mean_wait", 0) > 0.05}
    return {"H1": H1, "H1_all": all(H1.values()) if H1 else None, "H2": H2, "H2_all": all(H2.values()) if H2 else None,
            "H3": H3, "H3_all": all(all(v.values()) for v in H3.values()) if H3 else None, "H4": H4,
            "H4_all": all(H4.values()) if H4 else None}


# ------------------------------------------------------------ write paths

# Ledger rows before this moment belong to runs on data from before the stable-ID fix (runs/discarded/);
# it is a minute before the data in data/ was generated.
DATA_CUTOFF = 1791586914


def write_paths(slug: str) -> dict[str, Any]:
    """Each history's write-path cost and measures, from the state files of the
    counted runs (which record each call's cost); embedding calls from the
    ledger, after DATA_CUTOFF."""
    emb: dict[tuple[str, str], float] = defaultdict(float)
    for r in (row for path in sorted(RUNS.glob("ledger*.jsonl")) for row in read_jsonl(path)):
        meta = r.get("meta") or {}
        if "error" in r or r.get("ts", 0) < DATA_CUTOFF or r.get("model") != "voyageai/voyage-4":
            continue
        if r.get("tag") == "retrieval/embed_records":
            emb[(str(meta.get("history")), "retrieval")] += r.get("cost") or 0.0
        elif r.get("tag") == "mem0/embed" and meta.get("slug", "haiku") == slug:
            emb[(str(meta.get("history")), "mem0")] += r.get("cost") or 0.0
    out: dict[str, Any] = {"hierarchy": {}, "rolling": {}, "mem0": {}, "retrieval": {}}
    hier, mem0, roll = {}, {}, {}
    for hdir in sorted((RUNS / slug).glob("s[0-9]*")):
        h = hdir.name
        if (RUNS / "emb" / f"{h}.npy").exists():
            out["retrieval"][h] = {"cost": emb[(h, "retrieval")], "interactions": len(np.load(RUNS / "emb" / f"{h}.npy"))}
        p = hdir / "hierarchy.json"
        if p.exists():
            nodes = {f"{n['axis']}{n['level']}:{n['block']}": n for n in json.loads(p.read_text())["nodes"]}
            called = [n for n in nodes.values() if n["called"]]
            n_int = max((n["last"] or 0) for n in nodes.values())
            out["hierarchy"][h] = {"cost": sum(n["cost"] for n in called), "calls": len(called), "interactions": n_int,
                                   "prompt_tokens": sum(n["prompt_tokens"] for n in called), "completion_tokens": sum(n["completion_tokens"] for n in called)}
            lv = defaultdict(list)
            for n in called:
                lv[f"{n['axis']} level {n['level']}"].append(n["prompt_tokens"])
            hier[h] = {"summaries": len(nodes), "calls": len(called),
                       "prompt_tokens_by_level": {k: {"median": statistics.median(v), "max": max(v), "n": len(v)} for k, v in sorted(lv.items())},
                       "inline_wait": cascades(nodes)}
        log = read_jsonl(hdir / "mem0" / "log.jsonl")
        if log:
            calls = [c for r in log for c in r["calls"]]
            out["mem0"][h] = {"cost": sum(c["cost"] for c in calls) + emb[(h, "mem0")], "embed_cost": emb[(h, "mem0")], "calls": len(calls),
                              "interactions": len(log), "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
                              "completion_tokens": sum(c["completion_tokens"] for c in calls)}
            w = [r["wall_s"] for r in log]
            mem0[h] = {"exchanges": len(log), "write_s_median": statistics.median(w), "write_s_p90": quantile(w, 0.9),
                       "calls_per_exchange": mean([len(r["calls"]) for r in log])}
        vs = read_jsonl(hdir / "rolling.jsonl")
        if vs:
            out["rolling"][h] = {"cost": sum(v["cost"] for v in vs), "calls": len(vs), "interactions": vs[-1]["last"],
                                 "prompt_tokens": sum(v["prompt_tokens"] for v in vs), "completion_tokens": sum(v["completion_tokens"] for v in vs)}
            roll[h] = {"updates": len(vs), "update_s_median": statistics.median(v["latency_s"] for v in vs),
                       "final_words": len(vs[-1]["text"].split()), "max_prompt_tokens": max(v["prompt_tokens"] for v in vs)}
    for k in ("hierarchy", "rolling", "mem0", "retrieval"):
        for h, v in out[k].items():
            if v.get("interactions"):
                v["cost_per_interaction"] = v["cost"] / v["interactions"]
    out["hierarchy_detail"], out["mem0_detail"], out["rolling_detail"] = hier, mem0, roll
    return out


def cascades(nodes: dict[str, dict[str, Any]]) -> dict[str, float]:
    """What each turn would wait if the summaries its logging triggers were
    awaited inline. A turn's logging closes the days since the previous active
    day (each a chain: day, then the week if it completes one, and so on) and,
    every tenth interaction, a block of 10 (then 40, 120, 480 if completed).
    Chains run side by side; levels within a chain run one after another."""
    lat = {k: (n["latency_s"] if n["called"] else 0.0) for k, n in nodes.items()}

    def chain(axis: str, level: int, block: int) -> float:
        levels = TIME_LEVELS if axis == "time" else COUNT_LEVELS
        total = lat.get(f"{axis}{level}:{block}", 0.0)
        if level < len(levels):
            fan = levels[level][1]
            parent = f"{axis}{level + 1}:{(block + 1) // fan - 1}"
            if (block + 1) % fan == 0 and parent in nodes:
                total += chain(axis, level + 1, (block + 1) // fan - 1)
        return total

    days = sorted((n for n in nodes.values() if n["axis"] == "time" and n["level"] == 1), key=lambda n: n["block"])
    closes_at: dict[int, list[int]] = {}
    prev = 0
    for d in days:
        if d["first"] is not None:
            closes_at[d["first"]] = list(range(prev, d["block"]))
            prev = d["block"]
    n_inter = max((n["last"] or 0) for n in nodes.values())
    size = COUNT_LEVELS[0][1]
    waits = []
    for idx in range(1, n_inter + 1):
        w = [chain("time", 1, b) for b in closes_at.get(idx, [])]
        if idx % size == 0:
            w.append(chain("count", 1, idx // size - 1))
        waits.append(max(w) if w else 0.0)
    nz = [w for w in waits if w > 0]
    return {"interactions": len(waits), "share_of_turns_waiting": len(nz) / len(waits) if waits else 0.0,
            "wait_median_s": statistics.median(nz) if nz else 0.0, "wait_p90_s": quantile(nz, 0.9) if nz else 0.0,
            "wait_max_s": max(nz) if nz else 0.0, "wait_mean_per_turn_s": mean(waits) if waits else 0.0}


def live_summary(slug: str) -> dict[str, Any] | None:
    rows = read_jsonl(RUNS / slug / "live.jsonl")
    if not rows:
        return None
    modes: dict[str, Any] = {}
    for mode in ("off", "background", "inline", "mem0_inline"):
        rs = [r for r in rows if r["mode"] == mode]
        if not rs:
            continue
        ft = [r["turn_first_token_s"] for r in rs if r["turn_first_token_s"] is not None]
        modes[mode] = {"turns": len(rs), "median_first_token": med(ft), "p90_first_token": p90(ft), "mean_wait": mean([r["wait_s"] for r in rs]),
                       "max_wait": max(r["wait_s"] for r in rs), "max_loop_lag": max(r["max_loop_lag_s"] for r in rs),
                       "median_ttft": med([r["ttft_s"] for r in rs])}
    return {"modes": modes}


def fmt(x: float, d: int = 2) -> str:
    return "–" if x is None or x != x else f"{x:.{d}f}"


def ci(m: float, lo: float, hi: float) -> str:
    return f"{fmt(m)} ({fmt(lo)} to {fmt(hi)})"


def main() -> None:
    results: dict[str, Any] = {}
    for slug in MODELS:
        rows = load_answers(slug)
        if not rows:
            continue
        R: dict[str, Any] = {"by_checkpoint": {}, "comparisons": {}, "by_type": {}}
        for cp in CPS:
            rs = [r for r in rows if r["checkpoint"] == cp]
            if not rs:
                continue
            R["by_checkpoint"][cp] = arm_table(rs, ARMS)
            if any(r["arm"] == "hierarchy" for r in rs):
                R["comparisons"][cp] = comparisons(rs)
            R["by_type"][cp] = by_type(rs)
        hc = HEADLINE_CP[slug]
        if hc in R["comparisons"]:
            R["hypotheses"] = {"checkpoint": hc, **hypotheses(R["by_checkpoint"][hc], R["comparisons"][hc], live_summary("haiku"))}
        # the gap to the best fixed-budget baseline, by checkpoint
        sweep = {}
        for cp, tab in R["by_checkpoint"].items():
            if "hierarchy" not in tab:
                continue
            best = max((a for a in FIXED if a in tab), key=lambda a: tab[a]["macro"], default=None)
            best_tc = max((a for a in FIXED if a in tab), key=lambda a: tab[a]["tc"], default=None)
            sweep[cp] = {"best_fixed_macro": best, "diff_macro": R["comparisons"][cp][best]["macro"] if best else None,
                         "best_fixed_tc": best_tc, "diff_tc": R["comparisons"][cp][best_tc]["tc"] if best_tc else None}
        R["sweep"] = sweep
        # details by age, all checkpoints pooled
        D = defaultdict(dict)
        for arm in ARMS:
            for b in ("recent", "medium", "old"):
                v = [r["score"] for r in rows if r["arm"] == arm and r["family"] == "D" and r["meta"].get("bucket") == b]
                if v:
                    D[arm][b] = {"mean": mean(v), "n": len(v)}
        R["details_by_age"] = dict(D)
        # after the longest break: time view against count view, on count and time questions
        brk = [r for r in rows if r["checkpoint"] == "break"]
        if brk:
            R["after_break"] = {}
            for fam in ("C", "T"):
                fr = [r for r in brk if r["family"] == fam]
                arms = {a: [r for r in fr if r["arm"] == a] for a in ARMS}
                R["after_break"][fam] = {
                    "means": {a: mean([r["score"] for r in v]) for a, v in arms.items() if v},
                    "count_minus_time": pooled_diff(arms["hierarchy_time"], arms["hierarchy_count"]) if arms["hierarchy_count"] else None,
                    "both_minus_time": pooled_diff(arms["hierarchy_time"], arms["hierarchy"]) if arms["hierarchy"] else None,
                }
        # the budget guard, and replies that did not parse
        R["missing_replies"] = MISSING.get(slug, 0)
        R["missing_by_arm"] = MISSING_BY_ARM.get(slug, {})
        R["guard"] = {a: {"rows": sum(1 for r in rows if r["arm"] == a),
                          "dropped_any": sum(1 for r in rows if r["arm"] == a and (r["info"] or {}).get("dropped", 0) > 0),
                          "unparsed": sum(1 for r in rows if r["arm"] == a and not r["parsed"])} for a in ARMS}
        R["write_paths"] = write_paths(slug)
        R["live"] = live_summary(slug)
        results[slug] = R
    results["tier_trend"] = tier_trend()
    led = summarise([str(p) for p in sorted(RUNS.glob("ledger*.jsonl"))])
    by_model: dict[str, float] = defaultdict(float)
    for k, v in led.items():
        by_model[k.split(" | ", 1)[1]] += v.get("cost", 0.0)
    era = sum(c for m, c in by_model.items() if m in ERA_MODELS)
    total = sum(by_model.values())
    results["ledger"] = {"total_cost": total, "era_models_cost": era, "todays_models_cost": total - era,
                         "excluded_sonnet_cost": by_model.get("anthropic/claude-sonnet-5.5", 0.0),
                         "by_model": dict(by_model), "by_tag_model": led}
    (HERE / "results.json").write_text(json.dumps(results, indent=1, default=str))
    report(results)


def tier_trend() -> dict[str, Any]:
    """For each (seed, checkpoint) a stronger tier ran, every tier's arms on exactly
    the questions the stronger tier answered: means, and the hierarchy's margin
    over the long context and over the best fixed-budget arm present."""
    rows = {slug: load_answers(slug) for slug in MODELS}
    out: dict[str, Any] = {}
    for strong in ("sol", "opus"):
        if not rows[strong]:
            continue
        keys = {(r["seed"], r["checkpoint"]) for r in rows[strong]}
        for seed, cp in sorted(keys):
            qset = {(r["qid"], r["arm"]) for r in rows[strong] if r["seed"] == seed and r["checkpoint"] == cp}
            tab: dict[str, Any] = {}
            for slug in MODELS:
                rs = [r for r in rows[slug] if (r["qid"], r["arm"]) in qset]
                if not rs:
                    continue
                by = defaultdict(list)
                for r in rs:
                    by[r["arm"]].append(r)
                t: dict[str, Any] = {"arms": {a: {"mean": mean([x["score"] for x in v]), "n": len(v), "cost": mean([x["cost"] for x in v]),
                                                  "first_token": med([x["reply_first_token_s"] for x in v])} for a, v in by.items()}}
                h = by.get("hierarchy", [])
                tc = {a: [x for x in v if x["family"] in ("T", "C")] for a, v in by.items()}
                t["tc_means"] = {a: mean([x["score"] for x in v]) for a, v in tc.items() if v}
                if h and by.get("full"):
                    t["hierarchy_minus_full"] = pooled_diff(by["full"], h)
                    t["hierarchy_minus_full_tc"] = pooled_diff(tc["full"], tc["hierarchy"])
                fixed = [a for a in FIXED if a in by]
                if h and fixed:
                    best = max(fixed, key=lambda a: mean([x["score"] for x in by[a]]))
                    t["best_fixed"] = best
                    t["hierarchy_minus_best_fixed"] = pooled_diff(by[best], h)
                    best_tc = max(fixed, key=lambda a: t["tc_means"].get(a, -1.0))
                    t["best_fixed_tc"] = best_tc
                    t["hierarchy_minus_best_fixed_tc"] = pooled_diff(tc[best_tc], tc["hierarchy"])
                if h and by.get("query_time"):
                    qt_q = {x["qid"] for x in by["query_time"]}
                    t["hierarchy_minus_query_time_on_its_questions"] = pooled_diff(by["query_time"], [x for x in h if x["qid"] in qt_q])
                tab[slug] = t
            out[f"{strong}: seed {seed}, {cp}"] = tab
    return out


def report(results: dict[str, Any]) -> None:
    for slug in MODELS:
        R = results.get(slug)
        if not R:
            continue
        print(f"\n## {MODELS[slug]} ({TIER[slug]})")
        for cp, tab in R["by_checkpoint"].items():
            print(f"\n### {cp}\n")
            print("| arm | macro (95% CI) | T+C (95% CI) | T | C | D | N | G | $/question | first token, median (p90) s | memory chars |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for arm, v in tab.items():
                print(f"| {arm} | {ci(v['macro'], v['macro_lo'], v['macro_hi'])} | {ci(v['tc'], v['tc_lo'], v['tc_hi'])} | "
                      + " | ".join(fmt(v.get(f)) for f in FAMILIES)
                      + f" | {v['cost_mean']:.5f} | {fmt(v['reply_first_token_median'])} ({fmt(v['reply_first_token_p90'])}) | {v['memory_chars_median']:,.0f} |")
            if cp in R["comparisons"]:
                print("\nhierarchy minus arm, paired (95% CI):\n")
                print("| arm | macro | T+C | T | C | D | N | G |")
                print("|---|---|---|---|---|---|---|---|")
                for arm, c in R["comparisons"][cp].items():
                    print(f"| {arm} | {ci(c['macro']['diff'], c['macro']['lo'], c['macro']['hi'])} | {ci(c['tc']['diff'], c['tc']['lo'], c['tc']['hi'])} | "
                          + " | ".join(ci(c[f]["diff"], c[f]["lo"], c[f]["hi"]) if c[f]["items"] else "–" for f in FAMILIES) + " |")
        print("\n### sweep: macro / T+C by checkpoint\n")
        cps = [c for c in CPS if c in R["by_checkpoint"]]
        print("| arm | " + " | ".join(cps) + " |")
        print("|---|" + "---|" * len(cps))
        for arm in ARMS:
            cells = []
            for c in cps:
                v = R["by_checkpoint"][c].get(arm)
                cells.append(f"{fmt(v['macro'])} / {fmt(v['tc'])}" if v else "–")
            print(f"| {arm} | " + " | ".join(cells) + " |")
        print("\nper-seed macro at 26w:", json.dumps({a: {k: round(x, 3) for k, x in v["per_seed_macro"].items()}
                                                      for a, v in R["by_checkpoint"].get("26w", {}).items()}))
        if "after_break" in R:
            print("\nafter the break:", json.dumps(R["after_break"], indent=1, default=str))
        print("\nguard and parsing:", json.dumps(R["guard"]))
        print("missing replies (refused or empty / cut off by the provider's filter), left out:", R["missing_replies"], json.dumps(R["missing_by_arm"]))
        for cp in cps:
            f = R["by_checkpoint"][cp].get("full")
            if f:
                print(f"full at {cp}: cold first token {fmt(f['cold_first_token_median'])}s, warm {fmt(f['warm_first_token_median'])}s, "
                      f"cold ${f['cold_cost_mean']:.4f}, warm ${f['warm_cost_mean']:.4f}, uncached list ${f['uncached_list_cost_mean']:.4f}, "
                      f"warm cache hits {fmt(f['cache_hit_share_warm'])}")
        if "hypotheses" in R:
            print(f"\nhypotheses at {R['hypotheses']['checkpoint']}:", json.dumps(R["hypotheses"], indent=1))
        print("\nsweep:", json.dumps(R["sweep"], indent=1, default=str))
        print("\ndetails by age:", json.dumps(R["details_by_age"], indent=1))
        print("\nwrite paths:", json.dumps({k: v for k, v in R["write_paths"].items()}, indent=1, default=str)[:4000])
        print("\nlive:", json.dumps(R["live"], indent=1))
    for key, tab in results.get("tier_trend", {}).items():
        print(f"\n### tier trend, {key} (the questions the stronger tier answered)\n")
        arms = sorted({a for t in tab.values() for a in t["arms"]}, key=ARMS.index)
        print("| model | " + " | ".join(arms) + " | hierarchy - full | hierarchy - best fixed |")
        print("|---|" + "---|" * (len(arms) + 2))
        for slug, t in tab.items():
            cells = [fmt(t["arms"][a]["mean"]) if a in t["arms"] else "–" for a in arms]
            hf = t.get("hierarchy_minus_full")
            hb = t.get("hierarchy_minus_best_fixed")
            print(f"| {MODELS[slug]} | " + " | ".join(cells) + f" | {ci(hf['diff'], hf['lo'], hf['hi']) if hf else '–'} | "
                  + (f"{ci(hb['diff'], hb['lo'], hb['hi'])} ({t['best_fixed']})" if hb else "–") + " |")
        print("\nT and C questions only:\n")
        print("| model | " + " | ".join(arms) + " | hierarchy - full | hierarchy - best fixed |")
        print("|---|" + "---|" * (len(arms) + 2))
        for slug, t in tab.items():
            cells = [fmt(t["tc_means"][a]) if a in t["tc_means"] else "–" for a in arms]
            hf = t.get("hierarchy_minus_full_tc")
            hb = t.get("hierarchy_minus_best_fixed_tc")
            print(f"| {MODELS[slug]} | " + " | ".join(cells) + f" | {ci(hf['diff'], hf['lo'], hf['hi']) if hf else '–'} | "
                  + (f"{ci(hb['diff'], hb['lo'], hb['hi'])} ({t['best_fixed_tc']})" if hb else "–") + " |")
    L = results["ledger"]
    print(f"\nledger total ${L['total_cost']:.2f}: today's models ${L['todays_models_cost']:.2f} (of which Sonnet 5.5, excluded, "
          f"${L['excluded_sonnet_cost']:.2f}), era models ${L['era_models_cost']:.2f}")
    print("by model:", json.dumps({k: round(v, 3) for k, v in sorted(L["by_model"].items(), key=lambda x: -x[1])}))


if __name__ == "__main__":
    main()
