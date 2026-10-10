"""Every number in the README: per condition and arm, the measures with
intervals, the paired differences the plan's criteria use, and cost from the
ledgers. Writes results.json and prints the tables.

  uv run python analyze.py [--conditions anthropic-realistic-d0,...]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from demos_common.ledger import summarise
from demos_common.stats import bootstrap_diff, mean, quantile, wilson

from fast_slow_voice.sim import is_placeholder

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SPLIT = ["relay", "slow_speaks"]
SLOW = ["slow_only", "slow_filler"]
FAST = ["fast_only", "fast_only_think", "fast_context", "fast_context_think"]
ORDER = SPLIT + SLOW + FAST
NOT_MAIN = ("smoke", "sanity", "calibration")
MAIN = ["anthropic-realistic-d0", "anthropic-small-d0", "anthropic-realistic-d8", "cheap-anthropic-realistic-d0",
        "top-anthropic-realistic-d0", "cheap-openai-realistic-d0"]
REQUEST_KINDS = {"question", "topic_change", "action", "repeat_request"}


def load(cond: str) -> list[dict]:
    recs: dict[tuple, dict] = {}
    for line in (RUNS / f"{cond}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[(r["arm"], r["script"], r["seed"])] = r
    judged: dict[tuple, dict] = {}
    jp = RUNS / "judge" / f"{cond}.jsonl"
    if jp.exists():
        for line in jp.read_text().splitlines():
            j = json.loads(line)
            if j.get("judgement") is not None:
                judged[(j["arm"], j["script"], j["seed"])] = j["judgement"]
    out = []
    for k, r in recs.items():
        r["judgement"] = judged.get(k)
        out.append(r)
    return out


def ttfw_plan(r: dict) -> list[tuple[str, float]]:
    """Time to the first word as the plan defines it: for each turn of Sam's
    that expects a reply, from the end of his speech to the start of the next
    agent speech. A turn that got no reply before Sam spoke again (recorded as
    None) counts as the wait until Ava's next words, whenever they came (the
    call's end if she never spoke again): at least 3.0 s in every recorded call,
    since Sam waits 3 s before going on. The first analysis dropped those turns
    from the median and p90, which left out exactly the longest waits."""
    spoken = sorted(u["t_start"] for u in r["utterances"] if u["t_start"] is not None)
    persons = {p["turn"]: p for p in r["persons"]}
    out = []
    for t in r["metrics"]["ttfw"]:
        if t["ttfw"] is not None:
            out.append((t["kind"], t["ttfw"]))
            continue
        end = persons[t["turn"]]["t_end"]
        nxt = next((x for x in spoken if x >= end - 1e-9), r["t_end"])
        out.append((t["kind"], round(max(nxt, end) - end, 3)))
    return out


def conv_measures(r: dict) -> dict:
    m = r["metrics"]
    wanted = [i for i in m["items"] if not i["withdrawn"]]
    withdrawn = [i for i in m["items"] if i["withdrawn"]]
    tt = [t["ttfw"] for t in m["ttfw"]]
    answered = [t for t in tt if t is not None]
    plan = ttfw_plan(r)
    j = r.get("judgement") or {}
    return {
        "arm": r["arm"], "script": r["script"], "seed": r["seed"],
        "correct": [int(i["delivered"]) for i in wanted],
        "correct_easy": [int(i["delivered"]) for i in wanted if i["difficulty"] == "easy"],
        "correct_hard": [int(i["delivered"]) for i in wanted if i["difficulty"] == "hard"],
        "correct_by_kind": {k: [int(i["delivered"]) for i in wanted if i["kind"] == k] for k in {i["kind"] for i in wanted}},
        "correct_by_episode": {k: [int(i["delivered"]) for i in wanted if i["episode"] == k] for k in {i["episode"] for i in wanted}},
        "t_answer": [i["t_answer"] for i in wanted if i["delivered"]],
        "trap": [int(i["trap_said"]) for i in m["items"] if "trap_said" in i and not i["withdrawn"]],
        "stale": [int(bool(i.get("stale"))) for i in withdrawn],
        "ttfw": [v for _, v in plan],
        "ttfw_answered": answered,
        "ttfw_all": tt,
        "ttfw_requests": [v for k, v in plan if k in REQUEST_KINDS],
        "ttfw_other": [v for k, v in plan if k not in REQUEST_KINDS],
        "ttfw_requests_answered": [t["ttfw"] for t in m["ttfw"] if t["kind"] in REQUEST_KINDS and t["ttfw"] is not None],
        "slow_requests": [int(t["ttfw"] is None or t["ttfw"] > 2.0) for t in m["ttfw"] if t["kind"] in REQUEST_KINDS],
        "slow_turns": [int(t is None or t > 2.0) for t in tt],
        "slow_turns_3": [int(t is None or t > 3.0) for t in tt],
        "no_reply": [int(t is None) for t in tt],
        "ttfw_median": quantile(answered, 0.5) if answered else float("nan"),
        "longest_silence": m["longest_silence"],
        "waiting_silence": m["waiting_silence"],
        "invented_prog": len(m["invented"]),
        "invented_any": int(bool(m["invented"])),
        "nat": j.get("naturalness"),
        "double": len(j.get("double_replies", [])) if j else None,
        "contra": len(j.get("contradictions", [])) if j else None,
        "wrong": len(j.get("wrong_or_invented", [])) if j else None,
        "wrong_any": None if not j else int(bool(j.get("wrong_or_invented")) or bool(m["invented"])),
        "judge_stale": len(j.get("stale", [])) if j else None,
        "judge_fillers": j.get("redundant_fillers") if j else None,
        "redundant_holds": m["redundant_holds"],
        "double_proxy": m["double_reply_proxy"],
        "holding_lines": m["holding_lines"],
        "cost": m["cost"],
        "n_calls": m["n_calls"],
        "duration": m["duration"],
        "action_ok": [int(a["sent_right"]) for a in m["actions"]],
        "action_premature": [int(a["premature_claim"]) for a in m["actions"]],
        "action_confirmed": [int(a["confirmed"]) for a in m["actions"]],
        "flags": m["flags"],
        # a single model's calls that call a tool: did it also say something (the holding line its prompt asks for)?
        "words_with_tools": [int(bool(c["text"]) and not is_placeholder(c["text"])) for c in r["calls"]
                             if c["role"] == "voice" and c["tool_calls"]],
    }


def by_script(convs: list[dict], key: str, flatten: bool = True) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for c in convs:
        v = c[key]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if isinstance(v, list):
            if v:  # a call with no such items adds nothing (and no empty script to resample)
                out[c["script"]] += [float(x) for x in v]
        else:
            out[c["script"]].append(float(v))
    return dict(out)


def prop(convs: list[dict], key: str) -> dict:
    xs = [x for c in convs for x in c[key]]
    k, n = int(sum(xs)), len(xs)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": k / n if n else float("nan"), "lo": lo, "hi": hi}


def avg(convs: list[dict], key: str, stat=mean) -> dict:
    """A per-call (or pooled) statistic with a bootstrap interval over scripts."""
    d = by_script(convs, key)
    if not d:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    zero = {s: [0.0] * len(v) for s, v in d.items()}
    b = bootstrap_diff(zero, d, stat=stat, iters=4000)
    return {"mean": b["diff"] if stat is mean else stat([x for v in d.values() for x in v]), "lo": b["lo"], "hi": b["hi"],
            "n": sum(len(v) for v in d.values())}


def med(xs):
    return quantile(xs, 0.5)


def diff(a: list[dict], b: list[dict], key: str, stat=mean) -> dict:
    """stat(b) - stat(a), paired over scripts."""
    r = bootstrap_diff(by_script(a, key), by_script(b, key), stat=stat, iters=4000)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}


def summarise_arm(convs: list[dict]) -> dict:
    allt = [x for c in convs for x in c["ttfw"]]
    ans = [x for c in convs for x in c["ttfw_answered"]]
    return {
        "calls": len(convs), "scripts": len({c["script"] for c in convs}), "seeds": sorted({c["seed"] for c in convs}),
        "correct": prop(convs, "correct"), "correct_easy": prop(convs, "correct_easy"), "correct_hard": prop(convs, "correct_hard"),
        "trap": prop(convs, "trap"), "stale": prop(convs, "stale"),
        "invented_any": prop([{"x": [c["invented_any"]]} for c in convs], "x"),
        "invented_per_call": avg(convs, "invented_prog"),
        "judge_wrong_per_call": avg(convs, "wrong"), "judge_double_per_call": avg(convs, "double"),
        "judge_contra_per_call": avg(convs, "contra"), "judge_stale_per_call": avg(convs, "judge_stale"),
        "judge_fillers_per_call": avg(convs, "judge_fillers"),
        "naturalness": avg(convs, "nat"),
        "ttfw_median": quantile(allt, 0.5) if allt else float("nan"), "ttfw_p90": quantile(allt, 0.9) if allt else float("nan"),
        "ttfw_median_ci": avg(convs, "ttfw", stat=med),
        "ttfw_answered_median": quantile(ans, 0.5) if ans else float("nan"),
        "ttfw_answered_p90": quantile(ans, 0.9) if ans else float("nan"),
        "ttfw_requests_answered_median": avg(convs, "ttfw_requests_answered", stat=med),
        "wrong_any": prop([{"x": [c["wrong_any"]]} for c in convs if c["wrong_any"] is not None], "x"),
        "slow_turns": prop(convs, "slow_turns"), "slow_turns_3": prop(convs, "slow_turns_3"), "no_reply": prop(convs, "no_reply"),
        "ttfw_requests_median": avg(convs, "ttfw_requests", stat=med), "ttfw_other_median": avg(convs, "ttfw_other", stat=med),
        "slow_requests": prop(convs, "slow_requests"),
        "longest_silence": avg(convs, "longest_silence"),
        "t_answer_median": avg(convs, "t_answer", stat=med),
        "waiting_silence_mean": avg(convs, "waiting_silence"),
        "redundant_holds_per_call": avg(convs, "redundant_holds"), "double_proxy_per_call": avg(convs, "double_proxy"),
        "cost_per_call": avg(convs, "cost"), "model_calls_per_call": avg(convs, "n_calls"),
        "action_ok": prop(convs, "action_ok"), "action_premature": prop(convs, "action_premature"),
        "action_confirmed": prop(convs, "action_confirmed"),
        "by_kind": {k: prop([{"x": c["correct_by_kind"].get(k, [])} for c in convs], "x")
                    for k in sorted({k for c in convs for k in c["correct_by_kind"]})},
        "by_episode": {k: prop([{"x": c["correct_by_episode"].get(k, [])} for c in convs], "x")
                       for k in sorted({k for c in convs for k in c["correct_by_episode"]})},
        "flags": sum(bool(c["flags"]) for c in convs),
        "words_with_tools": prop(convs, "words_with_tools"),
        "unjudged": sum(c["nat"] is None for c in convs),
    }


COMPARE_KEYS = [("correct", mean), ("invented_any", mean), ("stale", mean), ("nat", mean), ("double", mean),
                ("contra", mean), ("wrong", mean), ("wrong_any", mean), ("ttfw", med), ("ttfw_answered", med),
                ("ttfw_requests", med), ("slow_turns", mean), ("slow_turns_3", mean), ("longest_silence", mean),
                ("t_answer", med), ("redundant_holds", mean), ("cost", mean)]


def compare(arms: dict[str, list[dict]]) -> dict:
    pairs = []
    for s in SPLIT:
        for other in SLOW + FAST:
            pairs.append((other, s))
    pairs += [("relay", "slow_speaks"), ("slow_filler", "slow_speaks"), ("slow_only", "slow_filler")]
    for f in FAST:
        pairs.append(("slow_only", f))
    out = {}
    for a, b in pairs:
        if a in arms and b in arms:
            out[f"{b} - {a}"] = {k: diff(arms[a], arms[b], k, st) for k, st in COMPARE_KEYS}
    return out


def verdict(arms: dict[str, list[dict]], cmp: dict) -> dict:
    """The plan's criteria, applied mechanically to each split design against slow_only."""
    out = {}
    for s in SPLIT:
        if s not in arms or "slow_only" not in arms:
            continue
        c = cmp[f"{s} - slow_only"]
        a_ok = c["ttfw"]["diff"] <= -0.5 and c["slow_turns"]["hi"] < 0
        a_ok_answered_only = c["ttfw_answered"]["diff"] <= -0.5 and c["slow_turns"]["hi"] < 0
        b_ok = c["correct"]["diff"] >= -0.05 and c["invented_any"]["diff"] <= 0.0
        dc = c["double"]["diff"] + c["contra"]["diff"]
        c_ok = c["nat"]["lo"] <= 0 <= c["nat"]["hi"] or c["nat"]["lo"] > 0
        c_ok = c_ok and dc <= 0.5
        ahead = {}
        for f in FAST:
            if f in arms:
                cf = cmp[f"{s} - {f}"]
                ahead[f] = bool(cf["correct"]["lo"] > 0 or cf["invented_any"]["hi"] < 0)
        out[s] = {"a_responsive": a_ok, "b_correct": b_ok, "c_natural": c_ok, "holds": a_ok and b_ok and c_ok,
                  "double_plus_contra_diff": round(dc, 3), "ahead_of_fast": ahead,
                  "a_with_answered_only_median": a_ok_answered_only,
                  "holds_with_answered_only_median": a_ok_answered_only and b_ok and c_ok,
                  "judged_wrong_or_invented_diff": c["wrong_any"]}
    return out


ERA_MODELS = {"openai/gpt-5-mini", "anthropic/claude-opus-4.6", "openai/gpt-5.4-mini", "openai/gpt-5.6-terra"}
ERA_TAGS = ("/feb-", "/summer-", "/smoke-feb", "/smoke-summer")


def cost_group(cond: str) -> str:
    """Where today's-model spend went, for the README."""
    if cond == "judge":
        return "judge"
    for prefix, group in (("cheap-", "cheap tier"), ("top-", "top tier"), ("anthropic-", "mid tier")):
        if cond.startswith(prefix) and "sanity" not in cond:
            return group
    return "set-up (calibration, probes, sanity and smoke runs, D = 4 calls in flight when stopped)"


def costs() -> dict:
    """Spend from both ledgers (the laptop's and the cloud machine's; neither
    is a copy of the other). Calls on the era models of Plan amendment 3 are
    counted apart: they are outside the analysis and outside today's budget."""
    paths = [str(p) for p in [RUNS / "ledger.jsonl", RUNS / "ledger-vm.jsonl"] if p.exists()]
    s = summarise(paths)
    by_tag: dict[str, float] = defaultdict(float)
    era, today = 0.0, 0.0
    groups: dict[str, float] = defaultdict(float)
    for k, d in s.items():
        tag, model = k.split(" | ")[0], k.split(" | ")[-1]
        cond = tag.split("/")[1] if "/" in tag else tag
        c = d.get("cost", 0.0)
        by_tag[cond] += c
        if model in ERA_MODELS or any(x in tag for x in ERA_TAGS):
            era += c
        else:
            today += c
            groups[cost_group(cond)] += c
    return {"total": round(sum(d.get("cost", 0.0) for d in s.values()), 4), "today_models": round(today, 4), "era_models": round(era, 4),
            "today_by_group": {k: round(v, 4) for k, v in sorted(groups.items())},
            "by_condition": {k: round(v, 4) for k, v in sorted(by_tag.items())},
            "by_tag_model": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in d.items()} for k, d in s.items()}}


def fmt_p(d):
    return f"{100 * d['p']:.0f}% [{100 * max(0.0, d['lo']):.0f}, {100 * d['hi']:.0f}] ({d['k']}/{d['n']})" if d["n"] else "-"


KEY_PAIRS = ["relay - slow_only", "slow_speaks - slow_only", "relay - fast_only", "slow_speaks - fast_only",
             "relay - fast_context", "slow_speaks - fast_context", "relay - fast_only_think", "slow_speaks - fast_only_think",
             "slow_speaks - relay", "slow_speaks - slow_filler", "fast_only - slow_only", "fast_context - slow_only"]


def fmt_d(d, scale=1.0, nd=1, unit=""):
    if d is None or d.get("diff") is None or (isinstance(d["diff"], float) and math.isnan(d["diff"])):
        return "-"
    return f"{scale * d['diff']:+.{nd}f}{unit} [{scale * d['lo']:+.{nd}f}, {scale * d['hi']:+.{nd}f}]"


def diff_table(cond: str, cmp: dict) -> str:
    rows = [f"Paired differences, {cond} (second arm subtracted from the first; 95% bootstrap over scripts):", "",
            "| Comparison | Correct (points) | Calls with invented value (points) | Stale (points) | Naturalness | Double replies / call | First word, median (s) | Turns > 2 s (points) | Turns > 3 s (points) | Longest silence (s) | Answer time (s) | Cost / call ($) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in KEY_PAIRS:
        if k not in cmp:
            continue
        c = cmp[k]
        rows.append(f"| {k} | {fmt_d(c['correct'], 100)} | {fmt_d(c['invented_any'], 100)} | {fmt_d(c['stale'], 100)} | {fmt_d(c['nat'], 1, 2)} | "
                    f"{fmt_d(c['double'], 1, 2)} | {fmt_d(c['ttfw'], 1, 2)} | {fmt_d(c['slow_turns'], 100)} | {fmt_d(c['slow_turns_3'], 100)} | "
                    f"{fmt_d(c['longest_silence'], 1, 1)} | {fmt_d(c['t_answer'], 1, 1)} | {fmt_d(c['cost'], 1, 4)} |")
    return "\n".join(rows)


def fmt_m(d, nd=2):
    return f"{d['mean']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]" if d.get("n") else "-"


def table(cond: str, summ: dict) -> str:
    rows = [f"### {cond}", "",
            "| Arm | Correct answers | Hard questions | Invented (calls with any) | Stale | Naturalness (1-5) | Double replies / call | Contradictions / call | First word, median (p90) | Turns waiting > 2 s | Longest silence | Answer time, median | Cost / call |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ORDER:
        if arm not in summ:
            continue
        a = summ[arm]
        rows.append(f"| `{arm}` | {fmt_p(a['correct'])} | {fmt_p(a['correct_hard'])} | {fmt_p(a['invented_any'])} | {fmt_p(a['stale'])} | "
                    f"{fmt_m(a['naturalness'])} | {fmt_m(a['judge_double_per_call'])} | {fmt_m(a['judge_contra_per_call'])} | "
                    f"{a['ttfw_median']:.2f} s ({a['ttfw_p90']:.2f}) | {fmt_p(a['slow_turns'])} | {fmt_m(a['longest_silence'], 1)} s | "
                    f"{fmt_m(a['t_answer_median'], 1)} s | ${a['cost_per_call']['mean']:.4f} |")
    return "\n".join(rows)


PAIR_LABELS = [("cheap-anthropic-realistic-d0", "cheap: Haiku 5.5 + Haiku 5.5"),
               ("anthropic-realistic-d0", "mid: Haiku 5.5 + Sonnet 5.5"),
               ("top-anthropic-realistic-d0", "top: Haiku 5.5 + Opus 5.5 (high)"),
               ("cheap-openai-realistic-d0", "cheap (OpenAI): GPT-6 Luna + GPT-6 Luna")]


def cross_pair_table(results: dict) -> str:
    rows = ["### As the slow model gets stronger (realistic size, measured timing)", "",
            "| Pair | Arm | Correct answers | Calls with invented value | Stale | Naturalness | Double replies / call | First word on requests, median | Requests waiting > 2 s | Answer time, median | Cost / call |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cond, label in PAIR_LABELS:
        c = results["conditions"].get(cond)
        if not c:
            continue
        for arm in ["relay", "slow_speaks", "slow_only", "slow_filler", "fast_only", "fast_context", "fast_only_think", "fast_context_think"]:
            if cond.startswith(("cheap-anthropic", "top-anthropic")) and arm in FAST:
                continue  # shown once, with the mid tier
            a = c["arms"].get(arm)
            if not a:
                continue
            rows.append(f"| {label} | `{arm}` | {fmt_p(a['correct'])} | {fmt_p(a['invented_any'])} | {fmt_p(a['stale'])} | "
                        f"{fmt_m(a['naturalness'])} | {fmt_m(a['judge_double_per_call'])} | {fmt_m(a['ttfw_requests_median'])} s | "
                        f"{fmt_p(a['slow_requests'])} | {fmt_m(a['t_answer_median'], 1)} s | ${a['cost_per_call']['mean']:.3f} |")
    return "\n".join(rows)


def _clean(o):
    """NaN is not JSON: write null."""
    if isinstance(o, float):
        return None if math.isnan(o) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="")
    a = ap.parse_args()
    conds = a.conditions.split(",") if a.conditions else [c for c in MAIN if (RUNS / f"{c}.jsonl").exists()]
    results = {"conditions": {}, "cost": costs()}
    for cond in conds:
        recs = load(cond)
        if cond.startswith(("cheap-anthropic-", "top-anthropic-")):
            # their single-fast arms are the Haiku runs made with the mid tier (the same model and prompts);
            # in the cheap tier the single slow model is the same Haiku, so slow_only is fast_only
            base = "anthropic-" + cond.split("-anthropic-", 1)[1]
            if (RUNS / f"{base}.jsonl").exists():
                extra = [r for r in load(base) if r["arm"] in FAST]
                if cond.startswith("cheap-"):
                    extra += [{**r, "arm": "slow_only"} for r in extra if r["arm"] == "fast_only"]
                recs += extra
        if cond.startswith("cheap-openai-"):
            recs += [{**r, "arm": "slow_only"} for r in recs if r["arm"] == "fast_only"]
        arms: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            arms[r["arm"]].append(conv_measures(r))
        summ = {arm: summarise_arm(cs) for arm, cs in arms.items()}
        cmp = compare(arms)
        results["conditions"][cond] = {"arms": summ, "compare": cmp, "verdict": verdict(arms, cmp)}
        print(table(cond, summ))
        print()
        print(diff_table(cond, cmp))
        print()
        for s, v in results["conditions"][cond]["verdict"].items():
            print(f"  verdict {s}: {v}")
        print()
    jv = RUNS / "judge_validation" / "comparison.json"
    if jv.exists():
        v = json.loads(jv.read_text())
        results["judge_validation"] = {k: v[k] for k in ("n", "naturalness", "events", "by_arm")}
    print(cross_pair_table(results))
    (HERE / "results.json").write_text(json.dumps(_clean(results), indent=1))
    print(f"total cost ${results['cost']['total']:.2f}")


if __name__ == "__main__":
    main()
