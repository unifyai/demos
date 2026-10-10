"""Every number in results.json and the README's tables.

    uv run python analyze.py                 # the counted batches listed below
    uv run python analyze.py --batches a,b   # any batches

Per run: checks.evaluate (the sites' state and log) plus the judge's verdicts
(runs/<batch>/judge.jsonl). Per model, size and arm: proportions with Wilson
intervals, medians and 90th percentiles of times. Differences between
`handle` and each baseline: bootstrap over items (instances), paired, from
demos_common.stats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))

from demos_common.stats import bootstrap_diff, quantile, wilson  # noqa: E402

from separate_loops.checks import evaluate  # noqa: E402

ARMS = ["flat", "blocking", "background", "handle"]
# Amendment 2: the cheap tier on both families at every size; the mid and top tiers at the headline sizes.
COUNTED = ["small-haiku", "medium-haiku", "realistic-haiku", "dual-haiku",
           "small-luna", "medium-luna", "realistic-luna", "dual-luna",
           "realistic-sonnet", "dual-sonnet", "realistic-opus"]
TIER = {"anthropic/claude-haiku-5.5": "cheap", "openai/gpt-6-luna": "cheap", "anthropic/claude-sonnet-5.5": "mid",
        "anthropic/claude-opus-5.5": "top"}
QUESTION_KINDS = ("status", "status2", "status_both", "unrelated")
STEER_KINDS = ("correction", "addition", "change")


def load(batch: str) -> tuple[list[dict], dict[str, dict]]:
    d = HERE / "runs" / batch
    recs = [json.loads(x) for x in (d / "records.jsonl").read_text().splitlines() if x.strip()]
    jp = d / "judge.jsonl"
    judge = {}
    if jp.exists():
        judge = {j["run_id"]: j for j in (json.loads(x) for x in jp.read_text().splitlines() if x.strip())}
    return recs, judge


def instances(condition: str) -> dict[str, dict]:
    return {i["id"]: i for i in json.loads((HERE / "data" / f"{condition}.json").read_text())}


def run_measures(rec: dict, inst: dict, j: dict | None) -> dict:
    ev = evaluate(rec, inst)
    end = rec["end"]["t"]
    qs = []
    for m in ev["messages"]:
        if m["kind"] not in QUESTION_KINDS or not m["while_working"]:
            continue
        if m["kind"] == "unrelated":
            a = m.get("answer_s_auto")
        else:
            jq = (j or {}).get("questions", {}).get(str(m["id"])) if j else None
            a = jq.get("t_first_accurate") if jq else None
        qs.append({"kind": m["kind"], "answer_s": a, "censored_s": round(end - m["t"], 2), "reply_s": m["reply_s"],
                   "judged": m["kind"] == "unrelated" or bool(j)})
    qs = [q for q in qs if q["judged"]]  # a progress question counts only once the judge has read it
    ev["questions"] = qs
    ev["answer_times"] = [q["answer_s"] if q["answer_s"] is not None else q["censored_s"] for q in qs]
    ev["unanswered"] = sum(q["answer_s"] is None for q in qs)
    steers = [m for m in ev["messages"] if m["kind"] in STEER_KINDS]
    ev["reach_times"] = [m["reach_s"] for m in steers if m.get("reach_s") is not None]
    ev["steers_unreached"] = sum(m.get("reach_s") is None for m in steers)
    ev["reply_times"] = [m["reply_s"] for m in ev["messages"] if m["reply_s"] is not None and m["while_working"]
                         and m["kind"] != "answer"]
    out = (j or {}).get("outcome") or {}
    ev["reported_judge"] = bool(out.get("reported")) if j else None
    ev["reported_accurate"] = bool(out.get("reported") and out.get("accurate")) if j else None
    ev["stalled"] = (not ev["complete"]) and ev["end_reason"] == "idle"
    return ev


def _prop(xs: list[bool]) -> dict:
    k, n = sum(bool(x) for x in xs), len(xs)
    lo, hi = wilson(k, n) if n else (math.nan, math.nan)
    return {"k": k, "n": n, "p": round(k / n, 3) if n else None, "lo": round(lo, 3), "hi": round(hi, 3)}


def _times(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "median": None, "p90": None}
    return {"n": len(xs), "median": round(quantile(xs, 0.5), 2), "p90": round(quantile(xs, 0.9), 2),
            "within_5s": round(sum(x <= 5 for x in xs) / len(xs), 3), "within_10s": round(sum(x <= 10 for x in xs) / len(xs), 3),
            "within_30s": round(sum(x <= 30 for x in xs) / len(xs), 3)}


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def summarise(rows: list[dict]) -> dict:
    qs = [q for r in rows for q in r["questions"]]
    s = {
        "runs": len(rows),
        "correct": _prop([r["correct"] for r in rows]),
        "checks_passed_mean": _mean([r["checks_passed"] for r in rows]),
        "complete": _prop([r["complete"] for r in rows]),
        "stalled": _prop([r["stalled"] for r in rows]),
        "crash_or_cap": _prop([r["end_reason"] != "idle" for r in rows]),
        "reported_judge": _prop([r["reported_judge"] for r in rows if r["reported_judge"] is not None]),
        "reported_accurate": _prop([r["reported_accurate"] for r in rows if r["reported_accurate"] is not None]),
        "reported_auto": _prop([r["reported_auto"] for r in rows]),
        "answer_s": _times([x for r in rows for x in r["answer_times"]]),
        "questions": len(qs),
        "unanswered": sum(q["answer_s"] is None for q in qs),
        "reply_s": _times([x for r in rows for x in r["reply_times"]]),
        "reach_s": _times([x for r in rows for x in r["reach_times"]]),
        "steers_unreached": sum(r["steers_unreached"] for r in rows),
        "stale_per_run": _mean([r["stale_total"] for r in rows]),
        "stale_by_kind": {k: _mean([r["stale"].get(k) for r in rows if k in r["stale"]]) for k in STEER_KINDS},
        "pause_steps_mean": _mean([r.get("pause_steps") for r in rows]),
        "cancel_side_effects_mean": _mean([r.get("cancel_side_effects") for r in rows]),
        "cancel_steps_mean": _mean([r.get("cancel_steps") for r in rows]),
        "gap_fact_given": _prop([r["gap"]["fact_given"] for r in rows if r["gap"]]),
        "gap_guessed": _prop([r["gap"]["guessed"] for r in rows if r["gap"]]),
        "gap_wrong_reached_site": _prop([r["gap"]["wrong_reached_site"] for r in rows if r["gap"]]),
        "cost_per_run": _mean([r["cost"] for r in rows]),
        "cost_total": round(sum(r["cost"] for r in rows), 4),
        "person_cost_total": round(sum(r["person_cost"] for r in rows), 4),
        "calls_per_run": _mean([r["calls"] for r in rows]),
        "prompt_tokens_per_run": _mean([r["prompt_tokens"] for r in rows]),
        "actions_per_run": _mean([r["actions"] for r in rows]),
        "work_done_s_median": _times([r["t_last_action"] for r in rows])["median"],
        "last_message_s_median": _times([r["t_last_message"] for r in rows])["median"],
        "queue_overhead_max": max((r["overhead_max"] for r in rows), default=None),
    }
    return s


def _median(xs):
    return quantile(xs, 0.5) if xs else math.nan


def compare(rows: list[dict], base: str) -> dict:
    """handle minus `base`, paired by instance (bootstrap over instances)."""
    by = {a: {r["instance"]: r for r in rows if r["arm"] == a} for a in (base, "handle")}

    def m(key, arm):
        return {i: key(r) for i, r in by[arm].items()}
    out = {}
    out["correct"] = bootstrap_diff(m(lambda r: [float(r["correct"])], base), m(lambda r: [float(r["correct"])], "handle"))
    out["stale_per_run"] = bootstrap_diff(m(lambda r: [float(r["stale_total"])], base), m(lambda r: [float(r["stale_total"])], "handle"))
    a = m(lambda r: r["answer_times"], base)
    b = m(lambda r: r["answer_times"], "handle")
    a = {k: v for k, v in a.items() if v}
    b = {k: v for k, v in b.items() if v}
    if a and b:
        out["answer_s_median"] = bootstrap_diff(a, b, stat=_median)
    out["cost_per_run"] = bootstrap_diff(m(lambda r: [r["cost"]], base), m(lambda r: [r["cost"]], "handle"))
    if any(r["reported_judge"] is not None for r in rows):
        out["reported_accurate"] = bootstrap_diff(m(lambda r: [float(bool(r["reported_accurate"]))], base),
                                                  m(lambda r: [float(bool(r["reported_accurate"]))], "handle"))
    return {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default=",".join(COUNTED))
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()
    results = {"batches": {}, "per_run": []}
    for batch in args.batches.split(","):
        if not (HERE / "runs" / batch / "records.jsonl").exists():
            print(f"(no records for {batch})")
            continue
        recs, judge = load(batch)
        insts = instances(recs[0]["condition"])
        rows = [run_measures(r, insts[r["instance"]], judge.get(r["run_id"])) for r in recs]
        res = {"model": recs[0]["model"], "condition": recs[0]["condition"], "judged": len(judge),
               "arms": {a: summarise([r for r in rows if r["arm"] == a]) for a in ARMS if any(r["arm"] == a for r in rows)},
               "by_family": {f: {a: summarise([r for r in rows if r["arm"] == a and r["family"] == f]) for a in ARMS}
                             for f in "ABC"},
               "handle_minus": {b: compare(rows, b) for b in ARMS if b != "handle" and any(r["arm"] == b for r in rows)}}
        results["batches"][batch] = res
        results["per_run"].extend({k: v for k, v in r.items() if k not in ("messages",)} for r in rows)
        print_tables(batch, res)
    print_sweep(results)
    print_tiers(results)
    results["criteria"] = {b: criteria(res) for b, res in results["batches"].items() if res["condition"] in ("realistic", "dual")}
    print_criteria(results)
    results["sanity"] = sanity()
    results["cost"] = costs()
    print_costs(results)
    Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwrote {args.out}")


def criteria(res: dict) -> dict:
    """The Plan's three tests of "the idea wins", fixed before any run, for one model at one size."""
    hm, arms = res["handle_minus"], res["arms"]
    out: dict = {}
    d = hm.get("flat", {}).get("answer_s_median")
    out["1_responsive"] = {"handle_median_s": arms["handle"]["answer_s"]["median"], "flat_median_s": arms["flat"]["answer_s"]["median"],
                           "diff": d, "holds": bool(d and d["hi"] < 0)}
    st = {b: hm[b]["stale_per_run"] for b in ("flat", "background", "blocking") if b in hm}
    vs_flat = st["flat"]["lo"] <= 0
    fewer = {b: st[b]["diff"] < 0 for b in ("background", "blocking") if b in st}
    out["2_steers"] = {"stale_per_run": {a: arms[a]["stale_per_run"] for a in arms}, "handle_minus": st,
                       "no_more_than_flat": vs_flat, "fewer_than": fewer, "holds": bool(vs_flat and all(fewer.values()))}
    best = max((b for b in arms if b != "handle"), key=lambda b: arms[b]["correct"]["p"] or 0)
    # "reported" as the Plan's Measures define it: the last messages state the outcome accurately (judged).
    rep = arms["handle"]["reported_accurate"]
    within = (arms["handle"]["correct"]["p"] or 0) >= (arms[best]["correct"]["p"] or 0) - 0.10
    out["3_no_worse"] = {"handle_correct": arms["handle"]["correct"], "best_baseline": best, "best_correct": arms[best]["correct"],
                         "within_10_points": within, "handle_reported": rep, "handle_reported_any": arms["handle"]["reported_judge"],
                         "reported_by_arm": {a: arms[a]["reported_accurate"] for a in arms},
                         "reported_90": None if not rep["n"] else rep["p"] >= 0.9,
                         "holds": bool(within and rep["n"] and rep["p"] >= 0.9)}
    c1, c2, c3 = (out[k]["holds"] for k in ("1_responsive", "2_steers", "3_no_worse"))
    out["verdict"] = ("holds" if c1 and c2 and c3 else "trade-off (1 without 2 or 3)" if c1 else "does not pull ahead (not 1)")
    return out


def print_criteria(results: dict) -> None:
    print("\n### The Plan's criteria\n")
    print("| batch | 1 responsive: handle vs flat accurate-answer median, s (diff, 95% CI) | 2 stale steps per run: flat / blocking / background / handle | 3 done right: handle vs best baseline; outcome reported | verdict |")
    print("|---|---|---|---|---|")
    for b, c in results["criteria"].items():
        r = c["1_responsive"]
        d = r["diff"]
        c1 = f"{r['handle_median_s']} vs {r['flat_median_s']} ({d['diff']:+.1f}, {d['lo']:+.1f} to {d['hi']:+.1f})" if d else "-"
        s = c["2_steers"]["stale_per_run"]
        c2 = " / ".join(str(s.get(a)) for a in ARMS) + f" ({'holds' if c['2_steers']['holds'] else 'fails'})"
        t = c["3_no_worse"]
        c3 = (f"{_p(t['handle_correct'])} vs {t['best_baseline']} {_p(t['best_correct'])}; {_p(t['handle_reported'])}"
              f" ({'holds' if t['holds'] else 'fails'})")
        print(f"| {b} | {c1} ({'holds' if r['holds'] else 'fails'}) | {c2} | {c3} | {c['verdict']} |")


def sanity() -> dict:
    """The sanity condition on every model: every arm should get every task right."""
    out = {}
    for batch in ("sanity-haiku", "sanity-luna", "sanity-sonnet", "sanity-opus"):
        if not (HERE / "runs" / batch / "records.jsonl").exists():
            continue
        recs, _ = load(batch)
        insts = instances("sanity")
        rows = [evaluate(r, insts[r["instance"]]) for r in recs]
        out[batch] = {"model": recs[0]["model"],
                      **{a: f"{sum(r['correct'] for r in rows if r['arm'] == a)}/{sum(r['arm'] == a for r in rows)}" for a in ARMS}}
    return out


ERA_MODELS = {"openai/o4-mini", "anthropic/claude-sonnet-4"}  # Amendment 1's June 2025 models: kept apart, not counted


def costs() -> dict:
    """Spend from both ledgers (the laptop's and the cloud machine's; neither is a copy of the other), by model and by
    what the call was for. The older models of Amendment 1 are reported apart and do not count against the cap."""
    by_model: dict = {}
    by_use: dict = {}
    for f in ("runs/ledger.jsonl", "runs/ledger-vm.jsonl"):
        p = HERE / f
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            c = r.get("cost") or 0.0
            m = r.get("meta") or {}
            by_model[r["model"]] = by_model.get(r["model"], 0.0) + c
            use = "judge" if m.get("phase") == "judge" else "person" if m.get("loop") == "person" else "agent" if m.get("run") else "probes and checks"
            if r["model"] in ERA_MODELS:
                use = "older models (Amendment 1)"
            by_use[use] = by_use.get(use, 0.0) + c
    era = sum(v for k, v in by_model.items() if k in ERA_MODELS)
    total = sum(by_model.values())
    return {"by_model": {k: round(v, 4) for k, v in sorted(by_model.items())}, "by_use": {k: round(v, 4) for k, v in sorted(by_use.items())},
            "older_models": round(era, 4), "todays_models": round(total - era, 4), "total": round(total, 4)}


def print_costs(results: dict) -> None:
    c = results["cost"]
    print("\n### Cost from the ledgers\n")
    for k, v in c["by_model"].items():
        print(f"- {k}: ${v:.2f}")
    for k, v in c["by_use"].items():
        print(f"- {k}: ${v:.2f}")
    print(f"- today's models: ${c['todays_models']:.2f}; older models (not counted): ${c['older_models']:.2f}; all: ${c['total']:.2f}")
    print("\nsanity:", json.dumps(results["sanity"]))


def print_tiers(results: dict) -> None:
    """The headline comparison at each tier: how the advantage changes as models get stronger."""
    for cond in ("realistic", "dual"):
        bs = [b for b in results["batches"].values() if b["condition"] == cond]
        if not bs:
            continue
        order = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5.5"]
        bs.sort(key=lambda b: order.index(b["model"]) if b["model"] in order else 9)
        print(f"\n### Tiers, {cond}\n")
        print("| tier, model | arm | done right | accurate answer, s (median / p90) | within 10 s | stale steps per run | outcome reported | cost per run, $ |")
        print("|---|---|---|---|---|---|---|---|")
        for b in bs:
            for a, st in b["arms"].items():
                print(f"| {TIER.get(b['model'], '?')}, {b['model'].split('/')[1]} | {a} | {_p(st['correct'])} | {_t(st['answer_s'])} | "
                      f"{st['answer_s'].get('within_10s', '-')} | {st['stale_per_run']} | {_p(st['reported_accurate'])} | "
                      f"{st['cost_per_run']:.4f} |")
        print("\nhandle minus flat, by tier (paired bootstrap over items):\n")
        print("| tier, model | done right | accurate answer, median s | stale steps per run | cost per run, $ |")
        print("|---|---|---|---|---|")
        for b in bs:
            c = b["handle_minus"].get("flat", {})
            def f(k, d=2):
                return f"{c[k]['diff']:+.{d}f} ({c[k]['lo']:+.{d}f} to {c[k]['hi']:+.{d}f})" if k in c else "-"
            print(f"| {TIER.get(b['model'], '?')}, {b['model'].split('/')[1]} | {f('correct')} | {f('answer_s_median')} | "
                  f"{f('stale_per_run')} | {f('cost_per_run', 4)} |")


def print_sweep(results: dict) -> None:
    """One line per size and arm, for each model: where the idea pulls ahead, or not."""
    order = ["small", "medium", "realistic", "dual"]
    for model in sorted({b["model"] for b in results["batches"].values()}):
        bs = sorted((b for b in results["batches"].values() if b["model"] == model), key=lambda b: order.index(b["condition"]))
        print(f"\n### Sweep, {model}\n")
        print("| size | arm | done right | accurate answer, s (median / p90) | within 10 s | stale steps per run | cost per run, $ | work finished, s |")
        print("|---|---|---|---|---|---|---|---|")
        for b in bs:
            for a, st in b["arms"].items():
                print(f"| {b['condition']} | {a} | {_p(st['correct'])} | {_t(st['answer_s'])} | {st['answer_s'].get('within_10s', '-')} | "
                      f"{st['stale_per_run']} | {st['cost_per_run']:.4f} | {st['work_done_s_median']} |")


def _p(x: dict) -> str:
    return f"{x['k']}/{x['n']} ({x['lo']:.2f}-{x['hi']:.2f})" if x["n"] else "-"


def _t(x: dict) -> str:
    return f"{x['median']} / {x['p90']}" if x.get("n") else "-"


def print_tables(batch: str, res: dict) -> None:
    print(f"\n### {batch}: {res['condition']}, {res['model']} (judged {res['judged']} runs)\n")
    arms = list(res["arms"])
    print("| measure | " + " | ".join(arms) + " |")
    print("|---|" + "---|" * len(arms))
    rows = [
        ("done right (all checks)", lambda s: _p(s["correct"])),
        ("checks passed, mean", lambda s: f"{s['checks_passed_mean']:.2f}"),
        ("stalled unfinished", lambda s: _p(s["stalled"])),
        ("crash or cap", lambda s: _p(s["crash_or_cap"])),
        ("questions while working (n)", lambda s: str(s["questions"])),
        ("accurate answer, s: median / p90", lambda s: _t(s["answer_s"])),
        ("answered within 10 s", lambda s: f"{s['answer_s'].get('within_10s', '-')}"),
        ("never answered", lambda s: str(s["unanswered"])),
        ("first reply, s: median / p90", lambda s: _t(s["reply_s"])),
        ("correction reached the work, s: median / p90", lambda s: _t(s["reach_s"])),
        ("stale steps per run", lambda s: f"{s['stale_per_run']}"),
        ("steps during a pause, mean", lambda s: f"{s['pause_steps_mean']}"),
        ("side effects after cancel, mean", lambda s: f"{s['cancel_side_effects_mean']}"),
        ("asked for the missing fact", lambda s: _p(s["gap_fact_given"])),
        ("guessed the missing fact", lambda s: _p(s["gap_guessed"])),
        ("outcome reported accurately (judge)", lambda s: _p(s["reported_accurate"])),
        ("outcome facts in last messages (auto)", lambda s: _p(s["reported_auto"])),
        ("work finished, s (median)", lambda s: f"{s['work_done_s_median']}"),
        ("model calls per run", lambda s: f"{s['calls_per_run']}"),
        ("cost per run, $", lambda s: f"{s['cost_per_run']:.4f}" if s["cost_per_run"] is not None else "-"),
    ]
    for name, f in rows:
        print(f"| {name} | " + " | ".join(f(res["arms"][a]) for a in arms) + " |")
    print("\nhandle minus baseline (paired bootstrap over items, 95% interval):\n")
    print("| baseline | done right | accurate answer, median s | stale steps per run | outcome reported | cost per run, $ |")
    print("|---|---|---|---|---|---|")
    for b, c in res["handle_minus"].items():
        def f(k):
            return f"{c[k]['diff']:+.2f} ({c[k]['lo']:+.2f} to {c[k]['hi']:+.2f})" if k in c else "-"
        def fc(k):
            return f"{c[k]['diff']:+.4f} ({c[k]['lo']:+.4f} to {c[k]['hi']:+.4f})" if k in c else "-"
        print(f"| {b} | {f('correct')} | {f('answer_s_median')} | {f('stale_per_run')} | {f('reported_accurate')} | {fc('cost_per_run')} |")


if __name__ == "__main__":
    main()
