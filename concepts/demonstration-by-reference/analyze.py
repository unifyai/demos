"""Summarise the runs: accuracy, cost, latency and tokens per arm and size, the paired
comparisons against by_reference, and the pre-registered verdict.

  uv run python analyze.py                       # runs/records.jsonl -> results.json, tables on stdout
  uv run python analyze.py --records runs/sanity-records.jsonl --out runs/sanity-results.json

Intervals: Wilson for proportions; a bootstrap over items (each item's seeds,
and for pooled sizes each item's sizes, kept together) for means, differences
and ratios.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

from demos_common.ledger import summarise
from demos_common.stats import bootstrap_diff, fisher_exact, mean, quantile, wilson

HERE = Path(__file__).parent
ARMS = ["inline_all", "inline_low", "inline_sampled", "captions", "pipeline", "by_reference"]
REALISTIC = (256, 500)
MARGIN = 0.10


def load_records(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    ok, bad = [], []
    for p in paths:
        text = gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()
        for line in text.splitlines():
            if line.strip():
                r = json.loads(line)
                (bad if r.get("error") else ok).append(r)
    # one record per (model, session, arm, seed): a re-run after an error replaces nothing that
    # succeeded, but keep the first success if a run was somehow recorded twice
    seen, uniq = set(), []
    for r in ok:
        k = (r["model"], r["session"], r["arm"], r["seed"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq, bad


_SESSIONS: dict[str, dict] = {}


def session_json(session: str) -> dict:
    if session not in _SESSIONS:
        _SESSIONS[session] = json.loads((HERE / "data" / "sessions" / f"{session}.json").read_text())
    return _SESSIONS[session]


def evidence_of(session: str) -> list[str]:
    return session_json(session)["evidence"]


def set_aside_of(session: str) -> set[str]:
    """Tasks the person showed and set aside out loud (not this one, already done, week after)."""
    try:
        meta = session_json(session)["meta"]
    except FileNotFoundError:
        return set()
    return {t for t in [meta.get("done"), meta.get("later"), *meta.get("not_this", [])] if t} | (
        {meta["correction"]["old"]} if meta.get("correction") else set())


def error_kind(truth: list[str], answer: list[str] | None) -> str:
    """How a wrong answer is wrong: none (no answer), order (right tasks, wrong order), or the
    mix of missing and extra tasks."""
    if answer is None:
        return "no answer"
    if answer == truth:
        return "right"
    missing, extra = set(truth) - set(answer), set(answer) - set(truth)
    if not missing and not extra:
        return "order"
    if missing and extra:
        return "swapped"
    return "missing" if missing else "extra"


def derive(r: dict) -> dict:
    calls = r["calls"]
    pre = [c for c in calls if c["phase"] == "pre"]
    req = [c for c in calls if c["phase"] == "request"]
    truth, ans = r["truth"], r["answer"] or []
    looked = set(r.get("looked") or [])
    ev = evidence_of(r["session"]) if r["arm"] == "by_reference" else []
    return {
        "model": r["model"], "arm": r["arm"], "session": r["session"], "item": r["item"], "size": r["size"],
        "seed": r["seed"],
        "correct": float(bool(r["correct"])),
        "set_correct": float(sorted(ans) == sorted(truth)),
        "positions": sum(1 for i, t in enumerate(truth) if i < len(ans) and ans[i] == t) / len(truth),
        "answered": float(r["answer"] is not None),
        "cost": sum(c["cost"] for c in calls),
        "cost_pre": sum(c["cost"] for c in pre),
        "latency": sum(c["latency_s"] for c in req),
        "pre_compute_s": sum(c["latency_s"] for c in pre),
        "prompt_per_step": mean([c["prompt_tokens"] for c in req]),
        "prompt_request": sum(c["prompt_tokens"] for c in req),
        "prompt_all": sum(c["prompt_tokens"] for c in calls),
        "steps": r["steps"],
        "calls": len(calls),
        "looked": len(looked),
        "looked_share": len(looked) / r["size"],
        "looked_evidence": float(all(e in looked for e in ev)) if ev else float("nan"),
        "providers": sorted({c.get("provider") or "?" for c in calls}),
        "refused": bool(r.get("limit")),
        "error_kind": "over the limit" if r.get("limit") else error_kind(truth, r["answer"]),
        "extra_set_aside": len((set(ans) - set(truth)) & set_aside_of(r["session"])),
        "extra_other": len((set(ans) - set(truth)) - set_aside_of(r["session"])),
    }


def by_item(rows: list[dict], key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        out[str(r["item"])].append(r[key])
    return dict(out)


def boot_mean(rows: list[dict], key: str, iters: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    m = by_item(rows, key)
    ids = sorted(m)
    rng = random.Random(seed)
    point = mean([v for i in ids for v in m[i]])
    stats = []
    for _ in range(iters):
        pick = [rng.choice(ids) for _ in ids]
        stats.append(mean([v for i in pick for v in m[i]]))
    return point, quantile(stats, 0.025), quantile(stats, 0.975)


def boot_ratio(a: list[dict], b: list[dict], key: str, iters: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """mean(b)/mean(a) over the items both have, resampling items."""
    ma, mb = by_item(a, key), by_item(b, key)
    ids = sorted(set(ma) & set(mb))
    rng = random.Random(seed)

    def ratio(pick: list[str]) -> float:
        return mean([v for i in pick for v in mb[i]]) / mean([v for i in pick for v in ma[i]])

    stats = [ratio([rng.choice(ids) for _ in ids]) for _ in range(iters)]
    return ratio(ids), quantile(stats, 0.025), quantile(stats, 0.975)


def cell(rows: list[dict]) -> dict:
    """Accuracy over every run; cost, latency and tokens over the requests the model accepted
    (a request refused as over the model's limits counts as wrong and costs nothing)."""
    k, n = int(sum(r["correct"] for r in rows)), len(rows)
    lo, hi = wilson(k, n)
    accepted = [r for r in rows if not r["refused"]]
    out = {"n": n, "correct": k, "acc": k / n if n else float("nan"), "acc_lo": lo, "acc_hi": hi,
           "set_acc": mean([r["set_correct"] for r in rows]), "positions": mean([r["positions"] for r in rows]),
           "answered": mean([r["answered"] for r in rows]), "refused": n - len(accepted)}
    kinds: dict[str, int] = defaultdict(int)
    for r in rows:
        kinds[r["error_kind"]] += 1
    out["error_kinds"] = dict(kinds)
    if not accepted:
        out["fits"] = False
        return out
    out["fits"] = True
    rows = accepted
    for key in ("cost", "latency"):
        p, l, h = boot_mean(rows, key)
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = p, l, h
    lat = [r["latency"] for r in rows]
    out["latency_median"] = quantile(lat, 0.5)
    out["latency_p90"] = quantile(lat, 0.9)
    for key in ("cost_pre", "pre_compute_s", "prompt_per_step", "prompt_request", "prompt_all", "steps", "calls",
                "looked", "looked_share"):
        out[key] = mean([r[key] for r in rows])
    ev = [r for r in rows if r["looked_evidence"] == r["looked_evidence"]]
    out["looked_evidence"] = mean([r["looked_evidence"] for r in ev]) if ev else None
    if ev:  # by_reference: did it look at every frame that shows a pick, and was it right either way
        saw = [r for r in ev if r["looked_evidence"] == 1.0]
        missed = [r for r in ev if r["looked_evidence"] == 0.0]
        out["looked_all_evidence"] = {"runs": len(saw), "correct": int(sum(r["correct"] for r in saw))}
        out["missed_some_evidence"] = {"runs": len(missed), "correct": int(sum(r["correct"] for r in missed))}
    out["providers"] = sorted({p for r in rows for p in r["providers"]})
    out["error_kinds"] = dict(out.get("error_kinds", {}))
    out["extra_set_aside"] = sum(r["extra_set_aside"] for r in rows)
    out["extra_other"] = sum(r["extra_other"] for r in rows)
    return out


def compare(ref: list[dict], other: list[dict]) -> dict:
    """by_reference (ref) minus the other arm, paired over items. Accuracy counts every run
    (a refused request is a failure); cost and latency compare the requests both arms could make,
    and are absent where the other arm never fitted."""
    acc = bootstrap_diff(by_item(other, "correct"), by_item(ref, "correct"))
    k1, n1 = int(sum(r["correct"] for r in ref)), len(ref)
    k2, n2 = int(sum(r["correct"] for r in other)), len(other)
    out = {"acc_diff": acc["diff"], "acc_lo": acc["lo"], "acc_hi": acc["hi"],
           "fisher_p": fisher_exact(k1, n1, k2, n2), "items": acc["items"],
           "other_refused": sum(r["refused"] for r in other)}
    ref_ok = [r for r in ref if not r["refused"]]
    other_ok = [r for r in other if not r["refused"]]
    if not other_ok or not ref_ok:
        out["other_fits"] = bool(other_ok)
        return out
    out["other_fits"] = True
    cost = bootstrap_diff(by_item(other_ok, "cost"), by_item(ref_ok, "cost"))
    lat = bootstrap_diff(by_item(other_ok, "latency"), by_item(ref_ok, "latency"))
    ratio = boot_ratio(other_ok, ref_ok, "cost")
    lat_ratio = boot_ratio(other_ok, ref_ok, "latency")
    out.update({"cost_diff": cost["diff"], "cost_lo": cost["lo"], "cost_hi": cost["hi"],
                "cost_ratio": ratio[0], "cost_ratio_lo": ratio[1], "cost_ratio_hi": ratio[2],
                "latency_diff": lat["diff"], "latency_lo": lat["lo"], "latency_hi": lat["hi"],
                "latency_ratio": lat_ratio[0], "latency_ratio_lo": lat_ratio[1], "latency_ratio_hi": lat_ratio[2]})
    return out


def verdict(cmp: dict[str, dict]) -> dict:
    """The pre-registered rules, applied to by_reference's comparisons with every other arm present."""
    out: dict = {}
    ia = cmp.get("inline_all")
    if ia:
        if ia.get("other_fits"):
            holds = ia["acc_lo"] > -MARGIN and ia["cost_hi"] < 0 and ia["latency_hi"] < 0
        else:  # inlining does not fit this model at this size: only accuracy can be compared
            holds = ia["acc_lo"] > -MARGIN
            out["inline_all_does_not_fit"] = True
        out["holds_vs_inline_all"] = holds
        out["pulls_ahead_of_inline_all"] = holds and ia["acc_lo"] > 0
    noninf_all = all(c["acc_lo"] > -MARGIN for c in cmp.values())
    # an arm is non-inferior to by_reference when by_reference is not ahead of it by the margin
    rivals = [a for a, c in cmp.items() if c["acc_hi"] < MARGIN]
    # "wins outright" is on top of holding against inline_all (the plan: "if it is *also* non-inferior
    # to every other arm and cheaper than every arm that is non-inferior to it")
    out["wins_outright"] = bool(out.get("holds_vs_inline_all")) and noninf_all and all(
        cmp[a].get("cost_hi", -1) < 0 for a in rivals)
    out["more_accurate_than_by_reference"] = [a for a, c in cmp.items() if c["acc_hi"] < 0]
    out["noninferior_cheaper_and_faster"] = [a for a in rivals if cmp[a].get("cost_lo", -1) > 0
                                             and cmp[a].get("latency_lo", -1) > 0]
    out["loses"] = bool(out["more_accurate_than_by_reference"] or out["noninferior_cheaper_and_faster"])
    return out


def fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}"


TIERS = {"anthropic/claude-haiku-5.5": "cheap", "openai/gpt-6-luna": "cheap", "anthropic/claude-sonnet-5.5": "mid",
         "openai/gpt-6.1-sol": "mid", "anthropic/claude-opus-5.5": "top"}


def headline(res: dict) -> str:
    """by_reference against inline_all at every tier and size, side by side."""
    lines = ["\n### Headline: by reference against inlining every frame, by tier\n",
             "| tier | model | frames | by reference: right | inline: right | cost by ref, $ | cost inline, $ | inline / by ref cost | latency by ref, s | latency inline, s | prompt per step by ref | prompt per step inline |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(res["models"], key=lambda m: (["cheap", "mid", "top"].index(TIERS.get(m, "cheap")), m))
    for model in order:
        cells = res["models"][model]["cells"]
        br, ia = cells.get("by_reference", {}), cells.get("inline_all", {})
        for size in sorted({int(x) for x in br} & {int(x) for x in ia}):
            b, i = br[str(size)], ia[str(size)]
            if not (b.get("fits") and i.get("fits")):
                continue
            lines.append(
                f"| {TIERS.get(model, '?')} | {model.split('/')[1]} | {size} | {b['correct']}/{b['n']} | {i['correct']}/{i['n']} "
                f"| {b['cost']:.4f} | {i['cost']:.4f} | {i['cost'] / b['cost']:.0f}x | {b['latency_median']:.1f} | {i['latency_median']:.1f} "
                f"| {b['prompt_per_step']:,.0f} | {i['prompt_per_step']:,.0f} |")
    return "\n".join(lines)


def tables(res: dict) -> str:
    lines = [headline(res)]
    for model, m in res["models"].items():
        lines.append(f"\n### {model}\n")
        lines.append("| frames | arm | correct | 95% CI | right set | cost per request, $ | request latency, s (median) | prompt tokens per step | looked at |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for size in sorted({int(s) for a in m["cells"].values() for s in a}):
            for arm in ARMS:
                c = m["cells"].get(arm, {}).get(str(size))
                if not c:
                    continue
                if not c["fits"]:
                    lines.append(f"| {size} | {arm} | 0/{c['n']}: request refused, does not fit | | | | | | |")
                    continue
                looked = f"{c['looked']:.1f} ({fmt_pct(c['looked_share'])}%)" if arm == "by_reference" else ""
                refused = f" ({c['refused']} refused)" if c["refused"] else ""
                lines.append(
                    f"| {size} | {arm} | {c['correct']}/{c['n']} ({fmt_pct(c['acc'])}%){refused} | {fmt_pct(c['acc_lo'])}–{fmt_pct(c['acc_hi'])} "
                    f"| {fmt_pct(c['set_acc'])}% | {c['cost']:.4f} [{c['cost_lo']:.4f}–{c['cost_hi']:.4f}] "
                    f"| {c['latency_median']:.1f} | {c['prompt_per_step']:,.0f} | {looked} |")
        lines.append("\nby_reference minus each arm (paired over items; accuracy in points, cost as a ratio):\n")
        lines.append("| frames | against | accuracy diff [95% CI] | cost ratio [95% CI] | latency diff, s [95% CI] | Fisher p |")
        lines.append("|---|---|---|---|---|---|")
        for size, cmps in m["comparisons"].items():
            for arm, c in cmps.items():
                acc = f"{100 * c['acc_diff']:+.0f} [{100 * c['acc_lo']:+.0f}, {100 * c['acc_hi']:+.0f}]"
                if c.get("other_fits") and "cost_ratio" in c:
                    cost = f"{c['cost_ratio']:.2f} [{c['cost_ratio_lo']:.2f}, {c['cost_ratio_hi']:.2f}]"
                    lat = f"{c['latency_diff']:+.1f} [{c['latency_lo']:+.1f}, {c['latency_hi']:+.1f}]"
                else:
                    cost = lat = "does not fit"
                lines.append(f"| {size} | {arm} | {acc} | {cost} | {lat} | {c['fisher_p']:.3g} |")
        lines.append("\nHow wrong answers were wrong (runs), and extra tasks that had been set aside out loud vs others:\n")
        lines.append("| frames | arm | right | order only | missing a pick | extra task | swapped | no answer | over the model's limit | extras set aside / other |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for size in sorted({int(s) for a in m["cells"].values() for s in a}):
            for arm in ARMS:
                c = m["cells"].get(arm, {}).get(str(size))
                if not c:
                    continue
                k = c["error_kinds"]
                lines.append(f"| {size} | {arm} | {k.get('right', 0)} | {k.get('order', 0)} | {k.get('missing', 0)} | "
                             f"{k.get('extra', 0)} | {k.get('swapped', 0)} | {k.get('no answer', 0)} | "
                             f"{k.get('over the limit', 0)} | {c['extra_set_aside']} / {c['extra_other']} |")
        lines.append("\nby_reference: whether it looked at every frame showing a pick (or the correction), and accuracy either way:\n")
        lines.append("| frames | frames looked at (mean) | share of session | looked at every pick's frame | right when it did | right when it did not | steps |")
        lines.append("|---|---|---|---|---|---|---|")
        for size in sorted({int(s) for a in m["cells"].values() for s in a}):
            c = m["cells"].get("by_reference", {}).get(str(size))
            if not c or not c.get("looked_all_evidence"):
                continue
            sa, mi = c["looked_all_evidence"], c["missed_some_evidence"]
            lines.append(f"| {size} | {c['looked']:.1f} | {fmt_pct(c['looked_share'])}% | {sa['runs']}/{c['n']} | "
                         f"{sa['correct']}/{sa['runs']} | {mi['correct']}/{mi['runs']} | {c['steps']:.1f} |")
        lines.append("\nVerdict (pre-registered rules):\n")
        for size, v in m["verdict"].items():
            lines.append(f"- {size}: {json.dumps(v)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", type=Path, default=[HERE / "runs" / "records.jsonl"])
    ap.add_argument("--ledgers", nargs="+", type=Path, default=[HERE / "runs" / "ledger.jsonl"])
    ap.add_argument("--out", type=Path, default=HERE / "results.json")
    ap.add_argument("--items", default=None, help="keep only these items, e.g. 0-1")
    a = ap.parse_args()
    recs, errored = load_records(a.records)
    if a.items:
        lo, hi = (int(x) for x in a.items.split("-"))
        recs = [r for r in recs if lo <= r["item"] <= hi]
    rows = [derive(r) for r in recs]
    res: dict = {"records": [str(p.relative_to(HERE)) for p in a.records], "items": a.items or "all", "runs": len(rows),
                 "errored_runs_excluded": len(errored), "models": {}}
    for model in sorted({r["model"] for r in rows}):
        mrows = [r for r in rows if r["model"] == model]
        cells: dict = defaultdict(dict)
        for arm in ARMS:
            for size in sorted({r["size"] for r in mrows}):
                sel = [r for r in mrows if r["arm"] == arm and r["size"] == size]
                if sel:
                    cells[arm][str(size)] = cell(sel)
        comparisons: dict = {}
        verdicts: dict = {}
        groups = {str(s): (s,) for s in sorted({r["size"] for r in mrows})}
        if all(any(r["size"] == s for r in mrows) for s in REALISTIC):
            groups["256+500"] = REALISTIC
        for label, sizes in groups.items():
            ref = [r for r in mrows if r["arm"] == "by_reference" and r["size"] in sizes]
            if not ref:
                continue
            cmps = {}
            for arm in ARMS:
                if arm == "by_reference":
                    continue
                other = [r for r in mrows if r["arm"] == arm and r["size"] in sizes]
                if other:
                    cmps[arm] = compare(ref, other)
            if cmps:
                comparisons[label] = cmps
                verdicts[label] = verdict(cmps)
        pooled = {}
        if "256+500" in groups:
            for arm in ARMS:
                sel = [r for r in mrows if r["arm"] == arm and r["size"] in REALISTIC]
                if sel:
                    pooled[arm] = cell(sel)
        res["models"][model] = {"cells": cells, "pooled_256_500": pooled, "comparisons": comparisons, "verdict": verdicts}
    ledgers = [p for p in a.ledgers if p.exists()]
    if ledgers:
        led = summarise([str(p) for p in ledgers])
        res["ledger"] = {"by_tag_model": led, "total_cost": sum(v.get("cost", 0.0) for v in led.values())}
    a.out.write_text(json.dumps(res, indent=1, default=float))
    print(f"{len(rows)} runs ({len(errored)} errored, excluded) -> {a.out.name}")
    print(tables(res))
    if ledgers:
        print(f"\nledger total ${res['ledger']['total_cost']:.2f}")


if __name__ == "__main__":
    main()
