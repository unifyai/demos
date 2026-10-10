"""Score and summarise runs: every number in the README comes from here.

    uv run python analyze.py --run main [--run main-luna ...] [--out results.json] [--tables]

Tasks: right procedure; for each rule that applies, current / stale / conflict
/ vague / missing / unjudged; extra rules; fully correct; tool calls, tokens,
cost and model seconds. Changes: writes, places written, tool calls, cost,
model seconds; completed (every procedure the rule governs is governed by the
new version alone); the governing texts' classes; unchanged rules left intact.
Differences are linked minus each other arm, bootstrapped over items with the
seeds kept together.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from demos_common.stats import bootstrap_diff, wilson

from linked_libraries.runner import Records
from linked_libraries.scoring import classify, function_found, joined, map_key, match_title, reported_items, score_task, version_key

HERE = Path(__file__).parent
TASK_ARMS = ["pasted", "rule-skills", "rules-file", "long-context", "linked", "linked-in-context"]
CHANGE_ARMS = ["pasted", "rule-skills", "rules-file", "linked"]
CLASSES = ["current", "stale", "conflict", "vague", "missing", "unjudged"]


def datasets() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in (HERE / "data").glob("*.json") if p.stem in ("sanity", "S", "M", "L")}


def prop(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": (k / n) if n else None, "lo": lo, "hi": hi}


def mean(xs):
    return statistics.fmean(xs) if xs else None


def load(run: str):
    rec = Records(HERE / "runs" / run)
    return {"tasks": rec.read("tasks"), "changes": rec.read("changes"), "state": rec.read("state"),
            "texts": {r["hash"]: r["text"] for r in rec.read("texts")},
            "judgments": {r["key"]: r for r in rec.read("judgments")}}


def score_tasks(data, dss) -> list[dict]:
    out = []
    for t in data["tasks"]:
        ds = dss[t["size"]]
        sc = score_task(t, ds, data["judgments"])
        out.append({**{k: t[k] for k in ("model", "size", "arm", "condition", "seed", "task", "procedure", "hidden",
                                         "parsed", "capped", "nudged", "n_tool_calls", "prompt_tokens",
                                         "completion_tokens", "reasoning_tokens", "cost", "model_seconds")},
                    "changed_expected": t["changed_expected"], **sc})
    return out


def summarise_tasks(rows: list[dict]) -> dict:
    n = len(rows)
    pairs = Counter()
    changed_pairs = Counter()
    unchanged_pairs = Counter()
    for r in rows:
        for rid, cls in r["rules"].items():
            pairs[cls] += 1
            (changed_pairs if rid in r["changed_expected"] else unchanged_pairs)[cls] += 1
    npairs = sum(pairs.values())
    hidden = [r for r in rows if r["hidden"]]
    return {
        "tasks": n,
        "function_found": prop(sum(r["function_found"] for r in rows), n),
        "hidden_found": prop(sum(r["function_found"] for r in hidden), len(hidden)),
        "visible_found": prop(sum(r["function_found"] for r in rows if not r["hidden"]), n - len(hidden)),
        "rules_current": prop(pairs["current"], npairs),
        "pairs": {c: pairs[c] for c in CLASSES},
        "changed_pairs": {c: changed_pairs[c] for c in CLASSES},
        "changed_current": prop(changed_pairs["current"], sum(changed_pairs.values())),
        "unchanged_pairs": {c: unchanged_pairs[c] for c in CLASSES},
        "unchanged_reported": prop(unchanged_pairs["current"], sum(unchanged_pairs.values())),
        "fully_correct": prop(sum(r["fully_correct"] for r in rows), n),
        "exact": prop(sum(r["exact"] for r in rows), n),
        "extras_per_task": mean([len(r["extras"]) + r["unknown_items"] for r in rows]),
        "tasks_with_extras": prop(sum(bool(r["extras"]) or r["unknown_items"] > 0 for r in rows), n),
        "tool_calls": mean([r["n_tool_calls"] for r in rows]),
        "prompt_tokens": mean([r["prompt_tokens"] for r in rows]),
        "completion_tokens": mean([r["completion_tokens"] for r in rows]),
        "cost_per_task": mean([r["cost"] for r in rows]),
        "model_seconds_mean": mean([r["model_seconds"] for r in rows]),
        "model_seconds_median": statistics.median([r["model_seconds"] for r in rows]) if rows else None,
        "unparsed": sum(not r["parsed"] for r in rows),
        "capped": sum(r["capped"] for r in rows),
    }


def items(rows: list[dict], what: str) -> dict[str, list[float]]:
    """item id -> values over seeds (and nothing else), for bootstrap_diff."""
    out: dict[str, list[float]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["seed"]):
        if what == "rules_current":
            for rid, cls in r["rules"].items():
                out[f"{r['condition']}|{r['task']}|{rid}"].append(1.0 if cls == "current" else 0.0)
        elif what == "changed_current":
            for rid in r["changed_expected"]:
                out[f"{r['condition']}|{r['task']}|{rid}"].append(1.0 if r["rules"][rid] == "current" else 0.0)
        elif what == "cost_per_task":
            out[f"{r['condition']}|{r['task']}"].append(r["cost"])
        elif what == "hidden_found":
            if r["hidden"]:
                out[f"{r['condition']}|{r['task']}"].append(1.0 if r["function_found"] else 0.0)
        else:
            out[f"{r['condition']}|{r['task']}"].append(1.0 if r[what] else 0.0)
    return out


def change_rows(data, dss) -> list[dict]:
    out = []
    for c in data["changes"]:
        ds = dss[c["size"]]
        places = {w.get("path") or w.get("id") for w in c["writes"]}
        row = {k: c[k] for k in ("model", "size", "arm", "condition", "seed", "change", "order", "n_tool_calls",
                                 "prompt_tokens", "completion_tokens", "cost", "model_seconds", "capped", "parsed")}
        row.update({"writes": len(c["writes"]), "places": len(places), "ops": Counter(w["op"] for w in c["writes"]),
                    "link_writes": sum(1 for w in c["writes"] if w["op"] == "update" and "functions" in w.get("fields", [])),
                    "adds": sum(1 for w in c["writes"] if w["op"] in ("add", "create"))})
        if c["condition"] == "passing":
            ans = c.get("answer") or {}
            row["plan_function_found"] = function_found(ans.get("function"), ds["procedures"], c["passing_procedure"])
            items_ = [i for i in reported_items(ans) if (match_title(i["title"], ds["rules"]) or
                      (data["judgments"].get(map_key(ds["size"], i)) or {}).get("rule")) == c["change"]]
            if not items_:
                row["plan_rule"] = "missing"
            else:
                j = data["judgments"].get(version_key(c["change"], joined(items_)))
                row["plan_rule"] = classify(j["states_new"], j["states_old"]) if j and j.get("states_new") is not None else "unjudged"
        out.append(row)
    return out


def state_rows(data) -> list[dict]:
    out = []
    for s in data["state"]:
        row = {k: s[k] for k in ("model", "size", "arm", "condition", "seed", "rule", "procedure", "changed", "original_intact")}
        if s["changed"]:
            j = data["judgments"].get(version_key(s["rule"], data["texts"][s["hash"]]))
            row["class"] = classify(j["states_new"], j["states_old"]) if j and j.get("states_new") is not None else "unjudged"
        out.append(row)
    return out


def summarise_changes(rows: list[dict], srows: list[dict]) -> dict:
    by_change = defaultdict(list)
    for s in srows:
        if s["changed"]:
            by_change[(s["seed"], s["rule"])].append(s["class"])
    completed = [all(c == "current" for c in v) for v in by_change.values()]
    state_classes = Counter(s["class"] for s in srows if s["changed"])
    unchanged = [s["original_intact"] for s in srows if not s["changed"]]
    passing = [r for r in rows if r["condition"] == "passing"]
    return {
        "changes": len(rows),
        "writes_mean": mean([r["writes"] for r in rows]), "writes_min": min((r["writes"] for r in rows), default=None),
        "writes_max": max((r["writes"] for r in rows), default=None),
        "places_mean": mean([r["places"] for r in rows]),
        "tool_calls_mean": mean([r["n_tool_calls"] for r in rows]),
        "prompt_tokens_mean": mean([r["prompt_tokens"] for r in rows]),
        "cost_per_change": mean([r["cost"] for r in rows]),
        "model_seconds_mean": mean([r["model_seconds"] for r in rows]),
        "no_write": sum(r["writes"] == 0 for r in rows),
        "adds": sum(r["adds"] for r in rows),
        "link_writes": sum(r["link_writes"] for r in rows),
        "ops": dict(sum((r["ops"] for r in rows), Counter())),
        "completed": prop(sum(completed), len(completed)),
        "governed_pairs": {c: state_classes[c] for c in CLASSES if c != "missing"} | {"missing": state_classes["vague"]},
        "governed_current": prop(state_classes["current"], sum(state_classes.values())),
        "unchanged_intact": prop(sum(unchanged), len(unchanged)),
        "plan_function_found": prop(sum(r.get("plan_function_found", False) for r in passing), len(passing)) if passing else None,
        "plan_rule_current": prop(sum(r.get("plan_rule") == "current" for r in passing), len(passing)) if passing else None,
        "capped": sum(r["capped"] for r in rows),
    }


def analyse(runs: list[str]) -> dict:
    dss = datasets()
    trows, crows, srows = [], [], []
    for run in runs:
        data = load(run)
        trows += [{**r, "run": run} for r in score_tasks(data, dss)]
        crows += [{**r, "run": run} for r in change_rows(data, dss)]
        srows += [{**r, "run": run} for r in state_rows(data)]
    out = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "runs": runs,
           "tasks": {}, "diffs": {}, "changes": {}}
    # Post hoc sensitivity (not in the plan): R02's request ends "the rest of the rule stays", and a text
    # that keeps "Refer to a customer by its company name" beside the new restriction is judged a
    # conflict; here such R02 conflicts count as current. Every other class is unchanged.
    lenient = [{**r, "rules": {k: ("current" if (k == "R02" and v == "conflict") else v) for k, v in r["rules"].items()}}
               for r in trows]
    sens = defaultdict(list)
    for r in lenient:
        sens[(r["model"], r["size"], r["arm"])].append(r)
    out["sensitivity_R02_conflict_as_current"] = {"tasks": {}, "diffs": {}}
    for (model, size, arm), rows in sorted(sens.items()):
        out["sensitivity_R02_conflict_as_current"]["tasks"]["|".join((model, size, arm, "both"))] = {
            "rules_current": summarise_tasks(rows)["rules_current"]}
        if arm == "linked":
            for other in TASK_ARMS:
                if other != "linked" and (model, size, other) in sens:
                    out["sensitivity_R02_conflict_as_current"]["diffs"][f"{model}|{size}|both|linked-vs-{other}"] = {
                        "rules_current": bootstrap_diff(items(sens[(model, size, other)], "rules_current"),
                                                        items(rows, "rules_current"), iters=4000, seed=0)}
    groups = defaultdict(list)
    for r in trows:
        groups[(r["model"], r["size"], r["arm"], r["condition"])].append(r)
        groups[(r["model"], r["size"], r["arm"], "both")].append(r)
    for k, rows in sorted(groups.items()):
        out["tasks"]["|".join(k)] = summarise_tasks(rows)
    for (model, size, arm, cond), rows in sorted(groups.items()):
        if arm != "linked":
            continue
        for other in TASK_ARMS:
            if other == "linked" or (model, size, other, cond) not in groups:
                continue
            base = groups[(model, size, other, cond)]
            d = {}
            for what in ("rules_current", "changed_current", "function_found", "hidden_found", "fully_correct", "cost_per_task"):
                a, b = items(base, what), items(rows, what)
                if a and b:
                    d[what] = bootstrap_diff(a, b, iters=4000, seed=0)
            out["diffs"][f"{model}|{size}|{cond}|linked-vs-{other}"] = d
    cg, sg = defaultdict(list), defaultdict(list)
    for r in crows:
        cg[(r["model"], r["size"], r["arm"], r["condition"])].append(r)
        cg[(r["model"], r["size"], r["arm"], "both")].append(r)
    for s in srows:
        sg[(s["model"], s["size"], s["arm"], s["condition"])].append(s)
    out["change_diffs"] = {}
    for (model, size, arm, cond), rows in sorted(cg.items()):
        if arm != "linked":
            continue
        for other in CHANGE_ARMS:
            if other == "linked" or (model, size, other, cond) not in cg:
                continue
            d = {}
            for what in ("writes", "cost", "n_tool_calls"):
                ia, ib = defaultdict(list), defaultdict(list)
                for r in sorted(cg[(model, size, other, cond)], key=lambda r: r["seed"]):
                    ia[f"{r['condition']}|{r['change']}"].append(float(r[what]))
                for r in sorted(rows, key=lambda r: r["seed"]):
                    ib[f"{r['condition']}|{r['change']}"].append(float(r[what]))
                d[what] = bootstrap_diff(ia, ib, iters=4000, seed=0)
            out["change_diffs"][f"{model}|{size}|{cond}|linked-vs-{other}"] = d
    for k, rows in sorted(cg.items()):
        model, size, arm, cond = k
        if cond == "both":
            ss = sg.get((model, size, arm, "direct"), []) + sg.get((model, size, arm, "passing"), [])
            ss = [{**s, "seed": f"{s['condition']}{s['seed']}"} for s in ss]
        else:
            ss = sg.get(k, [])
        out["changes"]["|".join(k)] = summarise_changes(rows, ss)
    return out


def ledger_costs(paths: list[str]) -> dict:
    """Spend from the ledgers, by model and by what the call was for (tag: arm|phase|size|condition, judge, validate, probe)."""
    by: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total, errors = 0.0, 0
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                errors += 1
                continue
            parts = (r.get("tag") or "").split("|")
            what = f"{parts[1]} {parts[2]}" if len(parts) >= 3 else parts[0]
            by[r["model"]][what] += r.get("cost") or 0.0
            total += r.get("cost") or 0.0
    return {"total": total, "errored_calls": errors, "by_model": {m: dict(v) for m, v in by.items()},
            "ledgers": paths}


def pct(p: dict | None) -> str:
    if not p or p["p"] is None:
        return "–"
    return f"{100 * p['p']:.1f}% [{100 * p['lo']:.0f}, {100 * p['hi']:.0f}]"


def pp(d: dict | None) -> str:
    if not d:
        return "–"
    return f"{100 * d['diff']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"


TIERS = [("cheap", "anthropic/claude-haiku-5.5"), ("cheap", "openai/gpt-6-luna"),
         ("mid", "openai/gpt-6.1-sol"), ("top", "anthropic/claude-opus-5.5")]


def tier_table(res: dict, size: str = "L") -> str:
    """Across tiers at one size: rules current per arm, linked minus each baseline, writes per change."""
    arms = ["pasted", "rule-skills", "rules-file", "long-context", "linked"]
    lines = [f"\n### Across model tiers, size {size} (conditions pooled where both ran)\n",
             "| Tier | Model | " + " | ".join(f"{a}: rules current" for a in arms) + " | " +
             " | ".join(f"linked minus {a}" for a in arms if a != "linked") + " | Right procedure, linked / rules-file | Writes per change, pasted / linked |",
             "|---|---|" + "---|" * (len(arms) + len(arms) - 1 + 2)]
    for tier, model in TIERS:
        if f"{model}|{size}|linked|both" not in res["tasks"]:
            continue
        cells = []
        for a in arms:
            t = res["tasks"].get(f"{model}|{size}|{a}|both")
            cells.append(pct(t["rules_current"]) if t else "–")
        for a in arms:
            if a == "linked":
                continue
            d = res["diffs"].get(f"{model}|{size}|both|linked-vs-{a}", {}).get("rules_current")
            cells.append(pp(d))
        lf = res["tasks"][f"{model}|{size}|linked|both"]["function_found"]
        rf = res["tasks"].get(f"{model}|{size}|rules-file|both", {}).get("function_found")
        cp = res["changes"].get(f"{model}|{size}|pasted|both")
        cl = res["changes"].get(f"{model}|{size}|linked|both")
        cells.append(f"{pct(lf)} / {pct(rf)}")
        cells.append(f"{cp['writes_mean']:.2f} / {cl['writes_mean']:.2f}" if cp and cl else "–")
        lines.append(f"| {tier} | {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def tables(res: dict) -> str:
    lines = [tier_table(res)]
    keys = sorted({tuple(k.split("|")[:2]) for k in res["tasks"]})
    for model, size in keys:
        for cond in ("both", "direct", "passing"):
            lines += [f"\n### Tasks: {model}, size {size}, {cond if cond != 'both' else 'both conditions'}\n",
                      "| Arm | Rules current | Changed rules current | Stale / conflict / vague / missing | Right procedure (hidden) | Fully correct | Extras per task | Tool calls | Prompt tokens | Cost per task | Model s (median) |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
            for arm in TASK_ARMS:
                s = res["tasks"].get(f"{model}|{size}|{arm}|{cond}")
                if not s:
                    continue
                pr = s["pairs"]
                lines.append(f"| {arm} | {pct(s['rules_current'])} | {pct(s['changed_current'])} | "
                             f"{pr['stale']} / {pr['conflict']} / {pr['vague']} / {pr['missing']} | "
                             f"{pct(s['function_found'])} ({s['hidden_found']['k']}/{s['hidden_found']['n']}) | {pct(s['fully_correct'])} | "
                             f"{s['extras_per_task']:.2f} | {s['tool_calls']:.1f} | {s['prompt_tokens']:,.0f} | ${s['cost_per_task']:.4f} | "
                             f"{s['model_seconds_median']:.1f} |")
            dl = [(k.split("|linked-vs-")[1], v) for k, v in res["diffs"].items()
                  if k.startswith(f"{model}|{size}|{cond}|linked-vs-")]
            if dl:
                lines += ["", "| linked minus | Rules current (pts) | Changed rules current | Right procedure | Hidden procedures | Fully correct | Cost per task ($) |",
                          "|---|---|---|---|---|---|---|"]
                for other, d in dl:
                    c = d.get("cost_per_task")
                    cost = f"{c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" if c else "–"
                    lines.append(f"| {other} | {pp(d.get('rules_current'))} | {pp(d.get('changed_current'))} | "
                                 f"{pp(d.get('function_found'))} | {pp(d.get('hidden_found'))} | {pp(d.get('fully_correct'))} | {cost} |")
        cdl = [(k.split("|linked-vs-")[1], v) for k, v in res.get("change_diffs", {}).items()
               if k.startswith(f"{model}|{size}|both|linked-vs-")]
        if cdl:
            lines += [f"\n### Changes, linked minus each arm: {model}, size {size}, both conditions\n",
                      "| linked minus | Writes per change | Tool calls per change | Cost per change ($) |", "|---|---|---|---|"]
            for other, d in cdl:
                w, t, c = d["writes"], d["n_tool_calls"], d["cost"]
                lines.append(f"| {other} | {w['diff']:+.2f} [{w['lo']:+.2f}, {w['hi']:+.2f}] | {t['diff']:+.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}] | "
                             f"{c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] |")
        for cond in ("direct", "passing"):
            lines += [f"\n### Changes: {model}, size {size}, {cond}\n",
                      "| Arm | Writes per change (range) | Places | Tool calls | Cost per change | Model s | Changes completed | Governed pairs current | Stale / conflict / missing pairs | Unchanged rules intact | Adds |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
            for arm in CHANGE_ARMS:
                s = res["changes"].get(f"{model}|{size}|{arm}|{cond}")
                if not s:
                    continue
                g = s["governed_pairs"]
                lines.append(f"| {arm} | {s['writes_mean']:.1f} ({s['writes_min']}–{s['writes_max']}) | {s['places_mean']:.1f} | "
                             f"{s['tool_calls_mean']:.1f} | ${s['cost_per_change']:.4f} | {s['model_seconds_mean']:.1f} | "
                             f"{pct(s['completed'])} | {pct(s['governed_current'])} | {g.get('stale', 0)} / {g.get('conflict', 0)} / {g.get('vague', 0)} | "
                             f"{pct(s['unchanged_intact'])} | {s['adds']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--ledger", action="append", default=[], help="ledgers whose spend to report (repeatable)")
    args = ap.parse_args()
    res = analyse(args.run)
    if args.ledger:
        res["cost"] = ledger_costs(args.ledger)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    if args.tables or not args.out:
        print(tables(res))
        sens = res.get("sensitivity_R02_conflict_as_current", {})
        if sens.get("diffs"):
            print("\n### Post hoc sensitivity: R02 keep-and-append conflicts counted as current (both conditions)\n")
            print("| Model, size | linked rules current | " + " | ".join(f"linked minus {a}" for a in TASK_ARMS if a != "linked") + " |")
            print("|---|---|" + "---|" * (len(TASK_ARMS) - 1))
            for key in sorted({"|".join(k.split("|")[:2]) for k in sens["tasks"]}):
                model, size = key.split("|")[:2]
                lin = sens["tasks"].get(f"{model}|{size}|linked|both")
                if not lin:
                    continue
                cells = [pp(sens["diffs"].get(f"{model}|{size}|both|linked-vs-{a}", {}).get("rules_current")) for a in TASK_ARMS if a != "linked"]
                print(f"| {model}, {size} | {pct(lin['rules_current'])} | " + " | ".join(cells) + " |")
        if "cost" in res:
            c = res["cost"]
            print(f"\n### Spend from the ledgers: ${c['total']:.2f} ({c['errored_calls']} errored calls, retried)\n")
            for m, v in sorted(c["by_model"].items()):
                print(f"- {m}: ${sum(v.values()):.2f} (" + ", ".join(f"{k} ${x:.2f}" for k, x in sorted(v.items())) + ")")


if __name__ == "__main__":
    main()
