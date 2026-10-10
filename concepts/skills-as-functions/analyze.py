"""Every number in the README: python analyze.py [--label main] [--out results.json]

Reads runs/<label>/<size>/<model>/<arm>-s<seed>.jsonl, and writes
results.json and the tables (markdown) to stdout. Comparisons pair the
library with each other arm by task position, keeping each position's seeds
together (demos_common.stats.bootstrap_diff).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from demos_common.stats import bootstrap_diff, mean, wilson  # noqa: E402

from skills.api import VERSIONS  # noqa: E402
from skills.tasks import SIZES  # noqa: E402

ARMS = ["none", "notes", "retrieval", "fullctx", "library"]
METRICS = {
    "passed": lambda r: float(r["passed"]),
    "cost": lambda r: r["cost"],
    "agent_cost": lambda r: r["agent_cost"],
    "agent_latency_s": lambda r: r["agent_latency_s"],
    "memory_latency_s": lambda r: r["memory_latency_s"],
    "calls": lambda r: r["calls"],
    "agent_calls": lambda r: r["agent_calls"],
    "prompt_tokens": lambda r: r["prompt_tokens"],
    "completion_tokens": lambda r: r["completion_tokens"],
}
WINDOW = 8  # tasks after a change


def load(label: str) -> dict[tuple[str, str, str], dict[int, list[dict[str, Any]]]]:
    """(size, model, arm) -> seed -> records in order. Only complete streams."""
    out: dict[tuple[str, str, str], dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    for p in sorted(glob.glob(str(HERE / "runs" / label / "*" / "*" / "*-s*.jsonl"))):
        if p.endswith(".transcripts.jsonl"):
            continue
        recs = [json.loads(line) for line in open(p) if line.strip()]
        if not recs:
            continue
        size, model, arm, seed = recs[0]["size"], recs[0]["model"], recs[0]["arm"], recs[0]["seed"]
        if len(recs) != SIZES[size]["tasks"] or not Path(p.replace(".jsonl", ".memory.json")).exists():
            print(f"skipping incomplete {p} ({len(recs)} tasks)", file=sys.stderr)
            continue
        out[(size, model, arm)][seed] = recs
    return out


def by_position(streams: dict[int, list[dict]], metric: str, keep=lambda r: True) -> dict[str, list[float]]:
    f = METRICS[metric]
    m: dict[str, list[float]] = defaultdict(list)
    for seed, recs in sorted(streams.items()):
        for r in recs:
            if keep(r):
                m[str(r["position"])].append(f(r))
    return dict(m)


def flat(m: dict[str, list[float]]) -> list[float]:
    return [v for vs in m.values() for v in vs]


def phase(size: str, position: int) -> str:
    changes = sorted(SIZES[size]["changes"])
    v = 1
    for c in changes:
        if position >= c:
            v += 1
    return f"v{v}"


def after_change(size: str, position: int) -> bool:
    return any(c <= position < c + WINDOW for c in SIZES[size]["changes"])


# The shared steps each change breaks (see api.VERSIONS): v2 retires the contact
# search and the company contact listing, v3 retires send_email.
BROKEN_BY = {2: {"find_contact", "company_contacts"}, 3: {"email"}}


def first_hit_positions(size: str, k: int = 3) -> set[int]:
    """For each change, the first k tasks after it (before the next change) whose
    kind uses a step the change broke: where stale memory is first tested."""
    from skills.tasks import KINDS, schedule

    sched = schedule(size)
    out: set[int] = set()
    changes = sorted(SIZES[size]["changes"].items())
    for i, (pos, ver) in enumerate(changes):
        end = changes[i + 1][0] if i + 1 < len(changes) else len(sched) + 1
        hits = [p for p in range(pos, end) if set(KINDS[sched[p - 1]["kind"]].uses) & BROKEN_BY[ver]]
        out |= set(hits[:k])
    return out


SUBSETS = {
    "all": lambda r: True,
    "first of kind": lambda r: r["first_of_kind"],
    "recurring kind, new values": lambda r: not r["first_of_kind"] and r["repeat_of"] is None,
    "word-for-word repeat": lambda r: r["repeat_of"] is not None,
    "level 1": lambda r: r["level"] == 1,
    "level 2": lambda r: r["level"] == 2,
    "level 3": lambda r: r["level"] == 3,
    # the plan predicts the library's advantage grows along the stream
    "second half of the stream": lambda r: r["position"] > SIZES[r["size"]]["tasks"] // 2,
}


def summarise_arm(streams: dict[int, list[dict]], keep=lambda r: True) -> dict[str, Any]:
    recs = [r for s in streams.values() for r in s if keep(r)]
    n = len(recs)
    k = sum(r["passed"] for r in recs)
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    out = {"n": n, "passed": k, "accuracy": k / n if n else float("nan"), "accuracy_ci": [lo, hi]}
    for m in METRICS:
        if m != "passed":
            out[m] = mean([METRICS[m](r) for r in recs]) if recs else float("nan")
    out["per_seed_accuracy"] = {s: mean([float(r["passed"]) for r in v if keep(r)]) for s, v in streams.items()}
    out["per_seed_cost"] = {s: sum(r["cost"] for r in v if keep(r)) for s, v in streams.items()}
    return out


def compare(lib: dict[int, list[dict]], other: dict[int, list[dict]], keep=lambda r: True) -> dict[str, Any]:
    """library minus other, paired by position, seeds kept together."""
    out = {}
    for m in ["passed", "cost", "agent_cost", "agent_latency_s", "calls", "agent_calls", "completion_tokens", "prompt_tokens"]:
        a = by_position(other, m, keep)
        b = by_position(lib, m, keep)
        if not a or not b:
            continue
        out[m] = bootstrap_diff(a, b, iters=5000, seed=0)
        out[m]["other_mean"] = mean(flat(a))
        out[m]["library_mean"] = mean(flat(b))
    return out


def call_graph(functions: dict[str, Any]) -> dict[str, set[str]]:
    """Which library functions each stored function calls (by name, from its source)."""
    import ast as _ast

    names = set(functions)
    graph = {}
    for n, f in functions.items():
        called = {node.func.id for node in _ast.walk(_ast.parse(f["source"]))
                  if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)}
        graph[n] = (called & names) - {n}
    return graph


def depth(graph: dict[str, set[str]], n: str, seen: frozenset = frozenset()) -> int:
    if n in seen or not graph.get(n):
        return 1
    return 1 + max(depth(graph, m, seen | {n}) for m in graph[n])


def library_stats(streams: dict[int, list[dict]], size: str, memory: dict[int, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seed, recs in streams.items():
        s: dict[str, Any] = {}
        if memory and seed in memory:
            fns = memory[seed]
            graph = call_graph(fns)
            callers = defaultdict(int)
            for n, cs in graph.items():
                for c in cs:
                    callers[c] += 1
            uses = defaultdict(int)
            for r in recs:
                for n in r["library_calls"]:
                    uses[n] += 1
            s["functions"] = len(fns)
            s["called_by_another_function"] = sum(1 for n in fns if callers[n] > 0)
            s["max_depth"] = max((depth(graph, n) for n in fns), default=0)
            s["ran_in_two_or_more_tasks"] = sum(1 for n in fns if uses[n] >= 2)
            s["never_ran_after_saving"] = sum(1 for n in fns if uses[n] == 0)
        s["final_size"] = recs[-1].get("library_size", recs[-1].get("library_size_before"))
        s["tasks_calling_library"] = sum(bool(r["library_calls"]) for r in recs)
        s["saves"] = defaultdict(int)
        for r in recs:
            s["saves"][(r.get("save") or "none").split(" (")[0].split(" on attempt")[0]] += 1
        s["saves"] = dict(s["saves"])
        s["tasks_with_library_failure"] = [r["position"] for r in recs if r["library_failed"]]
        s["replaced"] = [(r["position"], r["replaced"]) for r in recs if r.get("replaced")]
        # repairs: a task in which a stored function raised, that passed, and whose save replaced at
        # least one function in the failing call chain. Left broken: a stored function where an error
        # surfaced that raised again in a later task with no replacement in between.
        repairs, refailed = [], []
        last_fail: dict[str, int] = {}
        replaced_at: dict[str, list[int]] = defaultdict(list)
        for r in recs:
            for n in r.get("replaced") or []:
                replaced_at[n].append(r["position"])
        for r in recs:
            chain = set(r["library_failed"])
            if chain and r["passed"] and chain & set(r.get("replaced") or []):
                repairs.append(r["position"])
            for f in r.get("library_failed_at") or r["library_failed"][-1:]:
                if f in last_fail:
                    t1 = last_fail[f]
                    if not any(t1 <= t < r["position"] for t in replaced_at[f]):
                        refailed.append((f, t1, r["position"]))
                last_fail[f] = r["position"]
        s["repaired_in_the_same_task"] = repairs
        s["failed_again_unrepaired"] = refailed
        out[seed] = s
    return out


def verdict(g: dict[str, Any]) -> dict[str, Any]:
    """The plan's criterion. Against each arm, the library wins if it is more
    accurate (interval above zero), or at least as accurate (interval above -5
    points) and cheaper per task (interval below zero). The idea holds at a size
    if it wins against every arm. The change claim holds if it fails fewer of
    the tasks just after a change than notes, retrieval and fullctx."""
    out: dict[str, Any] = {"against": {}}
    for arm, c in g["compare"].items():
        acc, cost = c["all"]["passed"], c["all"]["cost"]
        more_accurate = acc["lo"] > 0
        not_worse = acc["lo"] > -0.05
        cheaper = cost["hi"] < 0
        less_accurate = acc["hi"] < 0
        out["against"][arm] = {
            "wins": bool(more_accurate or (not_worse and cheaper)),
            "more_accurate": bool(more_accurate), "at_least_as_accurate": bool(not_worse),
            "cheaper": bool(cheaper), "less_accurate": bool(less_accurate),
        }
        if "after change" in c:
            ch = c["after change"]["passed"]
            out["against"][arm]["after_change_diff"] = ch["diff"]
            out["against"][arm]["after_change_interval_above_zero"] = bool(ch["lo"] > 0)
    out["holds"] = all(v["wins"] for v in out["against"].values())
    rivals = [a for a in ("notes", "retrieval", "fullctx") if a in out["against"] and "after_change_diff" in out["against"][a]]
    if rivals:
        out["change_claim_point"] = all(out["against"][a]["after_change_diff"] > 0 for a in rivals)
        out["change_claim_intervals"] = all(out["against"][a]["after_change_interval_above_zero"] for a in rivals)
    out["loses_on_accuracy_to"] = [a for a, v in out["against"].items() if v["less_accurate"]]
    return out


def blocks(streams: dict[int, list[dict]], metric: str, width: int = 8) -> list[float]:
    f = METRICS[metric]
    n = len(next(iter(streams.values())))
    out = []
    for start in range(0, n, width):
        vals = [f(r) for recs in streams.values() for r in recs[start:start + width]]
        out.append(mean(vals))
    return out


def fmt_ci(d: dict[str, float], scale: float = 1.0, pct: bool = False, digits: int = 1) -> str:
    if pct:
        return f"{100 * d['diff']:+.{digits}f} [{100 * d['lo']:+.{digits}f}, {100 * d['hi']:+.{digits}f}]"
    return f"{scale * d['diff']:+.{digits}f} [{scale * d['lo']:+.{digits}f}, {scale * d['hi']:+.{digits}f}]"


# Models of May 2025, run briefly under the lead's amendment 1 and then dropped (amendment 2):
# their spend is reported apart and is not part of the concept's budget for today's models.
OLDER_MODELS = {"openai/gpt-4.1", "anthropic/claude-sonnet-4"}
# runs/ledger.jsonl holds the calls made on the laptop, runs/ledger-vm.jsonl those made on the
# cloud machine (copied back whole); no call is in both.
LEDGERS = ["ledger.jsonl", "ledger-vm.jsonl"]


def spend() -> dict[str, Any]:
    """Spend from the ledgers, by run label, size and model, for today's models and the older ones."""
    by: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    today = older = 0.0
    for name in LEDGERS:
        p = HERE / "runs" / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            m = r.get("meta") or {}
            label = m.get("label") or ("probe" if "probe" in m else "other")
            cost = r.get("cost") or 0.0
            key = f"{label}|{m.get('size') or '-'}|{r['model']}"
            by[key] += cost
            calls[key] += 1
            if r["model"] in OLDER_MODELS:
                older += cost
            else:
                today += cost
    return {"today_models_usd": round(today, 4), "older_models_usd": round(older, 4),
            "by_label_size_model": {k: {"usd": round(v, 4), "calls": calls[k]} for k, v in sorted(by.items())},
            "note": "label main and tiers include runs later discarded (runs/discarded/); probe, sanity and smoke ran on the laptop"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="main,tiers", help="run labels to analyse, each its own set of conditions")
    p.add_argument("--out", default="results.json")
    a = p.parse_args()
    results: dict[str, Any] = {"labels": a.labels.split(","), "groups": {}}
    for label in a.labels.split(","):
        analyse_label(label, results)
    results["spend"] = spend()
    Path(HERE / a.out).write_text(json.dumps(results, indent=1, default=str))
    print_tables(results)
    s = results["spend"]
    print(f"\n## Spend\n\nToday's models ${s['today_models_usd']:.2f}; older models (amendment 1, outside the analysis) ${s['older_models_usd']:.2f}.\n")
    print("| label | size | model | calls | cost |")
    print("|---|---|---|---|---|")
    for k, v in s["by_label_size_model"].items():
        label, size, model = k.split("|")
        print(f"| {label} | {size} | {model} | {v['calls']:,} | ${v['usd']:.2f} |")


def analyse_label(label: str, results: dict[str, Any]) -> None:
    data = load(label)
    groups = sorted({(s, m) for s, m, _ in data}, key=lambda x: (list(SIZES).index(x[0]), x[1]))
    for size, model in groups:
        g: dict[str, Any] = {"label": label, "size": size, "model": model, "arms": {}, "compare": {}, "subsets": {},
                             "phases": {}, "after_change": {}, "blocks": {}}
        arms = [x for x in ARMS if (size, model, x) in data]
        for arm in arms:
            st = data[(size, model, arm)]
            g["arms"][arm] = summarise_arm(st)
            g["arms"][arm]["seeds"] = sorted(st)
            g["arms"][arm]["failures"] = [
                {"seed": s, "position": r["position"], "kind": r["kind"], "version": r["version"],
                 "repeat": r["repeat_of"] is not None, "reason": (r["reasons"] or [""])[0][:160]}
                for s, recs in sorted(st.items()) for r in recs if not r["passed"]]
            g["subsets"][arm] = {name: summarise_arm(st, f) for name, f in SUBSETS.items()}
            g["phases"][arm] = {ph: summarise_arm(st, lambda r, ph=ph: phase(size, r["position"]) == ph)
                                for ph in sorted({phase(size, r["position"]) for r in next(iter(st.values()))})}
            if SIZES[size]["changes"]:
                g["after_change"][arm] = summarise_arm(st, lambda r: after_change(size, r["position"]))
                hits = first_hit_positions(size)
                g.setdefault("first_hits", {})[arm] = summarise_arm(st, lambda r, hits=hits: r["position"] in hits)
            g["blocks"][arm] = {m: blocks(st, m) for m in ["passed", "cost", "agent_calls", "agent_latency_s", "completion_tokens"]}
        if "library" in arms:
            lib = data[(size, model, "library")]
            for arm in arms:
                if arm == "library":
                    continue
                common = sorted(set(lib) & set(data[(size, model, arm)]))
                L = {s: lib[s] for s in common}
                O = {s: data[(size, model, arm)][s] for s in common}
                g["compare"][arm] = {"seeds": common, "all": compare(L, O)}
                for name, f in SUBSETS.items():
                    if name != "all":
                        g["compare"][arm][name] = compare(L, O, f)
                if SIZES[size]["changes"]:
                    g["compare"][arm]["after change"] = compare(L, O, lambda r: after_change(size, r["position"]))
                    hits = first_hit_positions(size)
                    g["compare"][arm]["first tasks to hit a broken step"] = compare(L, O, lambda r, hits=hits: r["position"] in hits)
            mem = {}
            for seed in lib:
                mp = HERE / "runs" / label / size / model.split("/")[-1] / f"library-s{seed}.memory.json"
                if mp.exists():
                    mem[seed] = json.loads(mp.read_text())
            g["library"] = library_stats(lib, size, mem)
            g["verdict"] = verdict(g)
        any_rec = next(iter(next(iter(data[(size, model, arms[0])].values()))))
        g["settings"] = any_rec.get("settings") or {"tool_choice": "required", "reasoning": None}
        results["groups"][f"{label}|{size}|{model}"] = g


def print_summary(results: dict[str, Any]) -> None:
    """One table across every condition: each arm's accuracy and cost, and the
    library's difference from each with its interval."""
    print("\n## Across conditions\n")
    print("| condition | seeds | arm | accuracy | cost per task | agent latency (s) | library minus arm: accuracy (points) | cost ($ thousandths) | agent latency (s) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key, g in results["groups"].items():
        st = g.get("settings") or {}
        cond = f"{g['size']}, {g['model'].split('/')[-1]}" + (", auto" if st.get("tool_choice") == "auto" else "") + (f", reasoning {st['reasoning']}" if st.get("reasoning") else "")
        seeds = sorted({x for a in g["arms"].values() for x in a["seeds"]})
        for arm, a in g["arms"].items():
            c = g["compare"].get(arm, {}).get("all")
            diff = (f"{fmt_ci(c['passed'], pct=True)} | {fmt_ci(c['cost'], scale=1000, digits=3)} | {fmt_ci(c['agent_latency_s'])}"
                    if c else "- | - | -")
            print(f"| {cond} | {len(seeds)} | {arm} | {100 * a['accuracy']:.1f}% | ${1000 * a['cost']:.3f}m | {a['agent_latency_s']:.1f} | {diff} |")


def print_tables(results: dict[str, Any]) -> None:
    print_summary(results)
    for key, g in results["groups"].items():
        size, model = g["size"], g["model"]
        seeds = sorted({s for a in g["arms"].values() for s in a["seeds"]})
        st = g.get("settings") or {}
        how = f"tool choice {st.get('tool_choice', 'required')}" + (f", reasoning {st['reasoning']}" if st.get("reasoning") else "")
        print(f"\n### {size}, {model.split('/')[-1]}, {how} ({SIZES[size]['tasks']} tasks x {len(seeds)} seeds; label {g['label']})\n")
        print("| arm | correct | accuracy [95% CI] | cost per task | agent latency (s) | memory latency (s) | calls (agent + memory) | prompt tokens | completion tokens |")
        print("|---|---|---|---|---|---|---|---|---|")
        for arm, s in g["arms"].items():
            print(f"| {arm} | {s['passed']}/{s['n']} | {100 * s['accuracy']:.1f}% [{100 * s['accuracy_ci'][0]:.1f}, {100 * s['accuracy_ci'][1]:.1f}] "
                  f"| ${1000 * s['cost']:.3f}m | {s['agent_latency_s']:.1f} | {s['memory_latency_s']:.1f} "
                  f"| {s['agent_calls']:.2f} + {s['calls'] - s['agent_calls']:.2f} | {s['prompt_tokens']:,.0f} | {s['completion_tokens']:,.0f} |")
        if g["compare"]:
            print("\nLibrary minus each arm, paired by position (95% bootstrap interval):\n")
            print("| against | accuracy (points) | cost per task ($ thousandths) | agent-loop cost only | agent latency (s) | calls | correct after a change (points) |")
            print("|---|---|---|---|---|---|---|")
            for arm, c in g["compare"].items():
                al = c["all"]
                ac = c.get("after change", {}).get("passed")
                print(f"| {arm} | {fmt_ci(al['passed'], pct=True)} | {fmt_ci(al['cost'], scale=1000, digits=3)} "
                      f"| {fmt_ci(al['agent_cost'], scale=1000, digits=3)} "
                      f"| {fmt_ci(al['agent_latency_s'])} | {fmt_ci(al['calls'], digits=2)} | {fmt_ci(ac, pct=True) if ac else '-'} |")
            print("\nLibrary minus each arm on subsets: accuracy (points) and cost per task ($ thousandths), 95% intervals:\n")
            subs = ["first of kind", "recurring kind, new values", "word-for-word repeat", "level 3", "second half of the stream",
                    "first tasks to hit a broken step"]
            print("| against | " + " | ".join(subs) + " |")
            print("|---|" + "---|" * len(subs))
            for arm, c in g["compare"].items():
                cells = []
                for sname in subs:
                    x = c.get(sname)
                    if not x or "passed" not in x:
                        cells.append("-")
                        continue
                    cells.append(f"{fmt_ci(x['passed'], pct=True)}; {fmt_ci(x['cost'], scale=1000, digits=2)}")
                print(f"| {arm} | " + " | ".join(cells) + " |")
        print("\nAccuracy by subset (%):\n")
        names = list(SUBSETS)
        print("| arm | " + " | ".join(names) + " | 8 tasks after a change | first 3 tasks to use a broken step |")
        print("|---|" + "---|" * (len(names) + 2))
        for arm in g["arms"]:
            row = [f"{100 * g['subsets'][arm][n]['accuracy']:.0f} (n={g['subsets'][arm][n]['n']})" for n in names]
            ac = g["after_change"].get(arm)
            row.append(f"{100 * ac['accuracy']:.0f} (n={ac['n']})" if ac else "-")
            fh = g.get("first_hits", {}).get(arm)
            row.append(f"{100 * fh['accuracy']:.0f} (n={fh['n']}), {fh['agent_calls']:.1f} calls, ${1000 * fh['cost']:.2f}m" if fh else "-")
            print(f"| {arm} | " + " | ".join(row) + " |")
        print("\nCost per task by subset ($ thousandths):\n")
        print("| arm | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for arm in g["arms"]:
            print(f"| {arm} | " + " | ".join(f"{1000 * g['subsets'][arm][n]['cost']:.3f}" for n in names) + " |")
        print("\nBy phase (accuracy %, cost $ thousandths, agent calls):\n")
        phases = sorted({ph for arm in g["phases"] for ph in g["phases"][arm]})
        print("| arm | " + " | ".join(phases) + " |")
        print("|---|" + "---|" * len(phases))
        for arm in g["phases"]:
            print(f"| {arm} | " + " | ".join(
                f"{100 * g['phases'][arm][ph]['accuracy']:.0f}%, {1000 * g['phases'][arm][ph]['cost']:.2f}, {g['phases'][arm][ph]['agent_calls']:.2f}"
                for ph in phases) + " |")
        print("\nPer block of 8 tasks: accuracy % / cost $ thousandths:\n")
        nb = len(next(iter(g["blocks"].values()))["passed"])
        print("| arm | " + " | ".join(f"{8 * i + 1}-{8 * i + 8}" for i in range(nb)) + " |")
        print("|---|" + "---|" * nb)
        for arm, b in g["blocks"].items():
            print(f"| {arm} | " + " | ".join(f"{100 * p:.0f} / {1000 * c:.2f}" for p, c in zip(b["passed"], b["cost"])) + " |")
        print("\nPer block of 8 tasks: agent calls / agent latency (s):\n")
        print("| arm | " + " | ".join(f"{8 * i + 1}-{8 * i + 8}" for i in range(nb)) + " |")
        print("|---|" + "---|" * nb)
        for arm, b in g["blocks"].items():
            print(f"| {arm} | " + " | ".join(f"{c:.2f} / {t:.1f}" for c, t in zip(b["agent_calls"], b["agent_latency_s"])) + " |")
        if "verdict" in g:
            v = g["verdict"]
            def acc_word(x):
                if x["more_accurate"]:
                    return "more accurate"
                if x["less_accurate"]:
                    return "less accurate (interval below zero), though within the 5-point margin" if x["at_least_as_accurate"] else "less accurate"
                return "as accurate within the 5-point margin" if x["at_least_as_accurate"] else "not shown as accurate"
            print("\nThe plan's criterion: " + "; ".join(
                f"against {a}: {'wins' if x['wins'] else 'does not win'} ({acc_word(x)}, {'cheaper' if x['cheaper'] else 'not cheaper'})"
                for a, x in v["against"].items()) + f". Holds at this size: {v['holds']}"
                + (f" (and less accurate than {', '.join(v['loses_on_accuracy_to'])}, which the plan says rules it out)" if v["loses_on_accuracy_to"] else "") + "."
                + (f" Fewer failures after a change than notes, retrieval and fullctx: point estimates {v['change_claim_point']}, intervals {v['change_claim_intervals']}." if "change_claim_point" in v else ""))
        if "library" in g:
            print("\nLibrary, per seed:\n")
            for seed, s in g["library"].items():
                if "functions" in s:
                    print(f"- seed {seed}: {s['functions']} functions at the end, {s['called_by_another_function']} called by another "
                          f"stored function, deepest chain {s['max_depth']}, {s['ran_in_two_or_more_tasks']} ran in two or more tasks, "
                          f"{s['never_ran_after_saving']} never ran again after being saved")
                print(f"- seed {seed}: final size {s['final_size']}, tasks calling it {s['tasks_calling_library']}, "
                      f"saves {s['saves']}, tasks where a stored function failed {s['tasks_with_library_failure']}, "
                      f"repaired in the same task {s['repaired_in_the_same_task']}, "
                      f"failed again unrepaired (function, first task, later task) {s['failed_again_unrepaired']}")


if __name__ == "__main__":
    main()
