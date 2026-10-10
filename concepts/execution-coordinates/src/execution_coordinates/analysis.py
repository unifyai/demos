"""Turn runs/ into results.json and the README's tables.

    uv run python analyze.py            # every complete run under runs/<phase>/

Units: a request is correct or not; an item for the paired bootstrap is one
client's requests in one seed (the same data and requests in every arm).
Wrong answers get a cause:
  leak        the code read a name another client's request last changed, or
              the answer equals what another client's data, rates or scores give
  own_whatif  the code read state the client's own what-if (request 3)
              changed, or the answer equals the what-if left in place
  other       anything else (misreading the request, a parsing slip, ...)
A reading of the wrong answers (review.json) overrides the automatic cause.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from demos_common.stats import bootstrap_diff, summary, wilson

from .reads import refined_reads

HERE = Path(__file__).resolve().parents[2]
RUNS = HERE / "runs"
# today's models only (Amendment 3); runs/superseded-era-models/ is not read
PHASES = ["main", "ablation", "family2", "capable", "top", "initials"]
SEPARATE_CONDS = {"initials"}
ARM_ORDER = ["fresh", "shared", "per_task", "coordinate", "long_shared", "long_coordinate", "per_task_modes"]


def load_runs(phases=PHASES, runs_dir: Path | None = None) -> tuple[list[dict], list[dict]]:
    runs_dir = runs_dir or RUNS
    reqs, setups = [], []
    for ph in phases:
        for p in sorted((runs_dir / ph).glob("*.jsonl")):
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            if not rows or rows[-1].get("kind") != "run":
                continue  # incomplete
            setups.append(rows[0])
            reqs.extend(r for r in rows if r["kind"] == "request")
    return reqs, setups


def reads_split(r: dict) -> tuple[list[str], list[str]]:
    """State the request's cells read from their session (confirmed by the
    scope-aware analysis): changed last by another client, or by an earlier
    request of the same client."""
    pairs = refined_reads(r["tool_calls"])
    me = f"{r['client']}#{r['req_idx'] + 1}"
    other = sorted({f"{n}<-{t}" for n, t in pairs if t.split("#")[0] != r["client"]})
    own = sorted({f"{n}<-{t}" for n, t in pairs if t.split("#")[0] == r["client"] and t != me})
    return other, own


def cause_of(r: dict, review: dict) -> str | None:
    if r["correct"]:
        return None
    key = f"{r['run_id']}#{r['pos']}"
    if key in review:
        return review[key]["cause"]
    sig = r.get("causes") or []
    other, own = reads_split(r)
    if other or any(c.startswith("leak:") for c in sig):
        return "leak"
    own3 = f"{r['client']}#3"  # the client's what-if
    if r["req_idx"] > 2 and (any(x.endswith(own3) for x in own) or any(c.startswith("own_whatif") for c in sig)):
        return "own_whatif"
    return "other"


def item_values(reqs: list[dict], fn) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in reqs:
        out[f"k{r['k']}s{r['seed']}:{r['client']}"].append(fn(r))
    return dict(out)


def per_client(reqs: list[dict]) -> dict[str, dict]:
    """Per (run, client): own loads and scores, and reloads beyond the first."""
    out: dict[str, dict] = defaultdict(lambda: {"loads": 0, "scores": 0, "other_loads": 0})
    for r in reqs:
        key = f"k{r['k']}s{r['seed']}:{r['client']}"
        d = out[key]
        own_loads = r["loads"] - r.get("loads_other_client", 0)
        d["loads"] += own_loads
        d["other_loads"] += r.get("loads_other_client", 0)
        d["scores"] += r["scores"]
    for d in out.values():
        d["reloads"] = max(0, d["loads"] - 1) + max(0, d["scores"] - 1)
    return dict(out)


def modes_summary(reqs: list[dict]) -> dict:
    c = Counter()
    sess_named_client = 0
    sess_calls = 0
    sessions_per_client = defaultdict(set)
    whatif_modes = Counter()
    tools = Counter()
    for r in reqs:
        for t in r["tool_calls"]:
            tools[t["name"]] += 1
            if t["name"] != "run_python":
                continue
            m = t["args"].get("mode") if t.get("args") else None
            m = m or t.get("mode") or "?"
            c[m] += 1
            s = (t.get("args") or {}).get("session") or t.get("session")
            if s and m in ("stateful", "read_only"):
                sess_calls += 1
                sessions_per_client[f"k{r['k']}s{r['seed']}:{r['client']}"].add(s)
                if r["client"] in s.lower():
                    sess_named_client += 1
            if r["req_id"].startswith("r3_"):
                whatif_modes[m] += 1
    shared_names = 0
    by_run = defaultdict(lambda: defaultdict(set))
    for key, names in sessions_per_client.items():
        run = key.split(":")[0]
        for n in names:
            by_run[run][n].add(key)
    for run, names in by_run.items():
        shared_names += sum(1 for n, users in names.items() if len(users) > 1)
    return {
        "run_python_by_mode": dict(c), "tool_calls": dict(tools), "whatif_calls_by_mode": dict(whatif_modes),
        "session_calls_named_after_client": sess_named_client, "session_calls": sess_calls,
        "sessions_used_by_more_than_one_client": shared_names,
        "mean_sessions_per_client": mean(len(v) for v in sessions_per_client.values()) if sessions_per_client else 0,
    }


def summarise_group(reqs: list[dict], review: dict) -> dict:
    n = len(reqs)
    k = sum(r["correct"] for r in reqs)
    lo, hi = wilson(k, n)
    causes = Counter(cause_of(r, review) for r in reqs if not r["correct"])
    pc = per_client(reqs)
    runs = defaultdict(list)
    for r in reqs:
        runs[r["run_id"]].append(r)
    run_minutes = [sum(x["agent_s"] for x in rs) / 60 for rs in runs.values()]
    out = {
        "requests": n, "correct": k, "accuracy": k / n if n else None, "accuracy_ci": [lo, hi],
        "wrong_by_cause": {c: causes.get(c, 0) for c in ("leak", "own_whatif", "other")},
        "state_caused_wrong": causes.get("leak", 0) + causes.get("own_whatif", 0),
        "state_caused_rate": (causes.get("leak", 0) + causes.get("own_whatif", 0)) / n if n else None,
        "state_caused_ci": list(wilson(causes.get("leak", 0) + causes.get("own_whatif", 0), n)),
        "loads_per_client": mean(d["loads"] for d in pc.values()),
        "reloads_per_client": mean(d["reloads"] for d in pc.values()),
        "other_client_loads": sum(d["other_loads"] for d in pc.values()),
        "agent_s_per_request": summary([r["agent_s"] for r in reqs]),
        "agent_minutes_per_run": summary(run_minutes),
        "cost_per_request": mean(r["cost"] for r in reqs),
        "cost_total": sum(r["cost"] for r in reqs),
        "prompt_tokens_per_request": mean(r["prompt_tokens"] for r in reqs),
        "completion_tokens_per_request": mean(r["completion_tokens"] for r in reqs),
        "cached_tokens_per_request": mean(r["cached_tokens"] for r in reqs),
        "model_calls_per_request": mean(r["n_calls"] for r in reqs),
        "no_answer": sum(r["stopped"] != "answered" for r in reqs),
        "requests_reading_other_clients_state": sum(bool(reads_split(r)[0]) for r in reqs),
        "of_which_wrong": sum(bool(reads_split(r)[0]) and not r["correct"] for r in reqs),
        "model_s_per_request": mean(r["model_s"] for r in reqs),
        "exec_s_per_request": mean(r["exec_s"] for r in reqs),
        "early_answers": sum(any(t.get("early") for t in r["tool_calls"]) for r in reqs),
        "by_request": {rid: {"correct": sum(r["correct"] for r in reqs if r["req_id"].startswith(rid)),
                             "n": sum(r["req_id"].startswith(rid) for r in reqs)}
                       for rid in ("r1", "r2", "r3", "r4", "r5", "r6")},
        "runs": len(runs),
    }
    if reqs and reqs[0]["arm"] in ("coordinate", "long_coordinate", "per_task_modes", "per_task"):
        out["choices"] = modes_summary(reqs)
    return out


def common_items(a: list[dict], b: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep only the items (k, seed, client) both arms ran, so a difference is
    always paired: the top tier ran some arms on seeds 1 to 10 and others on
    seeds 1 to 3, and a comparison between them uses seeds 1 to 3 only."""
    key = lambda r: (r["k"], r["seed"], r["client"])  # noqa: E731
    both = {key(r) for r in a} & {key(r) for r in b}
    return [r for r in a if key(r) in both], [r for r in b if key(r) in both]


def compare(a: list[dict], b: list[dict]) -> dict:
    """b minus a, paired over the items (k, seed, client) both ran."""
    a, b = common_items(a, b)
    acc = bootstrap_diff(item_values(a, lambda r: float(r["correct"])), item_values(b, lambda r: float(r["correct"])))
    pa, pb = per_client(a), per_client(b)
    reload = bootstrap_diff({k: [v["reloads"]] for k, v in pa.items()}, {k: [v["reloads"]] for k, v in pb.items()})
    t = bootstrap_diff(item_values(a, lambda r: r["agent_s"]), item_values(b, lambda r: r["agent_s"]))
    cost = bootstrap_diff(item_values(a, lambda r: r["cost"]), item_values(b, lambda r: r["cost"]))
    return {"accuracy": acc, "reloads_per_client": reload, "agent_s_per_request": t, "cost_per_request": cost}


def state_caused_compare(a: list[dict], b: list[dict], review: dict) -> dict:
    a, b = common_items(a, b)
    f = lambda r: float(cause_of(r, review) in ("leak", "own_whatif"))  # noqa: E731
    return bootstrap_diff(item_values(a, f), item_values(b, f))


def analyse(phases=PHASES, runs_dir: Path | None = None, review_path: Path | None = None) -> dict:
    reqs, setups = load_runs(phases, runs_dir)
    review_path = review_path or (HERE / "review.json")
    review = json.loads(review_path.read_text()).get("wrong_answers", {}) if review_path.exists() else {}
    groups: dict[tuple, list] = defaultdict(list)
    for r in reqs:
        # the added robustness condition is kept apart as arm@condition
        arm = f"{r['arm']}@{r['cond']}" if r["cond"] in SEPARATE_CONDS else r["arm"]
        groups[(r["model"], arm, r["k"])].append(r)
    results: dict = {"groups": {}, "comparisons": {}, "ledger": None}
    def order(arm: str) -> tuple:
        base, _, cond = arm.partition("@")
        return (cond, ARM_ORDER.index(base))

    for (model, arm, k), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2], order(kv[0][1]))):
        results["groups"][f"{model}|{arm}|k{k}"] = summarise_group(rs, review)
    pairs = [("per_task", "coordinate"), ("fresh", "coordinate"), ("shared", "coordinate"), ("shared", "per_task"),
             ("fresh", "per_task"), ("long_shared", "long_coordinate"), ("per_task", "long_coordinate"),
             ("per_task", "per_task_modes"), ("coordinate", "per_task_modes"), ("shared", "long_shared"),
             ("per_task", "shared"), ("per_task", "long_shared"),
             ("per_task@initials", "coordinate@initials")]
    for model in sorted({m for m, _, _ in groups}):
        for k in sorted({kk for m, _, kk in groups if m == model}):
            for a, b in pairs:
                ga, gb = groups.get((model, a, k)), groups.get((model, b, k))
                if not ga or not gb:
                    continue
                c = compare(ga, gb)
                c["state_caused_rate"] = state_caused_compare(ga, gb, review)
                results["comparisons"][f"{model}|k{k}|{b}-{a}"] = c
    return results


# the January-2026 models of Amendment 2, superseded by Amendment 3: their cost
# is in the ledgers and the total, but not in today's-model spend, which the
# lead capped at $80 at the hand-over
SUPERSEDED_MODELS = {"anthropic/claude-opus-4.5", "openai/gpt-5.2"}


def ledger_cost() -> dict:
    """Every ledger under runs/ (the laptop's and the cloud machine's; the
    machine's are copied back as ledger-vm*.jsonl, so none is counted twice)."""
    from demos_common.ledger import summarise

    paths = sorted(str(p) for p in RUNS.glob("ledger*.jsonl"))
    if not paths:
        return {}
    s = summarise(paths)
    total = sum(d.get("cost", 0.0) for d in s.values())
    old = sum(d.get("cost", 0.0) for key, d in s.items() if key.split(" | ")[-1] in SUPERSEDED_MODELS)
    return {"files": [Path(p).name for p in paths], "by_model": s, "total": total,
            "todays_models": total - old, "superseded_models": old}


def decisions(results: dict) -> dict:
    """The plan's rules for "the idea winning", per model and K, from the paired
    comparisons: (1) coordinate reloads less than fresh, (2) has fewer
    state-caused wrong answers than shared ("not shown" if shared never went
    wrong from state), (3) is no more than 5 points below per_task (lower end of
    the interval above -0.05), (4) pulls ahead: beats per_task (interval above
    0), or holds at lower agent time or cost (interval below 0). The long arms
    take rules 2 to 4 with long_shared as both baselines."""
    comps, groups = results["comparisons"], results["groups"]
    out: dict = {}

    def c(m, k, pair):
        return comps.get(f"{m}|k{k}|{pair}")

    cells = sorted({(key.split("|")[0], int(key.split("|")[1][1:])) for key in comps})
    for m, k in cells:
        row: dict = {}
        for idea, fresh, shared, default in (("coordinate", "fresh", "shared", "per_task"),
                                             ("long_coordinate", None, "long_shared", "long_shared")):
            vs_d, vs_s = c(m, k, f"{idea}-{default}"), c(m, k, f"{idea}-{shared}")
            if not vs_d:
                continue
            r: dict = {"baseline": default}
            if fresh:
                vs_f = c(m, k, f"{idea}-{fresh}")
                r["1_avoids_forgetting"] = None if not vs_f else bool(vs_f["reloads_per_client"]["hi"] < 0)
            g_s = groups.get(f"{m}|{shared}|k{k}")
            if vs_s and g_s:
                r["2_avoids_leaking"] = ("not shown: the shared interpreter never went wrong from state"
                                         if g_s["state_caused_wrong"] == 0
                                         else bool(vs_s["state_caused_rate"]["hi"] < 0))
            a = vs_d["accuracy"]
            r["3_holds_against_default"] = bool(a["lo"] > -0.05)
            faster = vs_d["agent_s_per_request"]["hi"] < 0
            cheaper = vs_d["cost_per_request"]["hi"] < 0
            r["4_pulls_ahead"] = bool(a["lo"] > 0 or (a["lo"] > -0.05 and (faster or cheaper)))
            r["accuracy_diff_vs_default"] = a
            r["agent_s_diff_vs_default"] = vs_d["agent_s_per_request"]
            r["cost_diff_vs_default"] = vs_d["cost_per_request"]
            row[idea] = r
        if row:
            out.setdefault(m, {})[str(k)] = row
    return out


def seed_blocks(reqs: list[dict], model: str = "anthropic/claude-opus-5.5", k: int = 4) -> dict:
    """The top tier at K = 4 by block of seeds: 1 to 3 (run first) and 4 to 10
    (added at the hand-over, when the budget was restated), so the extension
    can be seen not to have changed the picture."""
    out: dict = defaultdict(dict)
    for name, seeds in (("seeds_1_3", range(1, 4)), ("seeds_4_10", range(4, 11))):
        for arm in ARM_ORDER:
            rs = [r for r in reqs if r["model"] == model and r["k"] == k and r["arm"] == arm
                  and r["cond"] == "main" and r["seed"] in seeds]
            if rs:
                out[name][arm] = {"correct": sum(r["correct"] for r in rs), "requests": len(rs),
                                  "cost_per_request": mean(r["cost"] for r in rs),
                                  "agent_s_per_request": mean(r["agent_s"] for r in rs)}
    return dict(out)


def full_design_cost(runs_dir: Path | None = None, seeds: int = 10) -> dict:
    """What the cheap tier's full design (the six main arms at K = 1, 2, 4, 8 and
    per_task_modes at K = 4, ten seeds) would cost on the mid and top tiers,
    from each model's measured cost per request in each (arm, K) that ran. A cell
    that did not run takes the same arm's cost at K = 4, scaled for K = 8 by the
    mid tier's measured K = 8 / K = 4 ratio for that arm."""
    runs_dir = runs_dir or RUNS
    folders = {"anthropic/claude-sonnet-5.5": "capable", "anthropic/claude-opus-5.5": "top"}
    arms = ARM_ORDER[:6]

    def cells(folder):
        out: dict = defaultdict(lambda: [0, 0.0])
        for p in (runs_dir / folder).glob("*.jsonl"):
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            if not rows or rows[-1].get("kind") != "run":
                continue
            for r in rows:
                if r.get("kind") == "request" and r.get("cond") == "main":
                    c = out[(r["arm"], r["k"])]
                    c[0] += 1
                    c[1] += r["cost"]
        return out

    mid = cells("capable")
    ratio = {a: (mid[(a, 8)][1] / mid[(a, 8)][0]) / (mid[(a, 4)][1] / mid[(a, 4)][0])
             for a in arms if mid.get((a, 8), [0])[0] and mid.get((a, 4), [0])[0]}
    out = {}
    for model, folder in folders.items():
        c = cells(folder)
        if not c:
            continue
        plan = [(a, k) for a in arms for k in (1, 2, 4, 8)] + [("per_task_modes", 4)]
        full, n_full = 0.0, 0
        for a, k in plan:
            n = seeds * 6 * k
            n_full += n
            if c.get((a, k), [0])[0]:
                unit = c[(a, k)][1] / c[(a, k)][0]
            elif c.get((a, 4), [0])[0]:
                unit = c[(a, 4)][1] / c[(a, 4)][0] * (ratio.get(a, 1.0) if k == 8 else 1.0)
            else:
                unit = None
            if unit is None:
                full = float("nan")
                break
            full += unit * n
        ran_n = sum(v[0] for v in c.values())
        ran_cost = sum(v[1] for v in c.values())
        out[model] = {"requests_run": ran_n, "cost_run": ran_cost, "full_requests": n_full,
                      "full_cost_estimate": full, "rest_cost_estimate": full - ran_cost}
    return out


def fmt_ci(x, lo, hi, pct=True) -> str:
    if pct:
        return f"{100 * x:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
    return f"{x:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def tables(results: dict) -> str:
    lines = []
    by_model = defaultdict(list)
    for key, g in results["groups"].items():
        model, arm, k = key.split("|")
        by_model[model].append((int(k[1:]), arm, g))
    for model, rows in by_model.items():
        lines.append(f"\n#### {model}\n")
        lines.append("| K | arm | correct | accuracy [95% CI] | wrong from state (leak / own what-if) | other wrong | "
                     "loads per client | agent min per run | $ per request | tokens per request (prompt / out) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for k, arm, g in sorted(rows, key=lambda x: (x[0], x[1].partition("@")[2], ARM_ORDER.index(x[1].partition("@")[0]))):
            w = g["wrong_by_cause"]
            lines.append(
                f"| {k} | {arm} | {g['correct']}/{g['requests']} | "
                f"{fmt_ci(g['accuracy'], *g['accuracy_ci'])} | {g['state_caused_wrong']} ({w['leak']} / "
                f"{w['own_whatif']}) | {w['other']} | {g['loads_per_client']:.2f} | "
                f"{g['agent_minutes_per_run']['mean']:.1f} | {g['cost_per_request']:.4f} | "
                f"{g['prompt_tokens_per_request'] / 1000:.1f}k / {g['completion_tokens_per_request'] / 1000:.2f}k |")
    lines.append("\n#### Paired differences (row arm minus baseline), bootstrap over clients\n")
    lines.append("| model | K | comparison | accuracy | state-caused wrong rate | reloads per client | "
                 "agent s per request | $ per request |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key, c in results["comparisons"].items():
        model, k, pair = key.split("|")
        a, s, rl, t, co = (c["accuracy"], c["state_caused_rate"], c["reloads_per_client"],
                           c["agent_s_per_request"], c["cost_per_request"])
        lines.append(
            f"| {model.split('/')[-1]} | {k[1:]} | {pair} | {100 * a['diff']:+.1f} pts [{100 * a['lo']:+.1f}, "
            f"{100 * a['hi']:+.1f}] | {100 * s['diff']:+.1f} pts [{100 * s['lo']:+.1f}, {100 * s['hi']:+.1f}] | "
            f"{rl['diff']:+.2f} [{rl['lo']:+.2f}, {rl['hi']:+.2f}] | {t['diff']:+.1f} [{t['lo']:+.1f}, {t['hi']:+.1f}] | "
            f"{co['diff']:+.4f} [{co['lo']:+.4f}, {co['hi']:+.4f}] |")
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()
    res = analyse(args.phases.split(","))
    res["decisions"] = decisions(res)
    res["top_tier_seed_blocks"] = seed_blocks(load_runs(["top"])[0])
    res["ledger"] = ledger_cost()
    res["full_design_cost"] = full_design_cost()
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(tables(res))
    if res["ledger"]:
        led = res["ledger"]
        print(f"\nLedger total: ${led['total']:.2f} (today's models ${led['todays_models']:.2f}, "
              f"superseded January-2026 models ${led['superseded_models']:.2f})")


if __name__ == "__main__":
    main()
