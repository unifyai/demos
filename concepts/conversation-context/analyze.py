"""Every number: results.json and the tables in the README's Results.

    uv run python analyze.py            # writes results.json, prints the tables

Reads runs/<condition>.jsonl (runs/discarded/ is never read) and runs/judge.jsonl.
Rates have Wilson intervals over runs; differences between arms are paired over
items (each item's seeds kept together), 10,000 bootstrap resamples.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from demos_common.ledger import summarise
from demos_common.stats import bootstrap_diff, mean, quantile, wilson

from conversation_context.grade import outcome
from judge import key as judge_key

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
ARM_ORDER = ["alone", "brief", "summary", "full", "full_naive", "single"]
LENGTH_ORDER = ["short", "medium", "long", "xlong"]


SKIPPED: list[str] = []  # lines caught mid-write by a copy from the cloud machine: skipped, never counted


def lines_of(path: Path) -> list[dict]:
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            SKIPPED.append(f"{path.name}:{i}")
    return out


def load() -> list[dict]:
    judged = {}
    if (RUNS / "judge.jsonl").exists():
        for j in lines_of(RUNS / "judge.jsonl"):
            if j.get("label"):
                judged[j["key"]] = j["label"]
    out = []
    for p in sorted(RUNS.glob("*.jsonl")):
        if "__" not in p.stem or p.stem.endswith(("__prefixes", "__errors")):
            continue
        for r in lines_of(p):
            r.pop("transcript", None)
            r.pop("calls_detail", None)
            o = outcome(r, judged.get(judge_key(r)))
            r.update(o)
            r["subset"] = r["kind"] if r["kind"] != "open" else ("open_present" if r["present"] else "open_away")
            out.append(r)
    return out


def rate(rs: list[dict], key: str) -> dict:
    k = sum(1 for r in rs if r[key])
    lo, hi = wilson(k, len(rs))
    return {"k": k, "n": len(rs), "rate": k / len(rs) if rs else float("nan"), "lo": lo, "hi": hi}


def by_item(rs: list[dict], key: str) -> dict[str, list[float]]:
    d: dict[str, list[float]] = defaultdict(list)
    for r in rs:
        d[r["item"]].append(float(r[key]))
    return d


def summary(rs: list[dict]) -> dict:
    labels = defaultdict(int)
    for r in rs:
        labels[r["label"]] += 1
    lat = [r["latency"] for r in rs]
    out = {
        "n": len(rs), "right": rate(rs, "right"), "correct": rate(rs, "correct"), "asked": rate(rs, "asked"),
        "wrong_unasked": rate(rs, "wrong_unasked"), "labels": dict(labels),
        "q_caller": mean([r["q_caller"] for r in rs]), "q_person": mean([r["q_person"] for r in rs]),
        "asked_person": mean([float(r["q_person"] > 0) for r in rs]),
        "person_answered": mean([float(any(q["answered"] for q in r["person_questions"])) for r in rs]),
        # where callers' questions were settled: by a caller from what it knew, or only after asking on
        "caller_questions_answered_in_place": mean([float(not q["escalated"]) for r in rs for q in r["questions"]])
        if any(r["questions"] for r in rs) else None,
        "cost": mean([r["cost"] for r in rs]), "latency": mean(lat), "latency_median": quantile(lat, 0.5),
        "calls": mean([r["calls"] for r in rs]), "prompt_tokens": mean([r["prompt_tokens"] for r in rs]),
        "handovers": mean([r["handovers"] for r in rs]), "step_limits": sum(1 for r in rs if r["step_limits"]),
    }
    opens = [r for r in rs if r["kind"] == "open"]
    if opens:
        out["guessed_open"] = rate(opens, "guessed")
    with_req = [r for r in rs if r["first_request_has_detail"] is not None and r["kind"] != "complete"]
    if with_req:
        out["request_named_detail"] = rate([{**r, "x": r["first_request_has_detail"]} for r in with_req], "x")
    return out


def paired(rs: list[dict], base: str, other: str, key: str) -> dict | None:
    a = by_item([r for r in rs if r["arm"] == other], key)
    b = by_item([r for r in rs if r["arm"] == base], key)
    if not a or not b:
        return None
    return bootstrap_diff(a, b)  # base minus other


def analyse(rows: list[dict]) -> dict:
    res: dict = {"conditions": {}}
    conds = defaultdict(list)
    for r in rows:
        conds[r["condition"]].append(r)
    for name, rs in sorted(conds.items()):
        arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in rs)]
        c: dict = {"grid": rs[0]["grid"], "model": rs[0]["model"], "requests": rs[0]["requests"],
                   "length": rs[0]["length"], "depth": rs[0]["depth"], "arms": {}, "vs_full": {}}
        for arm in arms:
            ra = [r for r in rs if r["arm"] == arm]
            c["arms"][arm] = {"all": summary(ra)}
            for sub in ("settled", "open", "open_present", "open_away", "complete"):
                rsub = [r for r in ra if (r["kind"] == sub if sub in ("settled", "open", "complete") else r["subset"] == sub)]
                if rsub:
                    c["arms"][arm][sub] = summary(rsub)
        if "full" in arms:
            for other in arms:
                if other == "full":
                    continue
                d: dict = {}
                for sub in ("all", "settled", "open", "open_present", "open_away", "complete"):
                    rsub = rs if sub == "all" else [r for r in rs if (r["kind"] == sub if sub in ("settled", "open", "complete") else r["subset"] == sub)]
                    d[sub] = {k: paired(rsub, "full", other, k) for k in ("right", "asked", "wrong_unasked", "cost", "latency")}
                c["vs_full"][other] = d
        # model-written requests: the four shared arms split by whether the first request named the detail
        if rs[0]["requests"] == "model":
            split = {}
            for named in (True, False):
                sub = [r for r in rs if r["kind"] == "settled" and r["first_request_has_detail"] is named]
                split["named" if named else "left_out"] = {arm: summary([r for r in sub if r["arm"] == arm])
                                                           for arm in arms if any(r["arm"] == arm for r in sub)}
            c["settled_by_request"] = split
        res["conditions"][name] = c
    res["ladder"] = ladder(rows)
    res["ledger"] = summarise([str(RUNS / "ledger.jsonl")]) if (RUNS / "ledger.jsonl").exists() else {}
    res["ledger_total"] = sum(v.get("cost", 0.0) for v in res["ledger"].values())
    res["cost_by_phase"] = cost_by_phase()
    # The lead (9 Oct, on handing over): the older models' runs don't count against the $80 cap.
    era = sum(r.get("cost") or 0.0 for r in lines_of(RUNS / "ledger.jsonl") if r.get("model") in ERA) \
        if (RUNS / "ledger.jsonl").exists() else 0.0  # every call to them, probes included
    res["spend_older_models"] = era
    res["spend_today_models"] = res["ledger_total"] - era
    return res


ERA = ("openai/o4-mini", "openai/gpt-4.1", "anthropic/claude-sonnet-4")
# G2 on haiku was started under plan revision 1 and stopped by amendment 1 (00:13 UTC) before any run
# finished; the grid of the same name ran afresh under amendment 2 (from 00:49 UTC).
STOPPED_BEFORE = {"G2__claude-haiku-5.5__terse__long__d3": datetime(2026, 10, 10, 0, 40, tzinfo=timezone.utc).timestamp()}


def cost_by_phase() -> dict[str, float]:
    """Every dollar in the ledger, by what it was spent on (a run's condition, the judge, the data...)."""
    out: dict[str, float] = defaultdict(float)
    path = RUNS / "ledger.jsonl"
    if not path.exists():
        return {}
    kept = {p.stem for p in RUNS.glob("*.jsonl") if "__" in p.stem and not p.stem.endswith(("__prefixes", "__errors"))}
    for r in lines_of(path):
        c = r.get("cost") or 0.0
        run = str((r.get("meta") or {}).get("run", ""))
        cond = run.split("|")[0]
        if r.get("tag") == "data":
            k = "data: filler pools"
        elif r.get("tag") == "probe" or cond.startswith("probe"):
            k = "probes"
        elif "judge" in str(r.get("tag")):
            k = "judge"
        elif r.get("model") in ERA:
            k = "era models (excluded)"
        elif cond in STOPPED_BEFORE and r.get("ts", 0) < STOPPED_BEFORE[cond]:
            k = f"stopped by amendment 1: {cond}"
        elif cond in kept and not cond.startswith(("sanity", "smoke")):
            k = cond
        elif cond.startswith(("sanity", "smoke")):
            k = "sanity and smoke (all rounds)"
        else:
            k = f"stopped or discarded: {cond or 'other'}"
        out[k] += c
    return {k: round(v, 4) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


TIER = {"claude-haiku-5.5": "cheap", "gpt-6-luna": "cheap", "claude-sonnet-5.5": "mid", "gpt-6.1-sol": "mid",
        "claude-opus-5.5": "top"}


def ladder(rows: list[dict]) -> dict:
    """The tier comparison (amendment 2): G2 at the short length, `full` minus `alone`, settled and open
    items, for each worker model; the cheap tier also restricted to exactly the items and seeds a
    stronger tier ran, so every tier is compared on the same requests."""
    g2s = [r for r in rows if r["grid"] == "G2" and r["length"] == "short" and r["kind"] in ("settled", "open")
           and r["arm"] in ("alone", "full")]
    by_worker: dict[str, list[dict]] = defaultdict(list)
    for r in g2s:
        by_worker[r["worker_model"].split("/")[-1] + ("-" + r["worker_reasoning"] if r.get("worker_reasoning") else "")].append(r)
    out: dict = {}

    def block(rs: list[dict]) -> dict:
        d: dict = {"runs_per_arm": sum(1 for r in rs if r["arm"] == "full")}
        for sub in ("all", "settled", "open_present", "open_away"):
            ss = rs if sub == "all" else [r for r in rs if (r["kind"] == sub if sub == "settled" else r["subset"] == sub)]
            d[sub] = {arm: rate([r for r in ss if r["arm"] == arm], "right") for arm in ("alone", "full")}
            d[sub]["full_minus_alone"] = paired(ss, "full", "alone", "right")
        d["asked"] = {arm: rate([r for r in rs if r["arm"] == arm], "asked") for arm in ("alone", "full")}
        d["wrong_unasked"] = {arm: rate([r for r in rs if r["arm"] == arm], "wrong_unasked") for arm in ("alone", "full")}
        d["cost"] = {arm: mean([r["cost"] for r in rs if r["arm"] == arm]) for arm in ("alone", "full")}
        d["latency"] = {arm: mean([r["latency"] for r in rs if r["arm"] == arm]) for arm in ("alone", "full")}
        d["q_caller"] = {arm: mean([r["q_caller"] for r in rs if r["arm"] == arm]) for arm in ("alone", "full")}
        return d

    for w, rs in by_worker.items():
        out[w] = {"tier": TIER.get(w.split("-high")[0], "?"), "own": block(rs)}
    cheap = by_worker.get("claude-haiku-5.5", [])
    for w, rs in by_worker.items():
        if w == "claude-haiku-5.5" or not cheap:
            continue
        keys = {(r["item"], r["seed"]) for r in rs}
        out[w]["cheap_on_same_items"] = block([r for r in cheap if (r["item"], r["seed"]) in keys])
    return out


def pct(x: dict | float | None, digits: int = 0) -> str:
    if x is None:
        return "-"
    if isinstance(x, dict):
        return f"{100 * x['rate']:.{digits}f}% [{100 * x['lo']:.0f}, {100 * x['hi']:.0f}]"
    return f"{100 * x:.{digits}f}%"


def diff(d: dict | None, scale: float = 100, unit: str = "") -> str:
    if not d:
        return "-"
    return f"{scale * d['diff']:+.{0 if scale == 100 else 4}f}{unit} [{scale * d['lo']:+.{0 if scale == 100 else 4}f}, {scale * d['hi']:+.{0 if scale == 100 else 4}f}]"


def tables(res: dict) -> str:
    out = []
    for name, c in res["conditions"].items():
        out.append(f"\n#### {name}\n")
        out.append("| arm | right (all) | settled correct | open, Maya there: correct | open, Maya away: says so | complete correct | asked (settled) | asked (open) | wrong, unasked (open) | $ per run | latency s | calls |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm, s in c["arms"].items():
            g = lambda sub, key: pct(s[sub][key]) if sub in s else "-"  # noqa: E731
            out.append(f"| {arm} | {pct(s['all']['right'])} | {g('settled', 'correct')} | {g('open_present', 'correct')} | "
                       f"{g('open_away', 'right')} | {g('complete', 'correct')} | {g('settled', 'asked')} | {g('open', 'asked')} | "
                       f"{g('open', 'wrong_unasked')} | {s['all']['cost']:.4f} | {s['all']['latency']:.1f} | {s['all']['calls']:.1f} |")
        if c["vs_full"]:
            out.append("\nfull minus each arm, paired over items (points, or $ and s):\n")
            out.append("| vs | right (all) | settled right | open right | asked (settled) | asked (open) | wrong unasked (open) | $ per run | latency s |")
            out.append("|---|---|---|---|---|---|---|---|---|")
            for other, d in c["vs_full"].items():
                out.append(f"| {other} | {diff(d['all']['right'])} | {diff(d['settled']['right'])} | {diff(d['open']['right'])} | "
                           f"{diff(d['settled']['asked'])} | {diff(d['open']['asked'])} | {diff(d['open']['wrong_unasked'])} | "
                           f"{diff(d['all']['cost'], 1, '')} | {diff(d['all']['latency'], 1, '')} |")
        if "settled_by_request" in c:
            out.append("\nsettled items by whether the first request named the detail:\n")
            out.append("| first request | arm | runs | correct | asked |")
            out.append("|---|---|---|---|---|")
            for k, arms in c["settled_by_request"].items():
                for arm, s in arms.items():
                    out.append(f"| {k} | {arm} | {s['n']} | {pct(s['correct'])} | {pct(s['asked'])} |")
    lad = res.get("ladder") or {}
    if lad:
        out.append("\n#### Tier ladder: G2 at the short length, `full` minus `alone` (right outcome)\n")
        out.append("| workers | tier | items | runs per arm | alone | full | full minus alone | settled | open, Maya there | open, Maya away | asked alone / full | $ per run alone / full | s per run alone / full |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for w, d in sorted(lad.items(), key=lambda kv: ["cheap", "mid", "top", "?"].index(kv[1]["tier"])):
            for label, b in (("own", d["own"]),) + ((("haiku on the same items", d["cheap_on_same_items"]),) if "cheap_on_same_items" in d else ()):
                name = w if label == "own" else "claude-haiku-5.5 (same items)"
                out.append(f"| {name} | {d['tier'] if label == 'own' else 'cheap'} | {'all 48' if b['runs_per_arm'] > 100 else 'subset'} | {b['runs_per_arm']} | "
                           f"{pct(b['all']['alone'])} | {pct(b['all']['full'])} | {diff(b['all']['full_minus_alone'])} | "
                           f"{diff(b['settled']['full_minus_alone'])} | {diff(b['open_present']['full_minus_alone'])} | "
                           f"{diff(b['open_away']['full_minus_alone'])} | {pct(b['asked']['alone'])} / {pct(b['asked']['full'])} | "
                           f"{b['cost']['alone']:.4f} / {b['cost']['full']:.4f} | {b['latency']['alone']:.0f} / {b['latency']['full']:.0f} |")
    return "\n".join(out)


def main() -> None:
    rows = load()
    res = analyse(rows)
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(tables(res))
    print(f"\nledger total ${res['ledger_total']:.2f}; today's models ${res['spend_today_models']:.2f}; "
          f"older models (excluded, outside the cap) ${res['spend_older_models']:.2f}")
    for k, v in res["cost_by_phase"].items():
        print(f"  {k}: ${v:.2f}")
    if "--quiet" not in sys.argv:
        unj = sum(1 for r in rows if r["label"] == "unjudged")
        print(f"{len(rows)} runs, {unj} unjudged")
    if SKIPPED:
        print(f"skipped {len(SKIPPED)} incomplete line(s): {', '.join(SKIPPED[:5])}")


if __name__ == "__main__":
    main()
