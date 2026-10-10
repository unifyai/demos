"""Print what needs reading by hand.

    uv run python inspect_runs.py judge --run main-haiku [--sample 40]
        every version judgment that is not "new only", and a seeded sample of
        those that are, each with the text judged
    uv run python inspect_runs.py failures --run main-haiku --size L --arm linked [--what rules|procedure|change]
        the transcripts of tasks that missed a rule or the procedure, or of
        changes that were not completed
"""

from __future__ import annotations

import argparse
import json
import random

from analyze import change_rows, datasets, load, score_tasks, state_rows


def show_judgments(run: str, sample: int) -> None:
    data = load(run)
    dss = datasets()
    rules = {rid: r for ds in dss.values() for rid, r in ds["rules"].items()}
    vs = [j for j in data["judgments"].values() if j["kind"] == "version"]
    off = [j for j in vs if not (j["states_new"] and not j["states_old"])]
    on = [j for j in vs if j["states_new"] and not j["states_old"]]
    rng = random.Random(0)
    picked = rng.sample(on, min(sample, len(on)))
    print(f"{len(vs)} version judgments: {len(on)} new only, {len(off)} otherwise; showing all of the latter and {len(picked)} of the former\n")
    for j in off + picked:
        ch = rules[j["rule"]]["change"]
        print(f"=== {j['rule']} new={j['states_new']} old={j['states_old']} | before: {ch['old']} | after: {ch['new']}")
        text = j["text"]
        title = rules[j["rule"]]["title"]
        at = text.find(title)
        print("   " + (text[at:] if at > 0 else text).replace("\n", " ")[:600])


def show_failures(run: str, size: str, arm: str, what: str, limit: int) -> None:
    data = load(run)
    dss = datasets()
    ds = dss[size]
    if what == "change":
        srows = [s for s in state_rows(data) if s["size"] == size and s["arm"] == arm and s["changed"] and s["class"] != "current"]
        keys = {(s["condition"], s["seed"], s["rule"]) for s in srows}
        for c in data["changes"]:
            if c["size"] == size and c["arm"] == arm and (c["condition"], c["seed"], c["change"]) in keys:
                bad = sorted({s["procedure"] for s in srows if (s["condition"], s["seed"], s["rule"]) == (c["condition"], c["seed"], c["change"])})
                print(f"===== {arm} {c['condition']} s{c['seed']} change {c['change']} | not current for: {bad}")
                print("  MESSAGE:", c["message"])
                for t in c["tool_calls"]:
                    print("  CALL", t["name"], json.dumps(t["args"], ensure_ascii=False)[:300])
                    print("    ->", t["result"][:300].replace("\n", " | "))
                print("  ANSWER", (c["answer_text"] or "")[:500].replace("\n", " "))
        return
    recs = {(r["condition"], r["seed"], r["task"]): r for r in data["tasks"] if r["size"] == size and r["arm"] == arm}
    n = 0
    for s in score_tasks(data, dss):
        if s["size"] != size or s["arm"] != arm:
            continue
        bad = (not s["function_found"]) if what == "procedure" else any(v != "current" for v in s["rules"].values())
        if not bad:
            continue
        n += 1
        if n > limit:
            break
        r = recs[(s["condition"], s["seed"], s["task"])]
        task = next(t for t in ds["tasks"] if t["id"] == r["task"])
        print(f"===== {arm} {s['condition']} s{s['seed']} {s['task']} expected {r['procedure']} {r['rules_expected']} "
              f"(hidden={r['hidden']}) | found={s['function_found']} rules={s['rules']} extras={s['extras']}")
        print("  TASK:", task["text"])
        for t in r["tool_calls"]:
            print("  CALL", t["name"], json.dumps(t["args"], ensure_ascii=False)[:200], "->", t["result"][:200].replace("\n", " | "))
        print("  ANSWER", (r["answer_text"] or "")[:600].replace("\n", " "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["judge", "failures"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--size", default="L")
    ap.add_argument("--arm", default="linked")
    ap.add_argument("--kind", default="rules", choices=["rules", "procedure", "change"])
    ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()
    if a.what == "judge":
        show_judgments(a.run, a.sample)
    else:
        show_failures(a.run, a.size, a.arm, a.kind, a.limit)


if __name__ == "__main__":
    main()
