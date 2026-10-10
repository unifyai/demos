"""Build a dataset of one size from the catalog, deterministically from a seed.

    python -m linked_libraries.generate            # writes data/<size>.json for every size

A dataset holds the chosen folders and their procedures, the chosen rules and
their links (restricted to the chosen procedures), the rules that change, in
order, with the direct request and the in-passing message for each, and the
tasks, each with the procedure it needs and every rule that governs it.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from . import catalog

# folders, rules, changes, tasks
SIZES: dict[str, dict[str, int]] = {
    "sanity": {"folders": 2, "rules": 2, "changes": 1, "tasks": 4},
    "S": {"folders": 5, "rules": 5, "changes": 4, "tasks": 24},
    "M": {"folders": 10, "rules": 10, "changes": 6, "tasks": 48},
    "L": {"folders": 30, "rules": 25, "changes": 10, "tasks": 72},
}
DATA_SEED = 0
ROOT = Path(__file__).resolve().parents[2]


def _rule_spans(folders: set[str]) -> dict[str, tuple[int, int]]:
    """For each rule, (folders spanned, procedures governed) within these folders."""
    span: dict[str, set[str]] = defaultdict(set)
    count: dict[str, int] = defaultdict(int)
    for p in catalog.PROCEDURES:
        if p["folder"] in folders:
            for r in p["rules"]:
                span[r].add(p["folder"])
                count[r] += 1
    return {r: (len(span[r]), count[r]) for r in catalog.RULES if count[r]}


def choose_folders(n: int, rng: random.Random) -> list[str]:
    """The folder set (of size n) on which the most rules span several folders."""
    names = sorted(catalog.FOLDERS)
    if n >= len(names):
        return names
    best, best_score = None, None
    for _ in range(4000):
        cand = set(rng.sample(names, n))
        spans = _rule_spans(cand)
        multi = [(s, c) for s, c in spans.values() if s >= 2]
        score = (len(multi), sum(s for s, _ in multi), sum(c for _, c in multi))
        if best_score is None or score > best_score:
            best, best_score = cand, score
    return sorted(best)


def choose_rules(folders: list[str], n: int, rng: random.Random) -> list[str]:
    """The n rules spanning the most of these folders (ties: more procedures, then seeded)."""
    spans = _rule_spans(set(folders))
    tiebreak = {r: rng.random() for r in sorted(spans)}
    ranked = sorted(spans, key=lambda r: (-spans[r][0], -spans[r][1], tiebreak[r]))
    return sorted(ranked[:n])


def build(size: str, seed: int = DATA_SEED) -> dict:
    cfg = SIZES[size]
    rng = random.Random(f"{size}-{seed}")
    folders = choose_folders(cfg["folders"], rng)
    rules = choose_rules(folders, cfg["rules"], rng)
    rule_set = set(rules)

    procedures = {}
    for p in catalog.PROCEDURES:
        if p["folder"] in folders:
            procedures[p["name"]] = {
                "folder": p["folder"], "params": p["params"], "returns": p["returns"],
                "doc": p["doc"], "task": p["task"], "hidden": p["hidden"],
                "rules": [r for r in p["rules"] if r in rule_set],
            }

    spans = _rule_spans(set(folders))
    multi = [r for r in rules if spans[r][0] >= 2]
    single = [r for r in rules if spans[r][0] < 2]
    rng.shuffle(multi)
    rng.shuffle(single)
    changed = (multi + single)[: cfg["changes"]]  # rules that span several folders first
    rng.shuffle(changed)  # the order the changes arrive in

    changes = []
    for i, rid in enumerate(changed):
        governed = sorted(n for n, p in procedures.items() if rid in p["rules"])
        via = rng.choice(governed)
        ch = catalog.RULES[rid]["change"]
        changes.append({
            "order": i, "rule": rid, "passing_procedure": via,
            "direct_message": ch["direct"],
            "passing_message": procedures[via]["task"] + " " + ch["passing"],
        })

    tasks = choose_tasks(procedures, changed, cfg["tasks"], {c["passing_procedure"] for c in changes}, rng)

    return {
        "size": size, "data_seed": seed,
        "folders": {f: catalog.FOLDERS[f] for f in folders},
        "procedures": procedures,
        "rules": {r: {k: catalog.RULES[r][k] for k in ("title", "applies", "text", "change")} for r in rules},
        "changes": changes,
        "tasks": tasks,
    }


def choose_tasks(procedures: dict, changed: list[str], n: int, avoid: set[str], rng: random.Random) -> list[dict]:
    """Half the tasks use a procedure a changed rule governs (spread over the
    changed rules and their folders), a quarter one governed only by rules that
    did not change, and a quarter one governed by no rule."""
    changed_set = set(changed)
    a = sorted(x for x, p in procedures.items() if changed_set & set(p["rules"]))
    b = sorted(x for x, p in procedures.items() if p["rules"] and not changed_set & set(p["rules"]))
    c = sorted(x for x, p in procedures.items() if not p["rules"])
    want_a = min(len(a), n // 2)
    want_b = min(len(b), (n - want_a) // 2)
    want_c = min(len(c), n - want_a - want_b)
    shortfall = n - want_a - want_b - want_c
    if shortfall:  # fill from whichever stratum has procedures left
        extra_b = min(len(b) - want_b, shortfall)
        want_b += extra_b
        want_c += min(len(c) - want_c, shortfall - extra_b)

    # stratum A: round-robin over the changed rules, preferring a folder not yet used for that rule
    picked: list[str] = []
    used_folders: dict[str, set[str]] = defaultdict(set)
    order = list(changed)
    while len(picked) < want_a:
        progress = False
        for rid in order:
            if len(picked) >= want_a:
                break
            pool = [x for x in a if x not in picked and rid in procedures[x]["rules"]]
            if not pool:
                continue
            fresh = [x for x in pool if procedures[x]["folder"] not in used_folders[rid] and x not in avoid]
            fresh = fresh or [x for x in pool if procedures[x]["folder"] not in used_folders[rid]]
            fresh = fresh or [x for x in pool if x not in avoid] or pool
            x = rng.choice(fresh)
            picked.append(x)
            for r in procedures[x]["rules"]:
                used_folders[r].add(procedures[x]["folder"])
            progress = True
        if not progress:
            break
    picked += rng.sample(b, want_b) + rng.sample(c, want_c)
    rng.shuffle(picked)
    out = []
    for i, x in enumerate(picked, 1):
        p = procedures[x]
        out.append({"id": f"t{i:03d}", "procedure": x, "text": p["task"], "rules": list(p["rules"]),
                    "changed_rules": [r for r in p["rules"] if r in changed_set], "hidden": p["hidden"]})
    return out


def _largest_remainder(sizes: dict, n: int) -> dict:
    total = sum(sizes.values())
    quota = {k: n * v / total for k, v in sizes.items()}
    take = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - take[k]), str(k)))[: n - sum(take.values())]:
        take[k] += 1
    return take


def task_subset(ds: dict, n: int, seed: int = 0) -> list[str]:
    """A seeded subset of n task ids that keeps the task set's make-up: the same
    shares under a changed rule, under unchanged rules only and under none, and
    within each, hidden and visible procedures in proportion."""
    rng = random.Random(f"subset-{ds['size']}-{n}-{seed}")
    groups: dict[str, dict[bool, list[str]]] = defaultdict(lambda: defaultdict(list))
    for t in ds["tasks"]:
        kind = "changed" if t["changed_rules"] else ("unchanged" if t["rules"] else "none")
        groups[kind][t["hidden"]].append(t["id"])
    per_kind = _largest_remainder({k: sum(len(v) for v in g.values()) for k, g in groups.items()}, n)
    out: list[str] = []
    for kind in sorted(groups):
        per_hidden = _largest_remainder({h: len(v) for h, v in groups[kind].items()}, per_kind[kind])
        for h in sorted(groups[kind]):
            out += rng.sample(sorted(groups[kind][h]), per_hidden[h])
    return sorted(out)


def describe(ds: dict) -> str:
    links = sum(len(p["rules"]) for p in ds["procedures"].values())
    spans = defaultdict(set)
    for p in ds["procedures"].values():
        for r in p["rules"]:
            spans[r].add(p["folder"])
    multi = sum(1 for r in ds["rules"] if len(spans[r]) >= 2)
    t = ds["tasks"]
    return (f"{ds['size']}: {len(ds['folders'])} folders, {len(ds['procedures'])} procedures "
            f"({sum(p['hidden'] for p in ds['procedures'].values())} hidden), {len(ds['rules'])} rules "
            f"({multi} spanning several folders), {links} links, {len(ds['changes'])} changes "
            f"({', '.join(c['rule'] for c in ds['changes'])}), {len(t)} tasks "
            f"({sum(bool(x['changed_rules']) for x in t)} under a changed rule, "
            f"{sum(bool(x['rules']) and not x['changed_rules'] for x in t)} under unchanged rules only, "
            f"{sum(not x['rules'] for x in t)} under none; {sum(x['hidden'] for x in t)} hidden; "
            f"{sum(len(x['rules']) for x in t)} task-rule pairs)")


def main() -> None:
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "--subset":  # print a stratified subset: --subset L 54
        ds = json.loads((ROOT / "data" / f"{sys.argv[2]}.json").read_text())
        print(" ".join(task_subset(ds, int(sys.argv[3]))))
        return
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    catalog.check()
    for size in SIZES:
        ds = build(size)
        (out / f"{size}.json").write_text(json.dumps(ds, indent=1, ensure_ascii=False) + "\n")
        print(describe(ds))


if __name__ == "__main__":
    main()
