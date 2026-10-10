"""What a reduced run would have cost as the full design at its size.

The full design at a size is every arm, both change conditions, three seeds,
every task and every change. From a run's own records, the mean cost of a task
loop per arm and of a change loop per arm are multiplied out to those counts.
An arm the run left out is estimated from an arm it ran, by the ratio of their
costs in the Claude Haiku 5.5 runs (which ran every arm at every size).
Conditions the run left out are costed as the ones it ran.

    uv run python estimate_full_size.py --run top-opus
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
TASK_ARMS = ["pasted", "rule-skills", "rules-file", "long-context", "linked", "linked-in-context"]
CHANGE_ARMS = ["pasted", "rule-skills", "rules-file", "linked"]
STANDS_IN = {"linked-in-context": "long-context", "long-context": "linked-in-context", "rule-skills": "rules-file",
             "rules-file": "rule-skills", "pasted": "rules-file", "linked": "rules-file"}


def mean_cost(run: str, kind: str) -> dict[tuple[str, str], float]:
    acc: dict = defaultdict(list)
    for line in (HERE / "runs" / run / f"{kind}.jsonl").read_text().splitlines():
        r = json.loads(line)
        acc[(r["size"], r["arm"])].append(r["cost"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def estimate(run: str) -> dict:
    tasks, changes = mean_cost(run, "tasks"), mean_cost(run, "changes")
    h_tasks, h_changes = mean_cost("main-haiku", "tasks"), mean_cost("main-haiku", "changes")
    out = {}
    for size in sorted({s for s, _ in tasks}):
        ds = json.loads((HERE / "data" / f"{size}.json").read_text())
        per_task, per_change, filled = 0.0, 0.0, []
        for arm in TASK_ARMS:
            if (size, arm) in tasks:
                per_task += tasks[(size, arm)]
            else:
                ref = STANDS_IN[arm] if (size, STANDS_IN[arm]) in tasks else next(a for a in TASK_ARMS if (size, a) in tasks)
                per_task += tasks[(size, ref)] * h_tasks[(size, arm)] / h_tasks[(size, ref)]
                filled.append(arm)
        for arm in CHANGE_ARMS:
            if (size, arm) in changes:
                per_change += changes[(size, arm)]
            else:
                ref = next(a for a in CHANGE_ARMS if (size, a) in changes)
                per_change += changes[(size, ref)] * h_changes[(size, arm)] / h_changes[(size, ref)]
                filled.append(f"{arm} (changes)")
        full = (per_task * len(ds["tasks"]) + per_change * len(ds["changes"])) * 2 * 3
        out[size] = {"per_task_all_arms": per_task, "per_change_all_arms": per_change,
                     "full_design": full, "estimated_arms": filled}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    args = ap.parse_args()
    for run in args.run:
        for size, e in estimate(run).items():
            print(f"{run} {size}: the full design (every arm, both conditions, three seeds) would cost about "
                  f"${e['full_design']:,.0f}; one task in every arm ${e['per_task_all_arms']:.3f}, one change in "
                  f"every change arm ${e['per_change_all_arms']:.3f}; estimated from other arms: {e['estimated_arms'] or 'none'}")


if __name__ == "__main__":
    main()
