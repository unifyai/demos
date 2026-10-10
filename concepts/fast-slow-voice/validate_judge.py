"""Check the judge by hand.

  uv run python validate_judge.py sample --cond anthropic-realistic-d0 --per-arm 4
      writes runs/judge_validation/transcripts.md (calls under blind ids, exactly
      what the judge read) and mapping.json (id -> arm, script, seed; not read
      while labelling)
  (label by hand into runs/judge_validation/hand_labels.json:
      {"V01": {"naturalness": 3, "double_replies": 0, "contradictions": 0, "stale": 1, "wrong_or_invented": 0}, ...})
  uv run python validate_judge.py compare --cond anthropic-realistic-d0
      agreement between the hand labels and the judge's
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from demos_common.stats import mean

from fast_slow_voice.judge import item_brief
from fast_slow_voice.render import judge_transcript
from fast_slow_voice.script import Script

HERE = Path(__file__).parent
OUT = HERE / "runs" / "judge_validation"
KEYS = ["double_replies", "contradictions", "stale", "wrong_or_invented"]


def scripts() -> dict[str, Script]:
    return {s.id: s for s in (Script.from_json(d) for d in json.loads((HERE / "data" / "scripts-1.json").read_text()))}


def sample(cond: str, per_arm: int, seed: int) -> None:
    recs = [json.loads(l) for l in (HERE / "runs" / f"{cond}.jsonl").read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    by_arm = defaultdict(list)
    for r in recs:
        by_arm[r["arm"]].append(r)
    chosen = []
    for arm in sorted(by_arm):
        chosen += rng.sample(by_arm[arm], min(per_arm, len(by_arm[arm])))
    rng.shuffle(chosen)
    OUT.mkdir(parents=True, exist_ok=True)
    ss = scripts()
    mapping, parts = {}, []
    for k, r in enumerate(chosen, 1):
        vid = f"V{k:02d}"
        mapping[vid] = {"arm": r["arm"], "script": r["script"], "seed": r["seed"]}
        s = ss[r["script"]]
        parts.append(f"## {vid}\n\n{item_brief(s, s.world(r['size']), r)}\n\n```\n{judge_transcript(r)}\n```\n")
    (OUT / "transcripts.md").write_text(f"# Blinded calls from {cond}\n\n" + "\n".join(parts))
    (OUT / "mapping.json").write_text(json.dumps({"condition": cond, "ids": mapping}, indent=1))
    print(f"wrote {len(chosen)} calls to {OUT / 'transcripts.md'}")


def kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def compare(cond: str) -> dict:
    mp = json.loads((OUT / "mapping.json").read_text())["ids"]
    hand = json.loads((OUT / "hand_labels.json").read_text())
    judged = {}
    for l in (HERE / "runs" / "judge" / f"{cond}.jsonl").read_text().splitlines():
        j = json.loads(l)
        if j.get("judgement"):
            judged[(j["arm"], j["script"], j["seed"])] = j["judgement"]
    # the hand labels were made on the calls as first recorded; where a bug fix later replaced a call
    # (V30, silent-marker bug), compare with the judge's label of that same first version
    replaced = HERE / "runs" / "discarded" / "judge-replaced.jsonl"
    if replaced.exists():
        for l in replaced.read_text().splitlines():
            j = json.loads(l)
            if j.get("condition") == cond and j.get("judgement"):
                judged[(j["arm"], j["script"], j["seed"])] = j["judgement"]
    rows = []
    for vid, h in sorted(hand.items()):
        m = mp[vid]
        j = judged[(m["arm"], m["script"], m["seed"])]
        rows.append({"id": vid, "arm": m["arm"], "hand": h, "judge": {"naturalness": j["naturalness"], **{k: len(j[k]) for k in KEYS}}})
    hn = [r["hand"]["naturalness"] for r in rows]
    jn = [r["judge"]["naturalness"] for r in rows]
    out = {
        "n": len(rows),
        "naturalness": {"spearman": spearman(hn, jn), "exact": mean([a == b for a, b in zip(hn, jn)]),
                        "within_1": mean([abs(a - b) <= 1 for a, b in zip(hn, jn)]), "judge_minus_hand": mean([b - a for a, b in zip(hn, jn)])},
        "events": {},
        "by_arm": {},
        "rows": rows,
    }
    for k in KEYS:
        a = [int(r["hand"][k] > 0) for r in rows]
        b = [int(r["judge"][k] > 0) for r in rows]
        tp = sum(x and y for x, y in zip(a, b))
        out["events"][k] = {"hand_any": sum(a), "judge_any": sum(b), "both": tp, "agree": mean([x == y for x, y in zip(a, b)]),
                            "kappa": kappa(a, b) if 0 < sum(a) + sum(b) else None,
                            "hand_total": sum(r["hand"][k] for r in rows), "judge_total": sum(r["judge"][k] for r in rows)}
    arms = sorted({r["arm"] for r in rows})
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        out["by_arm"][arm] = {"hand_naturalness": mean([r["hand"]["naturalness"] for r in rs]),
                              "judge_naturalness": mean([r["judge"]["naturalness"] for r in rs]), "n": len(rs)}
    (OUT / "comparison.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sample", "compare"])
    ap.add_argument("--cond", default="anthropic-realistic-d0")
    ap.add_argument("--per-arm", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.what == "sample":
        sample(a.cond, a.per_arm, a.seed)
    else:
        compare(a.cond)


if __name__ == "__main__":
    main()
