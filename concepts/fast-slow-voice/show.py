"""Print calls from a run for reading: what Sam heard, with timings, and what
each model call did.

  uv run python show.py runs/<condition>.jsonl [--arm relay] [--script s1-03] [--seed 0] [--calls]
"""

from __future__ import annotations

import argparse
import json

from fast_slow_voice.render import judge_transcript


def show(rec: dict, calls: bool, judge: dict | None = None) -> None:
    m = rec["metrics"]
    print(f"=== {rec['condition']} {rec['arm']} {rec['script']} seed {rec['seed']}  correct {m['n_correct']}/{m['n_wanted']} "
          f"stale {m['n_stale']} invented {len(m['invented'])} cost ${m['cost']:.4f} flags {m['flags']}")
    print(judge_transcript(rec))
    missed = [r for r in m["items"] if not r["withdrawn"] and not r["delivered"]]
    if missed:
        print("MISSED:", missed)
    if m["invented"]:
        print("INVENTED:", m["invented"])
    if m["actions"]:
        print("ACTIONS:", m["actions"])
    if judge is not None:
        j = judge.get("judgement") or {}
        print(f"JUDGE: naturalness {j.get('naturalness')} - {j.get('naturalness_reason')}")
        for k in ("double_replies", "contradictions", "wrong_or_invented", "stale"):
            for e in j.get(k, []):
                print(f"   {k}: {e}")
        print(f"   redundant_fillers: {j.get('redundant_fillers')}")
    if calls:
        print("--- calls")
        for c in rec["calls"]:
            tc = "; ".join(f"{t['name']}({json.dumps(t['args'])})" for t in c["tool_calls"])
            lab = c["meta"].get("label", "")
            trig = c["meta"].get("triggers") or c["meta"].get("events") or c["meta"].get("turn_id") or ""
            print(f"  #{c['id']:<3} {c['role']:5s} t0={c['t0']:7.2f} first={c['t_first']} done={c['t_done']} "
                  f"{'CANCELLED ' if c['cancelled'] else ''}P={c['prompt_tokens']} C={c['completion_tokens']} "
                  f"R={c['reasoning_tokens']} {lab} trig={trig}\n        text={c['text']!r}\n        tools={tc}")
        for n in rec["notes"]:
            print(f"  note t={n['t']}: {n['text']!r}")
        for u in rec["utterances"]:
            if u["dropped"] or u["t_start"] is None:
                print(f"  unspoken utt #{u['id']} ({u['source']}, dropped={u['dropped']}): {u['text']!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--arm")
    ap.add_argument("--script")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--calls", action="store_true")
    ap.add_argument("--judge", action="store_true", help="also print the judge's labels (runs/judge/<file>)")
    a = ap.parse_args()
    judged = {}
    if a.judge:
        from pathlib import Path
        jp = Path(a.path).parent / "judge" / Path(a.path).name
        for line in open(jp):
            j = json.loads(line)
            judged[(j["arm"], j["script"], j["seed"])] = j
    for line in open(a.path):
        rec = json.loads(line)
        if a.arm and rec["arm"] != a.arm or a.script and rec["script"] != a.script or a.seed is not None and rec["seed"] != a.seed:
            continue
        show(rec, a.calls, judged.get((rec["arm"], rec["script"], rec["seed"])) if a.judge else None)
        print()


if __name__ == "__main__":
    main()
