"""List wrong answers with what is needed to judge their cause.

    uv run python scripts/wrong.py runs/main --k 4 [--arm shared] [--model haiku]
"""

import argparse
import glob
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--model", default="")
    ap.add_argument("--code", type=int, default=700, help="characters of each call's code to show")
    a = ap.parse_args()
    n = 0
    for p in sorted(glob.glob(f"{a.folder}/*.jsonl")):
        if a.model not in p:
            continue
        rows = [json.loads(line) for line in open(p)]
        if not rows or rows[-1].get("kind") != "run":
            continue
        s = rows[0]
        if (a.k and s["k"] != a.k) or (a.arm and s["arm"] != a.arm):
            continue
        for r in rows:
            if r["kind"] != "request" or r["correct"]:
                continue
            n += 1
            ans = r["answer"] if r["answer_type"] != "regions" else json.dumps(r["answer"])
            tru = r["truth"] if r["answer_type"] != "regions" else json.dumps(r["truth"])
            print("=" * 110)
            print(f"{r['run_id']}#{r['pos']}  {r['client']} {r['req_id']}  stopped={r['stopped']}")
            print(f"  answer={ans}\n  truth ={tru}")
            print(f"  signature causes={r['causes']}  reads_other={r['reads_other_client']}  "
                  f"reads_own={r['reads_own_earlier']}")
            print(f"  request: {r['text'][r['text'].index(']') + 2:][:300]}")
            for t in r["tool_calls"]:
                if t["name"] == "run_python":
                    print(f"  -- {t.get('mode')} {t.get('session') or ''} err={'yes' if t.get('error') else 'no'}")
                    print("     " + (t.get("code") or "")[: a.code].replace("\n", "\n     "))
                elif t["name"] != "final_answer":
                    print(f"  -- {t['name']} {json.dumps(t.get('args'))}")
    print(f"\n{n} wrong answers")


if __name__ == "__main__":
    main()
