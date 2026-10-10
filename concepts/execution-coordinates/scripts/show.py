"""Print a run's transcripts for reading.

    uv run python scripts/show.py runs/smoke/<run>.jsonl [--req 3] [--max 1500]
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--req", type=int, default=None, help="only this arrival position")
    ap.add_argument("--max", type=int, default=1500, help="characters of each tool result to show")
    ap.add_argument("--wrong", action="store_true", help="only wrong answers")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.path)]
    setup = rows[0]
    print(f"# {setup['run_id']}  arm={setup['arm']} design={setup['design']} k={setup['k']} seed={setup['seed']}")
    for r in rows:
        if r["kind"] != "request" or (a.req is not None and r["pos"] != a.req) or (a.wrong and r["correct"]):
            continue
        print("\n" + "=" * 100)
        print(f"[{r['pos']}] {r['client']} {r['req_id']}  correct={r['correct']}  answer={r['answer']}  "
              f"truth={r['truth']}  causes={r['causes']}")
        print(f"     calls={r['n_calls']} loads={r['loads']} scores={r['scores']} agent_s={r['agent_s']} "
              f"cost=${r['cost']:.4f} reads_other={r['reads_other_client']} reads_own={r['reads_own_earlier']}")
        for m in r["messages"]:
            if m["role"] == "user":
                print(f"\n--- USER: {m['content']}")
            elif m["role"] == "assistant":
                if m.get("content"):
                    print(f"\n--- ASSISTANT: {m['content']}")
                for tc in m.get("tool_calls") or []:
                    fn = tc["function"]
                    try:
                        args = json.loads(fn["arguments"])
                    except Exception:
                        args = {"raw": fn["arguments"]}
                    code = args.pop("code", None)
                    print(f"\n--- CALL {fn['name']} {json.dumps(args)}")
                    if code:
                        print(code)
            elif m["role"] == "tool":
                c = m["content"]
                print(f"--- RESULT: {c[:a.max]}{' ...' if len(c) > a.max else ''}")


if __name__ == "__main__":
    main()
