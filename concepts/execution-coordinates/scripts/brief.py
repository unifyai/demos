"""One line per run (and per wrong answer) for a quick look: uv run python scripts/brief.py runs/smoke"""
import glob
import json
import sys

for p in sorted(glob.glob(f"{sys.argv[1]}/*.jsonl")):
    rows = [json.loads(line) for line in open(p)]
    reqs = [r for r in rows if r["kind"] == "request"]
    if not reqs:
        continue
    done = rows[-1]["kind"] == "run"
    modes = {}
    for r in reqs:
        for t in r["tool_calls"]:
            if t["name"] == "run_python":
                k = f"{t.get('mode')}:{t.get('session')}" if t.get("args", {}).get("mode") or t.get("args", {}).get("session") else None
                if k:
                    modes[k] = modes.get(k, 0) + 1
            elif t["name"] != "final_answer":
                modes[t["name"]] = modes.get(t["name"], 0) + 1
    print(f"{p.split('/')[-1].split('__', 1)[1][:-6]:<36} {'done' if done else 'RUNNING'} "
          f"correct {sum(r['correct'] for r in reqs)}/{len(reqs)} loads {sum(r['loads'] for r in reqs)} "
          f"scores {sum(r['scores'] for r in reqs)} calls {sum(r['n_calls'] for r in reqs)} "
          f"min {sum(r['agent_s'] for r in reqs) / 60:.1f} ${sum(r['cost'] for r in reqs):.3f} "
          f"other-reads {sum(bool(r['reads_other_client']) for r in reqs)}  {modes if modes else ''}")
    for r in reqs:
        if not r["correct"]:
            a = r["answer"] if r["answer_type"] != "regions" else "{..}"
            t = r["truth"] if r["answer_type"] != "regions" else "{..}"
            print(f"     wrong [{r['pos']}] {r['client']} {r['req_id']} answer={a} truth={t} causes={r['causes']} "
                  f"other={r['reads_other_client']} own={r['reads_own_earlier']} stopped={r['stopped']}")
