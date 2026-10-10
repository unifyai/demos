"""Print one run's transcript in a readable form, for reading runs before trusting numbers.

  uv run python show.py --records runs/records.jsonl --session item03-n500 --arm by_reference --seed 0

Shows the answer against the truth, the script's beats (what was said at the
moments that matter), every model step (tool calls, text, reasoning summary),
the frames looked at, and, by arm, the captions of the frames that show the
picks or the pipeline's procedure.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=HERE / "runs" / "records.jsonl")
    ap.add_argument("--session", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--chars", type=int, default=1500)
    a = ap.parse_args()
    rec = None
    text = gzip.decompress(a.records.read_bytes()).decode() if a.records.suffix == ".gz" else a.records.read_text()
    for line in text.splitlines():
        r = json.loads(line)
        if r["session"] == a.session and r["arm"] == a.arm and r["seed"] == a.seed and (not a.model or r["model"] == a.model):
            rec = r
    if rec is None:
        raise SystemExit("no such run")
    sess = json.loads((HERE / "data" / "sessions" / f"{a.session}.json").read_text())
    print(f"{rec['model']} {a.session} {a.arm} seed {a.seed}: {'RIGHT' if rec['correct'] else 'WRONG'}")
    print(f"  answer {rec['answer']}\n  truth  {rec['truth']}\n  error  {rec['error']}")
    print("script beats:")
    for f in sess["frames"]:
        if f["role"] != "noise":
            print(f"  {f['id']} [{int(f['t']) // 60:02d}:{int(f['t']) % 60:02d}] {f['role']:<10} {f['task'] or '':<9} {f['spec']['app']}/{f['spec'].get('view', '')}: {f['said']!r}")
    reasoning = rec["artefacts"].get("reasoning", [])
    step = 0
    for m in rec["transcript"][2:]:
        if m["role"] == "assistant":
            calls = [(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls", [])]
            print(f"step {step}: text {(m.get('content') or '')[:400]!r}")
            for name, args in calls:
                print(f"  -> {name}({args[:300]})")
            if step < len(reasoning) and reasoning[step]:
                print(f"  reasoning: {reasoning[step][:a.chars]!r}")
            step += 1
        elif m["role"] == "tool":
            print(f"  <- {m['content'][:200]}")
        elif m["role"] == "user" and isinstance(m["content"], str):
            print(f"  user: {m['content'][:200]}")
    if rec.get("looked"):
        print(f"looked at {len(rec['looked'])}: {rec['looked']}; evidence {sess['evidence']}")
    caps = rec["artefacts"].get("captions")
    if caps:
        for e in sess["evidence"]:
            print(f"caption {e}: {caps.get(e, '')[:500]}")
    if rec["artefacts"].get("procedure"):
        print("procedure:", rec["artefacts"]["procedure"][: 2 * a.chars])
    cost = sum(c["cost"] for c in rec["calls"])
    req = [c for c in rec["calls"] if c["phase"] == "request"]
    print(f"cost ${cost:.4f}; request calls {len(req)}, prompt tokens {[c['prompt_tokens'] for c in req]}, "
          f"latency {sum(c['latency_s'] for c in req):.1f}s")


if __name__ == "__main__":
    main()
