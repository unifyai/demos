"""Print a run's whole transcript, for reading.

    uv run python show.py runs/<condition>.jsonl <arm> <item> [seed] [--conversation]
"""

from __future__ import annotations

import json
import sys

from conversation_context.data import conversation
from conversation_context.scenarios import BY_ID


def short(s: object, n: int = 1200) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + f"... [{len(s) - n} more chars]"


def show_messages(msgs: list[dict], indent: str = "  ") -> None:
    for m in msgs:
        if m["role"] == "assistant":
            if m.get("content"):
                print(f"{indent}ASSISTANT: {short(m['content'])}")
            for tc in m.get("tool_calls") or []:
                print(f"{indent}CALL {tc['function']['name']}({tc['function']['arguments']})")
        elif m["role"] == "tool":
            print(f"{indent}RESULT: {short(m['content'], 900)}")
        else:
            print(f"{indent}{m['role'].upper()}: {short(m['content'])}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path, arm, item = args[:3]
    seed = int(args[3]) if len(args) > 3 else 0
    for line in open(path):
        r = json.loads(line)
        if r["arm"] == arm and r["item"] == item and r["seed"] == seed:
            break
    else:
        sys.exit("not found")
    s = BY_ID[r["scenario"]]
    print(f"== {r['condition']} | {arm} | {item} | seed {seed} | Maya {'there' if r['present'] else 'away'}")
    print(f"question: {s['question_complete'] if r['kind'] == 'complete' else s['question']}")
    print(f"target: {s['target_answer']}   distractor: {s['distractor_answer']}")
    print(f"first request: {r['first_request']!r} (names detail: {r['first_request_has_detail']})")
    if "--conversation" in sys.argv:
        for m in conversation(s, r["kind"], r["length"], seed):
            print(f"  [{m['role']}] {short(m['content'], 300)}")
    for a in r["transcript"]:
        print(f"\n-- {a['name']} (level {a['level']}{', ' + a['source'] if a['source'] else ''}) request: {a['request']!r}")
        if a["background"]:
            print(f"  BACKGROUND ({a['background_chars']} chars): {short(a['background'], 2500)}")
        show_messages(a["messages"])
    for q in r["questions"]:
        print(f"\n?? {q['from']} -> {q['to']}: {q['question']!r}\n   answer: {q['answer']!r} (escalated: {q['escalated']})")
        show_messages(q.get("turn", []), indent="     ")
    for q in r["person_questions"]:
        print(f"\n?? {q['from']} -> Maya: {q['question']!r} -> {q['reply']!r}")
    print(f"\nFINAL: {r['final']}")
    print(f"calls {r['calls']}, prompt {r['prompt_tokens']}, completion {r['completion_tokens']}, ${r['cost']:.4f}, "
          f"{r['latency']:.1f}s, step limits {r['step_limits']}")
    for c in r.get("calls_detail", []):
        print(f"   {c['agent']:22s} p{c['prompt_tokens']:6d} c{c['completion_tokens']:5d} ${c['cost']:.5f} {c['latency']:.1f}s")


if __name__ == "__main__":
    main()
