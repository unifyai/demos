"""Check, on the recorded runs, that by_reference's frames reached the model matched to their ids.

  uv run python check_frames.py runs/records.jsonl

In every by_reference transcript: each look call is answered by a tool message keyed by its own call
id, in call order; the images that follow are each preceded by the label of the frame asked for (its
id, time and words), in the order asked, none shifted or dropped; and the run's list of frames looked
at is the same sequence. This is the old code's second bug (answers applied in arrival order) checked
on the real runs rather than on stubs."""
import json, re, sys
from collections import Counter
path = sys.argv[1]
stats = Counter()
problems = []
for line in open(path):
    r = json.loads(line)
    if r["arm"] != "by_reference" or r.get("error"):
        continue
    stats["runs"] += 1
    sess = json.load(open(f"data/sessions/{r['session']}.json"))
    ids = {f["id"] for f in sess["frames"]}
    tr = r["transcript"]
    looked_seq = []
    i = 2
    while i < len(tr):
        m = tr[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = m["tool_calls"]
            tools = tr[i + 1:i + 1 + len(calls)]
            if [t.get("tool_call_id") for t in tools] != [c["id"] for c in calls] or any(t["role"] != "tool" for t in tools):
                problems.append((r["model"], r["session"], r["seed"], "tool ids out of order")); break
            want = []
            for c in calls:
                if c["function"]["name"] == "look":
                    try:
                        a = json.loads(c["function"]["arguments"] or "{}")
                    except Exception:
                        a = {}
                    want += [str(x).strip() for x in (a.get("frame_ids") or []) if str(x).strip() in ids]
            j = i + 1 + len(calls)
            if want:
                stats["look_steps"] += 1
                if len([c for c in calls if c["function"]["name"] == "look"]) > 1:
                    stats["steps_with_parallel_looks"] += 1
                u = tr[j]
                assert u["role"] == "user" and isinstance(u["content"], list), (r["session"], u)
                parts = u["content"]
                labels, ok = [], True
                for k in range(0, len(parts), 2):
                    t, im = parts[k], parts[k + 1] if k + 1 < len(parts) else None
                    mm = re.match(r"Frame (f\d+) \[", t.get("text", ""))
                    if not mm or im is None or im.get("type") != "image":
                        ok = False; break
                    labels.append(mm.group(1))
                    # the label carries the frame's own words
                    f = next(x for x in sess["frames"] if x["id"] == mm.group(1))
                    said = f'"{f["said"]}"' if f["said"] else "—"
                    if not t["text"].endswith(f"said: {said}"):
                        ok = False; break
                if not ok or labels != want:
                    problems.append((r["model"], r["session"], r["seed"], "labels differ", labels[:5], want[:5])); break
                looked_seq += labels
                stats["images_checked"] += len(labels)
                j += 1
            i = j
        else:
            i += 1
    if looked_seq != r["looked"]:
        problems.append((r["model"], r["session"], r["seed"], "looked list differs"))
print(dict(stats)); print("problems", len(problems)); print(problems[:10])
