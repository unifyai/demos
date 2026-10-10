"""What each role sees of the call, and what the judge reads.

Model views are timestamped transcripts, one line per event, sent as one
content block per line so that a provider's prompt cache can reuse the part a
previous call already sent. Each view shows only what that role could know.
"""

from __future__ import annotations

import json
from typing import Any

from .timing import Timing, heard_prefix

AGENT_SPW = Timing().agent_spw

# How agent lines are labelled in each view, by who wrote them.
_AGENT_LABELS = {
    "single": {},
    "relay_fast": {},
    "relay_slow": {},
    "slow_speaks_fast": {"slow": "Ava (main assistant)", "replay": "Ava (replay of the main assistant)", "fast": "Ava (you)"},
    "slow_speaks_slow": {"slow": "Ava (you)", "replay": "Ava (voice, replaying you)", "fast": "Ava (voice)"},
}
_SHOWS_TOOLS = {"single", "relay_slow", "slow_speaks_slow"}
_SHOWS_NOTES = {"relay_fast": "(note)", "relay_slow": "(your note)"}


def _short(x: Any, limit: int = 1500) -> str:
    s = json.dumps(x, ensure_ascii=False, separators=(", ", ": "))
    return s if len(s) <= limit else s[:limit] + " ...(truncated)"


def view_lines(sim: Any, view: str) -> list[str]:
    """The transcript as `view` sees it now, oldest first, then queued speech."""
    rows: list[tuple[float, int, str]] = []
    labels = _AGENT_LABELS[view]
    for p in sim.persons:
        if p.t_heard <= sim.now + 1e-9:
            rows.append((p.t_start, 0, f"[{p.t_start:.1f}s] Sam: {p.turn.text}"))
    for u in sim.utts:
        if u.t_start is None:
            continue
        who = labels.get(u.source, "Ava")
        if u.t_end is None:
            text = f"{u.text} [still speaking]"
        elif u.cut:
            text = f"{u.heard} [cut off]" if u.heard else "[cut off before any words were heard]"
        else:
            text = u.heard
        rows.append((u.t_start, 1, f"[{u.t_start:.1f}s] {who}: {text}"))
    if view in _SHOWS_TOOLS:
        for tc in sim.tool_calls:
            if view == "single" or tc.role == "slow":
                rows.append((tc.t_call, 2, f"[{tc.t_call:.1f}s] (tool call #{tc.id}) {tc.name}({_short(tc.args, 400)})"))
                if tc.done:
                    rows.append((tc.t_done, 3, f"[{tc.t_done:.1f}s] (tool result #{tc.id}, {tc.name}) {_short(tc.result)}"))
    if view in _SHOWS_NOTES:
        tag = _SHOWS_NOTES[view]
        for n in sim.notes:
            rows.append((n.t, 4, f"[{n.t:.1f}s] {tag} {n.text}"))
    rows.sort(key=lambda r: (r[0], r[1]))
    lines = [r[2] for r in rows]
    for u in sim.queue:
        who = labels.get(u.source, "Ava")
        lines.append(f"[{u.t_queued:.1f}s] {who}: [about to say] {u.text}")
    return lines


def messages(system: str, lines: list[str], instruction: str, anthropic: bool) -> list[dict]:
    """System prompt plus one user message: the transcript (one block per line, the
    last one marked for caching on Anthropic models) and the instruction."""
    sys_block: dict = {"type": "text", "text": system}
    if anthropic:
        sys_block["cache_control"] = {"type": "ephemeral"}
    blocks: list[dict] = [{"type": "text", "text": "The call so far:\n"}]
    if lines:
        blocks += [{"type": "text", "text": line + "\n"} for line in lines]
    else:
        blocks.append({"type": "text", "text": "(nothing has been said yet)\n"})
    if anthropic:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
    blocks.append({"type": "text", "text": "\n" + instruction})
    return [{"role": "system", "content": [sys_block]}, {"role": "user", "content": blocks}]


def judge_transcript(rec: dict, min_gap: float = 1.0) -> str:
    """The call as Sam heard it, with timings and silences, without any arm label."""
    events = []
    for p in rec["persons"]:
        events.append((p["t_start"], p["t_end"], "Sam", p["text"], False, p["barged"]))
    for u in rec["utterances"]:
        if u["t_start"] is None:
            continue
        if u["t_end"] is None:  # still speaking when the call ended (recorded before that case was closed in Sim)
            heard, _ = heard_prefix(u["text"], AGENT_SPW, rec["t_end"] - u["t_start"])
            events.append((u["t_start"], rec["t_end"], "Ava", (heard + " " if heard else "") + "[the call ended here]", True, False))
            continue
        text = u["heard"] if u["heard"] else ""
        if u.get("call_ended"):
            text = (text + " ") if text else ""
            text += "[the call ended here]"
        elif u["cut"]:
            text = (text + " ") if text else ""
            text += "[cut off by Sam]"
        events.append((u["t_start"], u["t_end"], "Ava", text, u["cut"], False))
    events.sort(key=lambda e: (e[0], 0 if e[2] == "Ava" else 1))
    out, last_end = [], 0.0
    for t0, t1, who, text, _cut, barged in events:
        if t0 - last_end >= min_gap:
            out.append(f"      ({t0 - last_end:.1f} s of silence)")
        note = " [starts talking over Ava]" if barged else ""
        out.append(f"[{t0:6.1f}-{t1:6.1f}s] {who}:{note} {text}")
        last_end = max(last_end, t1)
    return "\n".join(out)
