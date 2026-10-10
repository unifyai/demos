"""The arms: one request, six ways for the screen share's frames to reach the model.

Every arm gets the same system prompt (apart from one sentence on how frames
arrive), the same task list, the same frame lines (id, time, what was said),
the same request and the same set_queue tool, with the same model. They differ
only in how the pixels travel:

  inline_all      every frame's image after its line
  inline_low      every frame's image at half width and height (a quarter of the pixels)
  inline_sampled  32 frames' images, evenly spaced; the other frames by their line only
  captions        a cheap model describes each frame (with what was said) as it arrives;
                  the request then reads the descriptions, text only
  pipeline        a demonstration pipeline: notes on each chunk of 24 frames as they arrive,
                  then a written procedure once the share ends; the request reads the procedure
  by_reference    the idea: the lines only, and a look tool that shows the frames asked for,
                  each image labelled with its frame id
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from demos_common import image_part

from .session import Frame, Session, world_for

ARMS = ["inline_all", "inline_low", "inline_sampled", "captions", "pipeline", "by_reference"]
SAMPLE_K = 32
CHUNK = 24
MAX_STEPS = {"by_reference": 12}
DEFAULT_MAX_STEPS = 4

SYSTEM = """{nonce}
You manage the user's task list.

The user shared their screen with you and talked while they did. The frames kept from the screen share are listed in order, one line each: the frame id, the time into the share, and what the user said while that frame was on screen ("—" if nothing). {how}

After the frames comes the user's request. The task list gives every task's ID. Call set_queue once with the task IDs in order. If you are not sure, give your best reading of what the user picked; do not ask them to clarify."""

HOW = {
    "inline_all": "Each frame's image follows its line.",
    "inline_low": "Each frame's image follows its line, at half resolution.",
    "inline_sampled": "To save space only {k} of the {n} frames are shown, evenly spaced: their image follows their line. The other frames have their line only.",
    "captions": "Each frame's line is followed by a description of the screen, written by another model that saw the frame and what was said.",
    "by_reference": "The frames' images are not shown up front: call look with the ids of the frames you need to see.",
}
PIPELINE_SYSTEM = """{nonce}
You manage the user's task list.

The user shared their screen with you and talked while they did. A demonstration pipeline that saw every frame turned the screen share into the written procedure below.

After the procedure comes the user's request. The task list gives every task's ID. Call set_queue once with the task IDs in order. If you are not sure, give your best reading of what the user picked; do not ask them to clarify."""

CAPTION_PROMPT = """This is one frame from a screen share. While it was on screen the person said: {said}

Describe the frame for someone who cannot see it, so they could act on it: the app and view; what is open, selected, highlighted or under the mouse pointer, with any IDs and titles exactly as written; and anything else needed to tell what the person was pointing at. One short paragraph."""

NOTES_PROMPT = """Below are frames {first} to {last} of a screen share, in order. Each frame follows a line with its id, the time into the share and what the person said while it was on screen ("—" if nothing).

Write notes on this part of the share for someone who will not see the frames: what the person did and showed, frame by frame where it matters, with the IDs and titles of anything they opened, selected or pointed at, exactly as written, and what they said about it."""

PROCEDURE_PROMPT = """Below is the transcript of a screen-share demonstration (each frame's id, time and what the person said), and then notes on each part of it, written by someone who saw the frames.

Write down what the person demonstrated and asked for as a procedure someone could follow without seeing the screen: each step in order, naming every item by its ID and title exactly as shown, and applying any corrections the person made. Then list anything you are unsure of."""

SET_QUEUE = {"type": "function", "function": {
    "name": "set_queue",
    "description": "Set next week's queue to these tasks, in this order, replacing what is in it now.",
    "parameters": {"type": "object", "properties": {"task_ids": {
        "type": "array", "items": {"type": "string"}, "description": "Task IDs from the task list, in queue order."}},
        "required": ["task_ids"]}}}
LOOK = {"type": "function", "function": {
    "name": "look",
    "description": "See frames from the screen share. Their images are returned, each labelled with its frame id.",
    "parameters": {"type": "object", "properties": {"frame_ids": {
        "type": "array", "items": {"type": "string"}, "description": "Frame ids, such as f012."}},
        "required": ["frame_ids"]}}}
NUDGE = "Please call set_queue with your answer."
# How a provider says a request is over the model's limits (Anthropic's image cap, a context window).
LIMIT_MARKERS = ("Too much media", "maximum context length", "context_length_exceeded", "prompt is too long")


class ChatModel(Protocol):
    async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> Any: ...


@dataclass
class Settings:
    model: str
    caption_model: str
    reasoning: dict | None = None  # for the request and the procedure
    caption_reasoning: dict | None = None  # for captions and chunk notes
    max_tokens: int = 16000
    provider: dict | None = None
    caption_provider: dict | None = None


@dataclass
class RunResult:
    arm: str
    session: str
    item: int
    size: int
    seed: int
    model: str
    answer: list[str] | None = None
    truth: list[str] = field(default_factory=list)
    correct: bool = False
    calls: list[dict] = field(default_factory=list)  # every model call: phase, kind, tokens, cost, latency
    looked: list[str] = field(default_factory=list)  # by_reference: frame ids looked at, in order
    steps: int = 0
    transcript: list[dict] = field(default_factory=list)
    artefacts: dict = field(default_factory=dict)  # captions, notes, procedure
    error: str | None = None  # a harness or transport failure: the run is excluded and re-run
    limit: str | None = None  # the model refused the request as over its context or image limit: a failed request
    wall_s: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------- frames

class FrameStore:
    """Frame images by id. by_reference reads from it only when the model asks.

    Half-size frames (inline_low) are read from half_root when given (make_data.py
    writes them), so no run resizes images while other runs' calls are in flight."""

    def __init__(self, root: Path, session: Session, half_root: Path | None = None):
        self.root = root
        self.half_root = half_root
        self.ids = [f.id for f in session.frames]

    async def get(self, frame_id: str) -> bytes:
        return (self.root / f"{frame_id}.webp").read_bytes()

    def get_sync(self, frame_id: str) -> bytes:
        return (self.root / f"{frame_id}.webp").read_bytes()

    def get_half_sync(self, frame_id: str) -> bytes:
        if self.half_root is not None:
            return (self.half_root / f"{frame_id}.webp").read_bytes()
        return _half(self.get_sync(frame_id))


def half_frame(data: bytes) -> bytes:
    """A frame at half width and height (LANCZOS), as lossless WebP."""
    return _half(data)


@lru_cache(maxsize=4096)
def _half(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", lossless=True, quality=80, method=4)
    return buf.getvalue()


def img(data: bytes) -> dict:
    return image_part(data, "image/webp")


def line(f: Frame) -> str:
    said = f'"{f.said}"' if f.said else "—"
    return f"{f.id} [{f.stamp}] said: {said}"


def sample_indices(n: int, k: int = SAMPLE_K) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def header(session: Session) -> str:
    world = world_for(session)
    s = int(session.frames[-1].t)
    return (f"Task list:\n{world.listing()}\n\n"
            f"Screen share: {len(session.frames)} frames over {s // 60} min {s % 60:02d} s.")


def request_text(session: Session) -> str:
    return f"Request: {session.request}"


# ----------------------------------------------------------------- calls

def _call_row(reply: Any, phase: str, kind: str, n_images: int) -> dict:
    raw = getattr(reply, "raw", {}) or {}
    return {"phase": phase, "kind": kind, "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens, "reasoning_tokens": reply.reasoning_tokens,
            "cached_tokens": reply.cached_tokens, "cost": reply.cost, "latency_s": round(reply.latency_s, 3),
            "n_images": n_images, "provider": raw.get("provider")}


def _count_images(messages: list[dict]) -> int:
    n = 0
    for m in messages:
        if isinstance(m.get("content"), list):
            n += sum(1 for p in m["content"] if p.get("type") == "image_url")
    return n


def _redact(messages: list[dict]) -> list[dict]:
    """The transcript without pixels (each image becomes a placeholder) or opaque reasoning signatures."""
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if k != "reasoning_details"}
        if isinstance(m.get("content"), list):
            m["content"] = [p if p.get("type") != "image_url" else {"type": "image", "omitted": True} for p in m["content"]]
        out.append(m)
    return out


def _merge_text(parts: list[dict]) -> list[dict]:
    """Join runs of text parts, so a frame list without images is one block."""
    out: list[dict] = []
    for p in parts:
        if p["type"] == "text" and out and out[-1]["type"] == "text":
            out[-1] = {"type": "text", "text": out[-1]["text"] + "\n" + p["text"]}
        else:
            out.append(p)
    return out


async def agent_loop(llm: ChatModel, st: Settings, res: RunResult, session: Session, system: str,
                     user: list[dict], tools: list[dict], store: FrameStore | None, meta: dict) -> None:
    """The one loop every arm's request runs through. Ends when set_queue is called with known IDs."""
    known = {t.id for t in world_for(session).tasks}
    by_id = {f.id: f for f in session.frames}
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": _merge_text(user)}]
    max_steps = MAX_STEPS.get(res.arm, DEFAULT_MAX_STEPS)
    nudged = False
    try:
        for step in range(max_steps):
            reply = await llm.chat(st.model, messages, tools=tools, max_tokens=st.max_tokens, reasoning=st.reasoning,
                                   provider=st.provider, meta={**meta, "phase": "request", "step": step})
            res.calls.append(_call_row(reply, "request", "step", _count_images(messages)))
            res.steps = step + 1
            msg: dict = {"role": "assistant", "content": reply.text or ""}
            if reply.message.get("reasoning_details"):
                # Providers that think before a tool call need their reasoning handed back with it.
                msg["reasoning_details"] = reply.message["reasoning_details"]
            res.artefacts.setdefault("reasoning", []).append(reply.message.get("reasoning") or "")
            calls = reply.tool_calls
            if calls:
                msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": c["function"]} for c in calls]
            messages.append(msg)
            if not calls:
                # One reminder for a reply that answers in words; a second reply without a tool call
                # ends the request unanswered (re-sending the request again would only add cost).
                if nudged:
                    break
                nudged = True
                messages.append({"role": "user", "content": NUDGE})
                continue
            # Answer every call by its own id, in the order the calls were made. Images go in one
            # user message after the tool results, each preceded by its frame id, so nothing
            # depends on the order in which the looks complete.
            results: list[tuple[dict, list[dict]]] = await asyncio.gather(
                *[_run_tool(c, known, by_id, store, res) for c in calls])
            done = False
            images: list[dict] = []
            for tool_msg, parts in results:
                done = done or bool(tool_msg.pop("_final", False))
                messages.append(tool_msg)
                images.extend(parts)
            if images:
                messages.append({"role": "user", "content": images})
            if done:
                break
    finally:
        res.transcript = _redact(messages)


async def _run_tool(call: dict, known: set[str], by_id: dict[str, Frame], store: FrameStore | None,
                    res: RunResult) -> tuple[dict, list[dict]]:
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"].get("arguments") or "{}")
    except json.JSONDecodeError:
        args = None
    base = {"role": "tool", "tool_call_id": call["id"]}
    if args is None or not isinstance(args, dict):
        return {**base, "content": "Error: arguments were not valid JSON."}, []
    if name == "set_queue":
        ids = [str(x).strip().upper() for x in (args.get("task_ids") or [])]
        unknown = [x for x in ids if x not in known]
        if not ids or unknown:
            return {**base, "content": f"Error: unknown task IDs {unknown or '(none given)'}. Use IDs from the task list."}, []
        if res.answer is None:
            res.answer = ids
        return {**base, "content": "Queue updated.", "_final": True}, []
    if name == "look" and store is not None:
        ids = [str(x).strip() for x in (args.get("frame_ids") or [])]
        bad = [x for x in ids if x not in by_id]
        good = [x for x in ids if x in by_id]
        datas = await asyncio.gather(*[store.get(x) for x in good])
        parts: list[dict] = []
        for fid, data in zip(good, datas):
            res.looked.append(fid)
            parts.append({"type": "text", "text": f"Frame {line(by_id[fid])}"})
            parts.append(img(data))
        text = (f"Showing {', '.join(good)} below, each after its label." if good else "No frames shown.")
        if bad:
            text += f" Unknown frame ids: {', '.join(bad)}."
        return {**base, "content": text}, parts
    return {**base, "content": f"Error: no tool named {name}."}, []


# ----------------------------------------------------------------- arms

def inline_parts(arm: str, session: Session, store: FrameStore) -> list[dict]:
    """The inline arms' user content: header, then each frame's line followed by its image."""
    keep = set(sample_indices(len(session.frames))) if arm == "inline_sampled" else None
    parts: list[dict] = [{"type": "text", "text": header(session)}]
    for i, f in enumerate(session.frames):
        parts.append({"type": "text", "text": line(f)})
        if keep is not None and i not in keep:
            continue
        parts.append(img(store.get_half_sync(f.id) if arm == "inline_low" else store.get_sync(f.id)))
    parts.append({"type": "text", "text": request_text(session)})
    return parts


async def run_arm(arm: str, llm: ChatModel, st: Settings, session: Session, frames_root: Path, seed: int,
                  name: str, half_root: Path | None = None) -> RunResult:
    res = RunResult(arm=arm, session=name, item=session.item, size=len(session.frames), seed=seed, model=st.model,
                    truth=list(session.answer))
    nonce = f"Run {uuid.uuid4().hex[:8]}."
    store = FrameStore(frames_root, session, half_root)
    meta = {"arm": arm, "session": name, "seed": seed}
    t0 = time.time()
    try:
        if arm in ("inline_all", "inline_low", "inline_sampled"):
            # read and encode the frames off the event loop, so other runs' calls are not held up
            parts = await asyncio.to_thread(inline_parts, arm, session, store)
            how = HOW[arm].format(k=SAMPLE_K, n=len(session.frames))
            system = SYSTEM.format(nonce=nonce, how=how)
            await agent_loop(llm, st, res, session, system, parts, [SET_QUEUE], None, meta)
        elif arm == "captions":
            caps = await caption_all(llm, st, session, store, res, meta)
            parts = [{"type": "text", "text": header(session)}]
            for f, c in zip(session.frames, caps):
                parts.append({"type": "text", "text": f"{line(f)}\n   screen: {c}"})
            parts.append({"type": "text", "text": request_text(session)})
            system = SYSTEM.format(nonce=nonce, how=HOW["captions"])
            await agent_loop(llm, st, res, session, system, parts, [SET_QUEUE], None, meta)
        elif arm == "pipeline":
            procedure = await demonstration_pipeline(llm, st, session, store, res, meta)
            world = world_for(session)
            parts = [{"type": "text", "text": f"Task list:\n{world.listing()}\n\nWritten procedure from the screen share:\n{procedure}"},
                     {"type": "text", "text": request_text(session)}]
            system = PIPELINE_SYSTEM.format(nonce=nonce)
            await agent_loop(llm, st, res, session, system, parts, [SET_QUEUE], None, meta)
        elif arm == "by_reference":
            parts = [{"type": "text", "text": header(session)}]
            parts += [{"type": "text", "text": line(f)} for f in session.frames]
            parts.append({"type": "text", "text": request_text(session)})
            system = SYSTEM.format(nonce=nonce, how=HOW["by_reference"])
            await agent_loop(llm, st, res, session, system, parts, [LOOK, SET_QUEUE], store, meta)
        else:
            raise ValueError(arm)
    except Exception as e:  # recorded, never silently dropped; the runner re-runs errored runs
        msg = f"{type(e).__name__}: {e}"
        if any(k in msg for k in LIMIT_MARKERS):
            res.limit = msg[:500]  # the request does not fit this model: an outcome, not a fault
        else:
            res.error = msg[:500]
    res.wall_s = round(time.time() - t0, 2)
    res.correct = res.answer == res.truth
    return res


async def caption_all(llm: ChatModel, st: Settings, session: Session, store: FrameStore, res: RunResult,
                      meta: dict) -> list[str]:
    """One caption per frame, made as the frame arrives (so off the request's critical path)."""

    async def one(f: Frame) -> str:
        said = f'"{f.said}"' if f.said else "nothing"
        content = [{"type": "text", "text": CAPTION_PROMPT.format(said=said)}, img(await store.get(f.id))]
        reply = await llm.chat(st.caption_model, [{"role": "user", "content": content}], max_tokens=1200,
                               reasoning=st.caption_reasoning, provider=st.caption_provider,
                               meta={**meta, "phase": "pre", "kind": "caption", "frame": f.id})
        res.calls.append(_call_row(reply, "pre", "caption", 1))
        return " ".join(reply.text.split())

    caps = await asyncio.gather(*[one(f) for f in session.frames])
    res.artefacts["captions"] = dict(zip([f.id for f in session.frames], caps))
    return list(caps)


async def demonstration_pipeline(llm: ChatModel, st: Settings, session: Session, store: FrameStore,
                                 res: RunResult, meta: dict) -> str:
    """Notes on each chunk of frames as it completes, then one procedure when the share ends."""
    chunks = [session.frames[i:i + CHUNK] for i in range(0, len(session.frames), CHUNK)]

    async def notes(chunk: list[Frame]) -> str:
        content: list[dict] = [{"type": "text", "text": NOTES_PROMPT.format(first=chunk[0].id, last=chunk[-1].id)}]
        for f in chunk:
            content.append({"type": "text", "text": line(f)})
            content.append(img(await store.get(f.id)))
        reply = await llm.chat(st.caption_model, [{"role": "user", "content": content}], max_tokens=4000,
                               reasoning=st.caption_reasoning, provider=st.caption_provider,
                               meta={**meta, "phase": "pre", "kind": "notes", "first": chunk[0].id})
        res.calls.append(_call_row(reply, "pre", "notes", len(chunk)))
        return reply.text.strip()

    all_notes = await asyncio.gather(*[notes(c) for c in chunks])
    transcript = "\n".join(line(f) for f in session.frames)
    body = "\n\n".join(f"Notes on frames {c[0].id} to {c[-1].id}:\n{n}" for c, n in zip(chunks, all_notes))
    content = f"{PROCEDURE_PROMPT}\n\nTranscript:\n{transcript}\n\n{body}"
    reply = await llm.chat(st.model, [{"role": "user", "content": content}], max_tokens=st.max_tokens,
                           reasoning=st.reasoning, provider=st.provider,
                           meta={**meta, "phase": "request", "kind": "procedure"})
    res.calls.append(_call_row(reply, "request", "procedure", 0))
    res.artefacts["notes"] = list(all_notes)
    res.artefacts["procedure"] = reply.text
    return reply.text.strip()
