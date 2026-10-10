"""Every arm with the model stubbed: what each request carries, and the two bugs the old code had.

Bug 1 (old code): frames never reached the first step that needed them (a forced
text-only search step). Here the by-reference loop is offered look from its first
step, and the pixels a look returns are in the very next request.

Bug 2 (old code): answers about frames came back keyed by call id and were applied
in arrival order. Here every look result is a tool message keyed by its own call,
and every image is preceded by its own frame id, whatever order the fetches finish in.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import io
import json
import random
from pathlib import Path

import pytest
from PIL import Image

from demos_common import Reply
from demoref import arms
from demoref.arms import FrameStore, Settings, run_arm, sample_indices
from demoref.render import encode, render
from demoref.session import build_session, world_for


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    """Render two sessions to disk once: a short one and one longer than the sample budget."""
    root = tmp_path_factory.mktemp("frames")
    out = {}
    for name, (item, size) in {"short": (3, 64), "long": (4, 128)}.items():
        s = build_session(item, size)
        world = world_for(s)
        d = root / name
        d.mkdir()
        for f in s.frames:
            (d / f"{f.id}.webp").write_bytes(encode(render(world, f.spec)))
        out[name] = (s, d)
    return out


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def decoded(part: dict) -> bytes:
    return base64.b64decode(part["image_url"]["url"].split(",", 1)[1])


class Stub:
    """Records every request; answers with a function of (model, messages, kw)."""

    def __init__(self, respond, delay=None):
        self.respond = respond
        self.delay = delay
        self.requests: list[dict] = []

    async def chat(self, model, messages, **kw):
        self.requests.append({"model": model, "messages": copy.deepcopy(messages), **kw})
        if self.delay:
            await asyncio.sleep(self.delay(messages))
        msg = self.respond(model, messages, kw)
        return Reply(message=msg, model=model, prompt_tokens=100, completion_tokens=5, reasoning_tokens=0,
                     cached_tokens=0, cost=0.001, latency_s=0.01)


def call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def answer_with(ids):
    return lambda model, messages, kw: {"role": "assistant", "content": "", "tool_calls": [call("set_queue", {"task_ids": ids}, "c1")]}


ST = Settings(model="m/main", caption_model="m/cheap")


def run(arm, llm, s, d, seed=0):
    return asyncio.run(run_arm(arm, llm, ST, s, d, seed, "test"))


def user_parts(req):
    return req["messages"][1]["content"]


# ----------------------------------------------------------------- inline arms

def test_inline_all_carries_every_frame_after_its_line(frames):
    s, d = frames["short"]
    llm = Stub(answer_with(s.answer))
    r = run("inline_all", llm, s, d)
    assert r.correct and r.answer == s.answer and r.error is None
    parts = user_parts(llm.requests[0])
    imgs = [(i, p) for i, p in enumerate(parts) if p["type"] == "image_url"]
    assert len(imgs) == len(s.frames)
    for (i, p), f in zip(imgs, s.frames):
        assert parts[i - 1]["text"].endswith(arms.line(f))
        assert sha(decoded(p)) == sha((d / f"{f.id}.webp").read_bytes())
    assert parts[-1]["text"] == arms.request_text(s)
    assert llm.requests[0]["tools"] == [arms.SET_QUEUE]


def test_inline_low_halves_every_frame(frames):
    s, d = frames["short"]
    llm = Stub(answer_with(s.answer))
    run("inline_low", llm, s, d)
    imgs = [p for p in user_parts(llm.requests[0]) if p["type"] == "image_url"]
    assert len(imgs) == len(s.frames)
    for p in imgs:
        assert Image.open(io.BytesIO(decoded(p))).size == (640, 400)


def test_inline_sampled_keeps_32_evenly_spaced_and_every_line(frames):
    s, d = frames["long"]
    llm = Stub(answer_with(s.answer))
    run("inline_sampled", llm, s, d)
    parts = user_parts(llm.requests[0])
    keep = sample_indices(len(s.frames))
    assert len(keep) == 32 and keep[0] == 0 and keep[-1] == len(s.frames) - 1
    imgs = [i for i, p in enumerate(parts) if p["type"] == "image_url"]
    assert len(imgs) == 32
    text = "\n".join(p["text"] for p in parts if p["type"] == "text")
    for f in s.frames:
        assert arms.line(f) in text
    shown = [parts[i - 1]["text"].split("\n")[-1].split()[0] for i in imgs]
    assert shown == [s.frames[i].id for i in keep]
    assert sample_indices(10) == list(range(10))


# ----------------------------------------------------------------- captions and pipeline

def test_captions_map_to_their_frames_whatever_order_they_finish(frames):
    s, d = frames["short"]
    by_hash = {sha((d / f"{f.id}.webp").read_bytes()): f.id for f in s.frames}

    def respond(model, messages, kw):
        content = messages[-1]["content"]
        if model == "m/cheap":
            fid = by_hash[sha(decoded(content[1]))]
            return {"role": "assistant", "content": f"caption-of-{fid}"}
        return answer_with(s.answer)(model, messages, kw)

    # captions of later frames finish first
    llm = Stub(respond, delay=lambda m: 0.0 if m[-1]["content"][0].get("text", "").startswith("Task") else random.random() * 0.02)
    r = run("captions", llm, s, d)
    cap_calls = [q for q in llm.requests if q["model"] == "m/cheap"]
    assert len(cap_calls) == len(s.frames)
    for q in cap_calls:
        fid = by_hash[sha(decoded(q["messages"][0]["content"][1]))]
        f = next(f for f in s.frames if f.id == fid)
        assert (f'"{f.said}"' if f.said else "nothing") in q["messages"][0]["content"][0]["text"]
    final = [q for q in llm.requests if q["model"] == "m/main"][0]
    text = user_parts(final)[0]["text"]
    for f in s.frames:
        assert f"{arms.line(f)}\n   screen: caption-of-{f.id}" in text
    assert not any(p["type"] == "image_url" for p in user_parts(final))
    assert r.correct and [c["phase"] for c in r.calls].count("pre") == len(s.frames)


def test_pipeline_covers_every_frame_once_and_keeps_chunk_order(frames):
    s, d = frames["long"]

    def respond(model, messages, kw):
        content = messages[-1]["content"]
        if model == "m/cheap":
            first = content[0]["text"].split("frames ")[1].split(" ")[0]
            return {"role": "assistant", "content": f"notes-from-{first}"}
        if isinstance(content, str):  # the procedure
            assert content.index("notes-from-f000") < content.index("notes-from-f024") < content.index("notes-from-f096")
            return {"role": "assistant", "content": "PROCEDURE: queue " + ", ".join(s.answer)}
        return answer_with(s.answer)(model, messages, kw)

    llm = Stub(respond, delay=lambda m: random.random() * 0.02)
    r = run("pipeline", llm, s, d)
    chunks = [q for q in llm.requests if q["model"] == "m/cheap"]
    seen = []
    for q in chunks:
        seen += [p["text"].split()[0] for p in q["messages"][0]["content"][1:] if p["type"] == "text"]
        assert sum(p["type"] == "image_url" for p in q["messages"][0]["content"]) <= arms.CHUNK
    assert sorted(seen) == [f.id for f in s.frames]
    act = [q for q in llm.requests if q["model"] == "m/main" and q.get("tools")][0]
    assert "PROCEDURE: queue" in user_parts(act)[0]["text"]
    assert not any(p["type"] == "image_url" for p in user_parts(act))
    assert r.correct and r.artefacts["procedure"].startswith("PROCEDURE")


# ----------------------------------------------------------------- by reference

def test_by_reference_first_step_has_lines_and_look_and_no_pixels(frames):
    """Bug 1: frames must be reachable from the first step."""
    s, d = frames["long"]
    targets = list(s.evidence)

    def respond(model, messages, kw):
        if len(messages) == 2:
            return {"role": "assistant", "content": "", "tool_calls": [call("look", {"frame_ids": targets}, "a1")]}
        return answer_with(s.answer)(model, messages, kw)

    llm = Stub(respond)
    r = run("by_reference", llm, s, d)
    first = llm.requests[0]
    assert {t["function"]["name"] for t in first["tools"]} == {"look", "set_queue"}
    parts = user_parts(first)
    assert len(parts) == 1 and parts[0]["type"] == "text"  # no pixels up front
    for f in s.frames:
        assert arms.line(f) in parts[0]["text"]
    # the very next request carries exactly the frames looked at, each after its own label
    second = llm.requests[1]["messages"]
    tool_msgs = [m for m in second if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a1"]
    img_msg = second[-1]
    assert img_msg["role"] == "user"
    labels = [p["text"] for p in img_msg["content"] if p["type"] == "text"]
    pixels = [decoded(p) for p in img_msg["content"] if p["type"] == "image_url"]
    assert len(labels) == len(pixels) == len(targets)
    for lab, px, fid in zip(labels, pixels, targets):
        assert lab.startswith(f"Frame {fid} ")
        assert sha(px) == sha((d / f"{fid}.webp").read_bytes())
    assert r.looked == targets and r.correct and r.steps == 2


def test_by_reference_parallel_looks_stay_matched_when_they_finish_out_of_order(frames):
    """Bug 2: three looks in one step; the first one's fetch is slowest."""
    s, d = frames["short"]
    ids = [s.frames[i].id for i in (5, 9, 14)]

    class SlowFirst(FrameStore):
        async def get(self, frame_id):
            await asyncio.sleep({ids[0]: 0.05, ids[1]: 0.02, ids[2]: 0.0}.get(frame_id, 0))
            return await super().get(frame_id)

    finished: list[str] = []
    orig_get = SlowFirst.get

    async def tracking_get(self, frame_id):
        data = await orig_get(self, frame_id)
        finished.append(frame_id)
        return data

    SlowFirst.get = tracking_get

    def respond(model, messages, kw):
        if len(messages) == 2:
            return {"role": "assistant", "content": "", "tool_calls": [
                call("look", {"frame_ids": [ids[0]]}, "x1"), call("look", {"frame_ids": [ids[1]]}, "x2"),
                call("look", {"frame_ids": [ids[2]]}, "x3")]}
        return answer_with(s.answer)(model, messages, kw)

    llm = Stub(respond)
    orig_store = arms.FrameStore
    arms.FrameStore = SlowFirst
    try:
        r = run("by_reference", llm, s, d)
    finally:
        arms.FrameStore = orig_store
    assert finished == [ids[2], ids[1], ids[0]]  # they really did finish in reverse
    msgs = llm.requests[1]["messages"]
    tools = [m for m in msgs if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["x1", "x2", "x3"]
    for m, fid in zip(tools, ids):
        assert fid in m["content"]
    content = msgs[-1]["content"]
    for k in range(0, len(content), 2):
        label, image = content[k], content[k + 1]
        fid = label["text"].split()[1]
        assert sha(decoded(image)) == sha((d / f"{fid}.webp").read_bytes())
    assert [content[k]["text"].split()[1] for k in range(0, len(content), 2)] == ids
    assert r.correct


def test_unknown_ids_and_no_tool_call_keep_the_loop_going(frames):
    s, d = frames["short"]
    turns = iter([
        {"role": "assistant", "content": "thinking out loud"},
        {"role": "assistant", "content": "", "tool_calls": [call("set_queue", {"task_ids": ["NOPE-1"]}, "q1")]},
        {"role": "assistant", "content": "", "tool_calls": [call("look", {"frame_ids": ["f999", s.frames[0].id]}, "q2")]},
        {"role": "assistant", "content": "", "tool_calls": [call("set_queue", {"task_ids": [x.lower() for x in s.answer]}, "q3")]},
    ])
    llm = Stub(lambda m, msgs, kw: next(turns))
    r = run("by_reference", llm, s, d)
    assert llm.requests[1]["messages"][-1] == {"role": "user", "content": arms.NUDGE}
    assert "unknown task IDs" in llm.requests[2]["messages"][-1]["content"]
    look_msg = [m for m in llm.requests[3]["messages"] if m.get("tool_call_id") == "q2"][0]
    assert "Unknown frame ids: f999" in look_msg["content"]
    assert r.correct and r.steps == 4 and r.looked == [s.frames[0].id]


def test_wrong_order_is_scored_wrong(frames):
    s, d = frames["short"]
    r = run("inline_sampled", Stub(answer_with(list(reversed(s.answer)))), s, d)
    assert r.answer == list(reversed(s.answer)) and not r.correct


def test_reasoning_details_are_handed_back_and_redacted_from_the_record(frames):
    s, d = frames["short"]
    turns = iter([
        {"role": "assistant", "content": "", "reasoning": "let me look", "reasoning_details": [{"type": "reasoning.text", "text": "x", "signature": "sig"}],
         "tool_calls": [call("look", {"frame_ids": [s.frames[1].id]}, "r1")]},
        {"role": "assistant", "content": "", "tool_calls": [call("set_queue", {"task_ids": s.answer}, "r2")]},
    ])
    llm = Stub(lambda m, msgs, kw: next(turns))
    r = run("by_reference", llm, s, d)
    sent = [m for m in llm.requests[1]["messages"] if m["role"] == "assistant"][0]
    assert sent["reasoning_details"][0]["signature"] == "sig"
    assert all("reasoning_details" not in m for m in r.transcript)
    assert r.artefacts["reasoning"][0] == "let me look"
    assert not any(p.get("type") == "image_url" for m in r.transcript if isinstance(m.get("content"), list) for p in m["content"])


def test_same_prompt_apart_from_the_mechanism(frames):
    s, d = frames["short"]
    systems = {}
    for arm in ("inline_all", "inline_low", "inline_sampled", "captions", "by_reference"):
        llm = Stub(lambda m, msgs, kw: ({"role": "assistant", "content": "c"} if m == "m/cheap" else answer_with(s.answer)(m, msgs, kw)))
        run(arm, llm, s, d)
        req = [q for q in llm.requests if q["model"] == "m/main"][0]
        sys_text = req["messages"][0]["content"].split("\n", 1)[1]  # drop the per-run nonce line
        systems[arm] = sys_text.replace(arms.HOW[arm].format(k=arms.SAMPLE_K, n=len(s.frames)), "<HOW>")
        assert arms.header(s) in user_parts(req)[0]["text"]
        assert req["max_tokens"] == ST.max_tokens and req["reasoning"] == ST.reasoning
    assert len(set(systems.values())) == 1


def test_one_reminder_then_the_request_ends_unanswered(frames):
    s, d = frames["short"]
    llm = Stub(lambda m, msgs, kw: {"role": "assistant", "content": "I'd rather ask you which ones you meant."})
    r = run("inline_all", llm, s, d)
    assert len(llm.requests) == 2 and r.answer is None and not r.correct
    assert llm.requests[1]["messages"][-1] == {"role": "user", "content": arms.NUDGE}
    assert r.steps == 2


def test_a_request_over_the_models_limits_is_a_failed_request_not_an_error(frames):
    s, d = frames["short"]

    class TooBig:
        async def chat(self, model, messages, **kw):
            from demos_common import LLMError
            raise LLMError('m: HTTP 400: {"message": "Too much media: 0 document pages + 101 images > 100"}')

    r = run("inline_all", TooBig(), s, d)
    assert r.limit and "Too much media" in r.limit and r.error is None and not r.correct and r.answer is None

    class Flaky:
        async def chat(self, model, messages, **kw):
            raise RuntimeError("connection reset")

    r = run("inline_all", Flaky(), s, d)
    assert r.error and r.limit is None



def test_an_empty_reply_is_the_models_reply_not_a_fault(frames):
    """Found in the mid tier: Sonnet 5.5 sometimes ended a long inline request with no text and no tool
    call, after some thinking or with no usage at all; OpenRouter gave the reason as content_filter
    (Anthropic's refusal stop reason). A refusal is the provider's answer to that request, so it gets
    the one reminder any reply without a tool call gets, and then the request ends unanswered."""
    s, d = frames["short"]

    class Empty:
        def __init__(self, usage):
            self.usage, self.n = usage, 0

        async def chat(self, model, messages, **kw):
            self.n += 1
            return Reply(message={"role": "assistant", "content": ""}, model=model, prompt_tokens=self.usage,
                         completion_tokens=self.usage and 30, reasoning_tokens=self.usage and 30, cached_tokens=0,
                         cost=0.0, latency_s=0.01)

    for usage in (0, 100):
        llm = Empty(usage)
        r = run("inline_all", llm, s, d)
        assert r.error is None and r.limit is None and r.answer is None and not r.correct
        assert llm.n == 2 and r.steps == 2
