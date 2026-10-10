"""The four arms. The same model, tools and shared instructions in each; they
differ only in how the person reaches the work.

flat        one loop does the work and talks; the person's messages are added
            between its steps.
blocking    a conversation loop whose run_task runs the work to the end.
background  a conversation loop that starts work in the background, reads its
            progress and can stop it; the work cannot be steered or ask.
handle      the idea: as background, plus a handle that takes messages
            (corrections, additions, answers), pauses and resumes, and a work
            loop that can ask the person through the conversation.

Rules every loop shares: a message from the person (or, for a work loop, a
message passed on to it) that arrives while the model is deciding its next
step discards that decision, and the model decides again with the message in
view. The tool calls of one response queue on the worker's session; one whose
turn comes after a message (or a pause or stop) has arrived is not started,
and its result says why. An action already under way finishes.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any

from demos_common import LLMError

from .core import (Mailbox, Run, append_user, assistant_message, fn_schema, parse_args,
                   tool_message)
from .sites import Env

MAX_DISCARDS = 3

# --------------------------------------------------------------------------- #
#  Instructions                                                               #
# --------------------------------------------------------------------------- #

TODAY = "Today is Friday 9 October 2026."

TALK = """You are a personal assistant, chatting with a person who has asked you to get things done.
- Reply to each message from the person promptly and briefly, in plain text (one to three sentences).
- Answer their questions directly. When they ask about progress, be specific: what is done and what is left.
- If they ask something unrelated to the work, just answer it.
- When a piece of work finishes or stops, tell them the outcome with the key details."""

WORK = """How to do the work:
- Do the whole task with the tools, one step at a time, including every check the task asks for.
- The person may send corrections, additions or changes of mind while the work runs. Apply each one as soon as you see it, to everything still to do, and fix anything already done that it affects (for example, send a correction to anyone who already got a wrong detail, or redo a check made with an old detail).
- If the task needs a fact that only the person can give (it is not in what they said and no tool can tell you), do not guess or make it up: ask the person. Carry on with whatever does not depend on the answer.
- If the person asks you to pause, take no further steps until they say to carry on. If they ask you to stop or cancel something, take no further steps on it.
- When the work is finished or stopped, give the outcome: what was done, with the key details, and anything skipped or not done."""

FLAT_SYSTEM = TODAY + "\n\n" + TALK + "\n\nYou do the work yourself, with the tools.\n\n" + WORK + """

How this works: the person's messages reach you between your steps. To reply to them while you work, call reply_to_person, in the same response as your next tool calls if there are any, so the work carries on. Your turn ends when you respond without calling any tools, and that response is shown to the person: do that only when all the work is finished, when the person has asked you to pause or stop, or when you cannot go on without an answer from the person."""

CONV_HANDLE = TODAY + "\n\n" + TALK + """

How this works: you do not do the work yourself. To get something done, call start_task with a complete description of the work, including every detail the person gave. It starts the work in the background and returns a task id at once, so you can keep talking while it runs. While a task runs:
- task_status shows what it has done so far and any question it is waiting on. Use it to answer questions about progress.
- send_to_task passes a message to the task: a correction, an addition, a change of mind, or the person's answer to a question the task asked. Include the person's own words.
- pause_task, resume_task and stop_task pause, resume and stop it.
You get a notice when a task asks the person something (put the question to the person, then pass their answer back with send_to_task) and when a task finishes."""

CONV_BACKGROUND = TODAY + "\n\n" + TALK + """

How this works: you do not do the work yourself. To get something done, call start_task with a complete description of the work, including every detail the person gave. It starts the work in the background and returns a task id at once, so you can keep talking while it runs. task_status shows what a task has done so far; use it to answer questions about progress. stop_task stops a task. A running task cannot be given new instructions and cannot ask the person anything. To change what it is doing, stop it and start a new task that says what is already done and what to do now, or start a follow-up task once it finishes. You get a notice when a task finishes; if it stopped because it needs something from the person, ask them, then start a task to carry on."""

CONV_BLOCKING = TODAY + "\n\n" + TALK + """

How this works: you do not do the work yourself. To get something done, call run_task with a complete description of the work, including every detail the person gave. It does the work and returns the outcome when it is finished. While it runs you cannot read or answer messages: the person's messages wait until it returns. If the outcome is a question for the person, ask them, then run a task to carry on."""

WORK_HANDLE = TODAY + """

You are doing a piece of work for a person, with the tools. You do not talk to the person directly: an assistant chats with them and passes their messages on to you, between your steps.

""" + WORK + """

How this works: to ask the person something, call ask_person. It returns at once; their answer arrives later as a message. If you have nothing else to do until it arrives, respond without calling any tools and you will wait for it. Otherwise, responding without calling any tools ends the work, and your response goes to the assistant as the outcome."""

WORK_NOREACH = TODAY + """

You are doing a piece of work for a person, with the tools. You cannot talk to the person while you work, and no messages from them will reach you.

""" + WORK + """

How this works: if you need a fact that only the person can give, do everything that does not depend on it, then finish with a clear question saying what you need and what you have done so far. Responding without calling any tools ends the work, and your response goes to the assistant as the outcome."""

_TID = {"type": "string", "description": "The task id, such as t1."}
_DESC = {"type": "string", "description": "A complete description of the work: what to do, and every detail the person gave."}
START = fn_schema("start_task", "Start a piece of work in the background. Returns a task id at once.", {"description": _DESC}, ["description"])
STATUS = fn_schema("task_status", "See what a task has done so far, what it is doing now, and any question it is waiting on.",
                   {"task_id": {"type": "string", "description": "The task id. Leave out to see every task."}}, [])
SEND = fn_schema("send_to_task", "Pass a message to a running task: a correction, an addition, a change of mind, or the person's answer to its question. The task reads it before its next step.",
                 {"task_id": _TID, "message": {"type": "string", "description": "The message, with the person's own words."}}, ["task_id", "message"])
PAUSE = fn_schema("pause_task", "Pause a running task: it starts no new step until it is resumed.", {"task_id": _TID}, ["task_id"])
RESUME = fn_schema("resume_task", "Resume a paused task.", {"task_id": _TID}, ["task_id"])
STOP = fn_schema("stop_task", "Stop a task for good: it starts no new step.", {"task_id": _TID}, ["task_id"])
RUN = fn_schema("run_task", "Do a piece of work and return its outcome when it is finished. You cannot read or answer messages until it returns.",
                {"description": _DESC}, ["description"])
REPLY = fn_schema("reply_to_person", "Send the person a message now, without ending your turn, so the work carries on.",
                  {"text": {"type": "string", "description": "The message."}}, ["text"])
ASK = fn_schema("ask_person", "Ask the person a question. Returns at once; their answer arrives later as a message.",
                {"question": {"type": "string", "description": "The question."}}, ["question"])


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _short(v: Any, n: int = 90) -> str:
    s = v if isinstance(v, str) else json.dumps(v, separators=(",", ":"))
    return s if len(s) <= n else s[: n - 3] + "..."


# --------------------------------------------------------------------------- #
#  The work loop (blocking, background and handle)                            #
# --------------------------------------------------------------------------- #


class Task:
    """One piece of work: a tool loop of its own, and the surface that steers it."""

    def __init__(self, arm: "Conversation", tid: str, description: str, steerable: bool, seen: int):
        self.arm, self.run, self.env = arm, arm.run, arm.env
        self.id = tid
        self.description = description
        self.steerable = steerable
        self.inbox = Mailbox()
        self.paused = False
        self.was_paused = False  # set by pause(); the loop tells the work when it carries on
        self.stopped = False
        self.waiting = False
        self.resumed = asyncio.Event()
        self.resumed.set()
        self.questions: list[dict[str, Any]] = []
        self.notes = ""
        self.state = "running"  # running | finished | stopped | failed
        self.result: str | None = None
        self.done = asyncio.Event()
        self.t_start = self.run.clock.now()
        self.rec: dict[str, Any] = {"id": tid, "arm": arm.name, "description": description, "t_start": self.t_start,
                                    "seen": seen, "t_end": None, "state": "running", "result": None,
                                    "interjections": [], "controls": [], "questions": self.questions}
        self.run.tasks.append(self.rec)
        system = WORK_HANDLE if steerable else WORK_NOREACH
        self.ctx: list[dict[str, Any]] = [{"role": "system", "content": system},
                                          {"role": "user", "content": f"The task: {description}"}]
        self.run.contexts[f"work:{tid}"] = self.ctx
        self.relays = [f"{tid}:desc"]
        self.aio = asyncio.create_task(self._main())

    # ---- the steering surface ------------------------------------------- #

    def _control(self, op: str, result: str) -> str:
        self.rec["controls"].append({"t": self.run.clock.now(), "op": op, "result": result})
        return result

    def interject(self, text: str, seen: int) -> str:
        if self.state != "running" or self.stopped:
            return self._control("send", f"Task {self.id} has already {self.state if self.state != 'running' else 'stopped'}, "
                                          f"so it cannot take messages. Start a new task for anything more.")
        iid = f"{self.id}:m{len(self.rec['interjections']) + 1}"
        now = self.run.clock.now()
        self.rec["interjections"].append({"id": iid, "t": now, "text": text, "seen": seen})
        pending = [q for q in self.questions if q["answered_t"] is None]
        for q in pending:
            q["answered_t"] = now
        self.inbox.put({"id": iid, "text": text})
        return self._control("send", f"Passed to task {self.id}" + (" as the answer to its question." if pending else "."))

    def pause(self) -> str:
        if self.state != "running" or self.stopped:
            return self._control("pause", f"Task {self.id} is not running (it has {self.state}), so there is nothing to pause.")
        if self.paused:
            return self._control("pause", f"Task {self.id} is already paused.")
        self.paused = True
        self.was_paused = True
        self.resumed.clear()
        return self._control("pause", f"Paused task {self.id}. A step already under way finishes; no new step starts until you resume it.")

    def resume(self) -> str:
        if self.state != "running" or self.stopped:
            return self._control("resume", f"Task {self.id} is not running (it has {self.state}), so there is nothing to resume.")
        if not self.paused:
            return self._control("resume", f"Task {self.id} is not paused; it is already running.")
        self.paused = False
        self.resumed.set()
        return self._control("resume", f"Resumed task {self.id}.")

    def stop(self) -> str:
        if self.state != "running" or self.stopped:
            return self._control("stop", f"Task {self.id} has already {self.state if self.state != 'running' else 'stopped'}.")
        self.stopped = True
        self.resumed.set()
        self.inbox.poke()
        return self._control("stop", f"Stopping task {self.id}: it starts no new step (a step already under way finishes).")

    def status(self) -> str:
        now = self.run.clock.now()
        acts = [a for a in self.env.log if a["worker"] == self.id and not a["skipped"]]
        done = [a for a in acts if a["t_end"] is not None]
        going = [a for a in acts if a["t_start"] is not None and a["t_end"] is None and not a["cancelled"]]
        queued = [a for a in acts if a["t_start"] is None and not a["cancelled"]]
        if self.state != "running":
            state = self.state
        elif self.stopped:
            state = "stopping"
        elif self.paused:
            state = "paused"
        elif self.waiting:
            state = "waiting for the person's answer"
        else:
            state = "running"
        lines = [f"Task {self.id}: {state}; started {_mmss(now - self.t_start)} ago.",
                 f"The task: {_short(self.description, 400)}",
                 f"Actions: {len(done)} finished" + (f", {len(going)} under way" if going else "")
                 + (f", {len(queued)} queued" if queued else "") + "."]
        for a in acts[-40:]:
            if a["t_end"] is not None:
                res = _short(a["result"], 140)
            elif a["cancelled"]:
                res = "(cancelled)"
            else:
                res = "(under way)" if a["t_start"] is not None else "(queued, not started)"
            args = ", ".join(f"{k}={_short(v, 70)}" for k, v in a["args"].items())
            lines.append(f"- {a['tool']}({args}) -> {res}")
        for q in self.questions:
            if q["answered_t"] is None:
                lines.append(f"Waiting on the person for: {q['text']}")
        if self.notes:
            lines.append(f"Its latest note: {_short(self.notes, 300)}")
        if self.result is not None:
            lines.append(f"Its report: {self.result}")
        return "\n".join(lines)

    # ---- the loop ----------------------------------------------------------- #

    async def _main(self) -> None:
        try:
            await self._loop()
        except LLMError as e:
            self.state, self.result = "failed", f"The task failed: {e}"
            self.run.errors.append(f"{self.id}: {e}")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # a bug: record it, never take the conversation down
            self.state, self.result = "failed", f"The task failed: {type(e).__name__}: {e}"
            self.run.errors.append(f"{self.id}: {traceback.format_exc()}")
        if self.stopped and self.state == "running":
            n = len([a for a in self.env.log if a["worker"] == self.id and a["t_end"] is not None])
            self.state = "stopped"
            self.result = f"Stopped on request after {n} actions. Its latest note: {self.notes or '(none)'}"
        self.rec.update({"t_end": self.run.clock.now(), "state": self.state, "result": self.result})
        self.done.set()
        self.arm.on_task_done(self)

    async def _loop(self) -> None:
        tools = self.env.schemas() + ([ASK] if self.steerable else [])
        discards = 0
        while True:
            while self.paused and not self.stopped:
                await self.resumed.wait()
            if self.stopped:
                return
            if self.was_paused:
                self.was_paused = False
                append_user(self.ctx, "Note from the assistant: the work was paused and has now been resumed. "
                                      "Carry on with whatever is still to do.")
            for item in self.inbox.drain():
                append_user(self.ctx, f"Message from the person, passed on by the assistant: {item['text']}")
                self.relays.append(item["id"])
            reply, rec = await self.run.chat(f"work:{self.id}", self.ctx, tools, relays=self.relays)
            if self.stopped:
                rec["discarded"] = True
                return
            if (self.inbox.has() or self.paused) and discards < MAX_DISCARDS:
                rec["discarded"] = True
                discards += 1
                continue
            discards = 0
            text = reply.text.strip()
            if text:
                self.notes = text
            if reply.tool_calls:
                msg = assistant_message(reply)
                self.ctx.append(msg)
                outs = await asyncio.gather(*(self._call(c) for c in msg["tool_calls"]))
                self.ctx.extend(tool_message(c, o) for c, o in zip(msg["tool_calls"], outs))
                continue
            if text:
                self.ctx.append(assistant_message(reply))
            if self.steerable and any(q["answered_t"] is None for q in self.questions) and not self.stopped:
                self.waiting = True
                while not self.inbox.has() and not self.stopped:
                    await self.inbox.wait_event()
                self.waiting = False
                continue
            self.state = "finished"
            self.result = text or "(The work ended without a report.)"
            return

    async def _call(self, call: dict[str, Any]) -> str:
        name = call["function"]["name"]
        if name == "ask_person" and self.steerable:
            args = parse_args(call)
            q = str((args or {}).get("question") or "").strip()
            if not q:
                return "Error: ask_person needs a question."
            qid = f"{self.id}:q{len(self.questions) + 1}"
            self.questions.append({"id": qid, "text": q, "t": self.run.clock.now(), "answered_t": None})
            self.arm.notify({"kind": "question", "task": self.id, "qid": qid, "text": q})
            return ("Your question has been passed to the person. Their answer will arrive as a message; carry on "
                    "with anything that does not depend on it.")
        return await self.env.call(self.id, name, call["function"].get("arguments"), gate=self._gate)

    def _gate(self) -> str | None:
        """Why an action whose turn has come should not start, if it should not."""
        if self.stopped:
            return "the work was stopped"
        if self.paused:
            return "the work was paused"
        if self.inbox.has():
            return "a new message from the person arrived"
        return None


# --------------------------------------------------------------------------- #
#  Arms                                                                       #
# --------------------------------------------------------------------------- #


class Arm:
    name = ""

    def __init__(self, run: Run, env: Env):
        self.run, self.env = run, env

    def deliver(self, msg: dict[str, Any]) -> None:
        raise NotImplementedError

    async def main(self) -> None:
        raise NotImplementedError

    def aio_tasks(self) -> list[asyncio.Task]:
        return []


class Flat(Arm):
    """One loop: it does the work and talks to the person."""

    name = "flat"

    def __init__(self, run: Run, env: Env):
        super().__init__(run, env)
        self.inbox = Mailbox()
        self.ctx: list[dict[str, Any]] = [{"role": "system", "content": FLAT_SYSTEM}]
        run.contexts["flat"] = self.ctx
        self.seen = -1

    def deliver(self, msg: dict[str, Any]) -> None:
        self.inbox.put(msg)

    def _gate(self) -> str | None:
        return "a new message from the person arrived" if self.inbox.has() else None

    def _take(self) -> None:
        for m in self.inbox.drain():
            append_user(self.ctx, m["text"])
            self.seen = max(self.seen, m["id"])

    async def _call(self, call: dict[str, Any]) -> str:
        name = call["function"]["name"]
        if name == "reply_to_person":
            text = str((parse_args(call) or {}).get("text") or "").strip()
            if not text:
                return "Error: the message is empty."
            self.run.say(text, "flat", self.seen)
            return "Sent."
        return await self.env.call("flat", name, call["function"].get("arguments"), gate=self._gate)

    async def main(self) -> None:
        tools = self.env.schemas() + [REPLY]
        working = False
        discards = 0
        while True:
            if not working:
                await self.inbox.wait()
            self._take()
            reply, rec = await self.run.chat("flat", self.ctx, tools, seen=self.seen)
            if self.inbox.has() and discards < MAX_DISCARDS:
                rec["discarded"] = True
                discards += 1
                working = True
                continue
            discards = 0
            text = reply.text.strip()
            if reply.tool_calls or text:
                msg = assistant_message(reply)
                self.ctx.append(msg)
            if text:
                self.run.say(text, "flat", self.seen)
            if reply.tool_calls:
                outs = await asyncio.gather(*(self._call(c) for c in msg["tool_calls"]))
                self.ctx.extend(tool_message(c, o) for c, o in zip(msg["tool_calls"], outs))
                working = True
            else:
                working = False


class Conversation(Arm):
    """A conversation loop that does no work itself (blocking, background, handle)."""

    system = ""
    tools: list[dict[str, Any]] = []
    steerable = False

    def __init__(self, run: Run, env: Env):
        super().__init__(run, env)
        self.mail = Mailbox()
        self.ctx: list[dict[str, Any]] = [{"role": "system", "content": self.system}]
        run.contexts["conv"] = self.ctx
        self.seen = -1
        self.tasks: dict[str, Task] = {}

    def deliver(self, msg: dict[str, Any]) -> None:
        self.mail.put({"kind": "person", **msg})

    def notify(self, event: dict[str, Any]) -> None:
        self.mail.put(event)

    def on_task_done(self, task: Task) -> None:
        self.notify({"kind": "done", "task": task.id, "state": task.state, "result": task.result})

    def aio_tasks(self) -> list[asyncio.Task]:
        return [t.aio for t in self.tasks.values()]

    def _take(self) -> None:
        for e in self.mail.drain():
            if e["kind"] == "person":
                append_user(self.ctx, e["text"])
                self.seen = max(self.seen, e["id"])
            elif e["kind"] == "question":
                append_user(self.ctx, f"[Notice from task {e['task']}, not from the person] The task asks the person: {e['text']}")
            elif e["kind"] == "done":
                append_user(self.ctx, f"[Notice from task {e['task']}, not from the person] The task has {e['state']}. "
                                      f"Its report: {e['result']}")

    async def main(self) -> None:
        while True:
            await self.mail.wait()
            self._take()
            discards = 0
            while True:
                reply, rec = await self.run.chat("conv", self.ctx, self.tools, seen=self.seen)
                if self.mail.has(lambda e: e["kind"] == "person") and discards < MAX_DISCARDS:
                    rec["discarded"] = True
                    discards += 1
                    self._take()
                    continue
                discards = 0
                text = reply.text.strip()
                if reply.tool_calls or text:
                    msg = assistant_message(reply)
                    self.ctx.append(msg)
                if text:
                    self.run.say(text, "conv", self.seen)
                if not reply.tool_calls:
                    break
                outs = await asyncio.gather(*(self.dispatch(c) for c in msg["tool_calls"]))
                self.ctx.extend(tool_message(c, o) for c, o in zip(msg["tool_calls"], outs))
                self._take()

    def _new_task(self, description: str) -> Task:
        tid = f"t{len(self.tasks) + 1}"
        t = Task(self, tid, description, self.steerable, self.seen)
        self.tasks[tid] = t
        return t

    def _get(self, args: dict[str, Any]) -> tuple[Task | None, str]:
        tid = str(args.get("task_id") or "").strip().lower()
        if tid in self.tasks:
            return self.tasks[tid], ""
        known = ", ".join(f"{k} ({t.state})" for k, t in self.tasks.items()) or "none"
        return None, f"Error: no task {args.get('task_id')!r}. Tasks: {known}."

    async def dispatch(self, call: dict[str, Any]) -> str:
        name = call["function"]["name"]
        args = parse_args(call)
        if args is None:
            return "Error: the arguments are not a JSON object."
        allowed = {t["function"]["name"] for t in self.tools}
        if name not in allowed:
            return f"Error: no tool {name!r}."
        try:
            return await self._dispatch(name, args)
        except Exception as e:  # a bug in a tool must not take the conversation down
            self.run.errors.append(f"conv tool {name}: {traceback.format_exc()}")
            return f"Error: {type(e).__name__}: {e}"

    async def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        if name in ("start_task", "run_task"):
            desc = str(args.get("description") or "").strip()
            if not desc:
                return "Error: describe the work."
            t = self._new_task(desc)
            if name == "start_task":
                return f"Started task {t.id}. It runs in the background; you get a notice when it finishes" + (
                    " or asks the person something." if self.steerable else ".")
            await t.done.wait()
            return f"Task {t.id} has {t.state}. Its report: {t.result}"
        if name == "task_status":
            if not args.get("task_id"):
                return "\n\n".join(t.status() for t in self.tasks.values()) or "No tasks yet."
            t, err = self._get(args)
            return err or t.status()
        t, err = self._get(args)
        if t is None:
            return err
        if name == "send_to_task":
            msg = str(args.get("message") or "").strip()
            return t.interject(msg, self.seen) if msg else "Error: the message is empty."
        if name == "pause_task":
            return t.pause()
        if name == "resume_task":
            return t.resume()
        if name == "stop_task":
            return t.stop()
        return f"Error: no tool {name!r}."


class Blocking(Conversation):
    name = "blocking"
    system = CONV_BLOCKING
    tools = [RUN]

    def on_task_done(self, task: Task) -> None:
        pass  # the outcome comes back as run_task's result


class Background(Conversation):
    name = "background"
    system = CONV_BACKGROUND
    tools = [START, STATUS, STOP]


class Handle(Conversation):
    name = "handle"
    system = CONV_HANDLE
    tools = [START, STATUS, SEND, PAUSE, RESUME, STOP]
    steerable = True


ARMS = {"flat": Flat, "blocking": Blocking, "background": Background, "handle": Handle}
