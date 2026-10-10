"""Each arm's tools. Every arm runs its code through the same Executor; the
baselines fix the coordinates (the harness chooses them), the coordinate arms
let the model choose them on every call.

The descriptions state each arm's policy truthfully and in the same words
where the policies agree. The coordinate tool gives the post's one line of
guidance per mode and no advice on naming.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .executor import Executor

MAX_SHOWN = 6000  # characters of output the model sees per call (head and tail kept)

RETURNS = "Returns the printed output, the value of the last expression (as in a REPL) and any error."

FRESH_DOC = (
    "Run Python code. Every call runs in a fresh Python interpreter with an empty working directory: nothing "
    "from earlier calls exists, and nothing this call defines, loads or writes is kept after it returns. " + RETURNS
)
SHARED_DOC = (
    "Run Python code. Every call runs in one Python interpreter that is kept for the whole run and shared by "
    "every request you handle, for every client: names, loaded data and files from earlier calls, including "
    "calls made for other clients, are still there, and whatever this call defines or changes stays for later "
    "calls. " + RETURNS
)
PER_TASK_DOC = (
    "Run Python code. Every call for this client runs in this client's own Python interpreter, kept across all "
    "of this client's requests: names, loaded data and files from this client's earlier calls are still there, "
    "and whatever this call defines or changes stays for this client's later calls. Each client has an "
    "interpreter of its own, which other clients' requests never touch. " + RETURNS
)
MODE_LINES = (
    "- \"stateful\": the code runs in the session, which is kept: whatever it defines or changes stays for later "
    "calls. Use it for work that takes several steps, such as loading data and then analysing it.\n"
    "- \"stateless\": the code runs in a fresh interpreter that is discarded afterwards. Use it for a one-off "
    "check.\n"
    "- \"read_only\": the code runs against a snapshot of the session: it sees everything in the session, and "
    "nothing it defines or changes (names, objects changed in place, files) is kept. Use it for a look or a "
    "what-if that must not change the session."
)
COORD_DOC = (
    "Run Python code at the coordinates you choose.\n\n"
    "mode:\n" + MODE_LINES + " A stateless call takes no session.\n\n"
    "session: the name of a session. A session is a Python interpreter with its own working directory, kept for "
    "the whole run. A stateful call to a name that does not exist yet starts a new, empty session under that "
    "name. Every request you handle, for every client, sees the same sessions; list_sessions shows them and "
    "inspect_session shows what one holds.\n\n" + RETURNS + " Also returns the mode and session used."
)
PER_TASK_MODES_DOC = (
    "Run Python code in this client's own Python interpreter (the session), which is kept across all of this "
    "client's requests; each client has an interpreter of its own, which other clients' requests never touch. "
    "Choose the mode on every call:\n" + MODE_LINES + "\n\n" + RETURNS + " Also returns the mode used."
)
LIST_DOC = "List the sessions that exist, with what each holds (names, types and sizes)."
INSPECT_DOC = ("Show what one session holds: every name, with its type and a short summary (shape, columns and "
               "first rows for tables).")
FINAL_DOC = ("Give the answer to the current request, which finishes it. Call it once, when you have the answer, "
             "in the form the request asks for.")


def _fn(name: str, doc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": doc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


CODE = {"type": "string", "description": "The Python code to run."}
MODE = {"type": "string", "enum": ["stateful", "stateless", "read_only"], "description": "What survives the call."}
SESSION = {"type": "string", "description": "The session's name (for stateful and read_only)."}
FINAL = _fn("final_answer", FINAL_DOC, {"answer": {
    "type": "string",
    "description": "The answer as JSON: a number, or an object mapping each name to a number."}}, ["answer"])


def shown(text: str | None) -> str | None:
    if text is None or len(text) <= MAX_SHOWN:
        return text
    half = MAX_SHOWN // 2
    return f"{text[:half]}\n...[{len(text) - MAX_SHOWN} characters cut]...\n{text[-half:]}"


@dataclass
class Ctx:
    client: str  # the id of the client whose request is being handled
    tag: str  # a unique id for the request, recorded by the executor


class Toolset:
    """The tools an arm offers. call() returns (text for the model, full record)."""

    arm = "base"
    run_doc = ""

    def __init__(self, ex: Executor):
        self.ex = ex

    @property
    def schemas(self) -> list[dict]:
        return [_fn("run_python", self.run_doc, {"code": CODE}, ["code"]), FINAL]

    def coords(self, args: dict, ctx: Ctx) -> tuple[str, str | None]:
        raise NotImplementedError

    async def call(self, name: str, args: dict, ctx: Ctx) -> tuple[str, dict]:
        if name == "run_python":
            mode, session = self.coords(args, ctx)
            res = await self.ex.run(args.get("code", ""), mode, session, tag=ctx.tag)
            return json.dumps(self.view(res)), res
        if name == "list_sessions" and self.lists:
            res = await self.ex.list_sessions()
            return json.dumps(res), res
        if name == "inspect_session" and self.lists:
            res = await self.ex.inspect(args.get("session") or "")
            text = json.dumps(res)
            return shown(text), res
        res = {"error": f"Unknown tool {name!r}.", "error_type": "validation"}
        return json.dumps(res), res

    lists = False
    shows_coords = False

    def view(self, res: dict) -> dict:
        """What the model sees of a run: output, value, error, seconds (and coordinates if it chose them)."""
        if res.get("error_type") == "validation":
            return {k: res[k] for k in ("error", "suggestion") if k in res}
        out: dict[str, Any] = {}
        if self.shows_coords:
            out["mode"] = res.get("mode")
            if res.get("session"):
                out["session"] = res["session"]
                if res.get("created"):
                    out["new_session"] = True
        out["stdout"] = shown(res.get("stdout") or "")
        if res.get("value") is not None:
            out["value"] = shown(res["value"])
        if res.get("error"):
            out["error"] = shown(res["error"])
        out["seconds"] = round(float(res.get("seconds") or 0), 1)
        return out


class Fresh(Toolset):
    arm = "fresh"
    run_doc = FRESH_DOC

    def coords(self, args, ctx):
        return "stateless", None


class Shared(Toolset):
    arm = "shared"
    run_doc = SHARED_DOC

    def coords(self, args, ctx):
        return "stateful", "shared"


class PerTask(Toolset):
    arm = "per_task"
    run_doc = PER_TASK_DOC

    def coords(self, args, ctx):
        return "stateful", ctx.client


class Coordinate(Toolset):
    arm = "coordinate"
    run_doc = COORD_DOC
    lists = True
    shows_coords = True

    @property
    def schemas(self) -> list[dict]:
        return [
            _fn("run_python", self.run_doc, {"code": CODE, "mode": MODE, "session": SESSION}, ["code", "mode"]),
            _fn("list_sessions", LIST_DOC, {}, []),
            _fn("inspect_session", INSPECT_DOC, {"session": {"type": "string", "description": "The session's name."}},
                ["session"]),
            FINAL,
        ]

    def coords(self, args, ctx):
        return args.get("mode") or "", args.get("session") or None


class PerTaskModes(Toolset):
    arm = "per_task_modes"
    run_doc = PER_TASK_MODES_DOC
    shows_coords = True

    @property
    def schemas(self) -> list[dict]:
        return [_fn("run_python", self.run_doc, {"code": CODE, "mode": MODE}, ["code", "mode"]), FINAL]

    def coords(self, args, ctx):
        mode = args.get("mode") or ""
        return mode, (None if mode == "stateless" else ctx.client)

    async def call(self, name, args, ctx):
        if name == "run_python" and args.get("mode") == "read_only" and ("default", ctx.client) not in self.ex.sessions:
            res = {"error": "This client's interpreter has not been started yet, so there is nothing to read.",
                   "error_type": "validation", "suggestion": "Use mode='stateful' to start it."}
            return json.dumps(res), res
        return await super().call(name, args, ctx)

    def view(self, res):
        out = super().view(res)
        out.pop("session", None)
        out.pop("new_session", None)
        return out


# arm name -> (toolset, context design)
ARMS = {
    "fresh": (Fresh, "threads"),
    "shared": (Shared, "threads"),
    "per_task": (PerTask, "threads"),
    "coordinate": (Coordinate, "threads"),
    "long_shared": (Shared, "inbox"),
    "long_coordinate": (Coordinate, "inbox"),
    "per_task_modes": (PerTaskModes, "threads"),
}
